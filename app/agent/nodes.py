from app.tools.calculator import calculator
from app.tools.text_summarizer import text_summarizer
from app.tools.file_writer import file_writer
from langchain_core.messages import SystemMessage, HumanMessage

from app.agent.state import AgentState
from app.schemas.agent import PlannerOutput
from app.services.llm_service import get_chat_model
import math
from pathlib import Path
from langgraph.types import interrupt

from app.agent.confirmation import (
    requires_confirmation,
    build_confirmation_request,
)
from time import perf_counter

from app.utils.logger import get_logger

logger = get_logger("agent_planner")
router_logger = get_logger("agent_router")
tool_executor_logger = get_logger("agent_tool_executor")
reviewer_logger = get_logger("agent_reviewer")
finalizer_logger = get_logger("agent_finalizer")
confirmation_logger = get_logger("agent_confirmation")


async def planner_node(state: AgentState) -> dict:
    """
    Planner 节点：
    接收用户目标，调用 DeepSeek 生成结构化任务计划。
    """

    # 1. 获取用户任务
    user_goal = state.get("user_goal", "").strip()
    thread_id = state.get("thread_id", "unknown")

    if not user_goal:
        logger.warning("Planner 收到空任务")
        raise ValueError("用户任务不能为空")

    logger.info(
        "Planner 开始规划 | thread_id=%s | goal_length=%d",
        thread_id,
        len(user_goal)
    )

    # 2. 获取 DeepSeek 模型
    model = get_chat_model()

    # 3. 使用结构化输出，要求模型按照 Pydantic 模型返回结果
    planner_model = model.with_structured_output(
        PlannerOutput,
        method="json_mode"
    )

    # 4. 定义 Planner 的系统提示词
    system_prompt = """
你是一个智能任务规划 Agent，负责将用户的复杂任务拆解成
可以由工具执行的步骤。

你必须返回 JSON 格式的数据。

目前系统只允许使用以下三个工具：

1. calculator
作用：执行数学计算。
参数格式：
{
    "operation": "average",
    "numbers": [10, 20, 30]
}
operation 可以是：
add、subtract、multiply、divide、average。

2. text_summarizer
作用：对用户提供的文本进行摘要。
参数格式：
{
    "text": "需要处理的文本",
    "task": "summarize"
}

3. file_writer
作用：生成 Markdown 文件。
参数格式：
{
    "filename": "report.md",
    "content": "报告内容或报告编写要求"
}

任务规划规则：

1. 根据用户目标拆解任务，按照执行顺序排列。
2. 每个步骤只能选择上述三个工具之一。
3. 不允许编造用户没有提供的数字或文本。
4. 工具参数必须符合对应工具的用途。
5. 如果报告依赖前面步骤的结果，
   content 应描述报告需要包含哪些执行结果，
   后续执行器负责注入真实结果。
6. step_id 从 1 开始，按顺序递增。
7. 每个计划最多包含 5 个步骤。
8. 不要执行工具，只生成任务计划。
9. 不要输出额外解释，只返回符合 JSON 结构的数据。

JSON 结构示例：

{
    "steps": [
        {
            "step_id": 1,
            "description": "计算三个数字的平均值",
            "tool": "calculator",
            "tool_input": {
                "operation": "average",
                "numbers": [10, 20, 30]
            }
        }
    ]
}
"""

    # 5. 构造模型输入消息
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"请规划以下任务：{user_goal}")
    ]

    # 6. 异步调用 DeepSeek
    try:
        result = await planner_model.ainvoke(messages)

    except Exception:
        logger.exception(
            "Planner 调用 DeepSeek 模型失败 | thread_id=%s",
            thread_id
        )
        raise

    # 7. 确保结果符合 Pydantic 数据模型
    if not isinstance(result, PlannerOutput):
        result = PlannerOutput.model_validate(result)

    # 8. 将 Pydantic 对象转换为 LangGraph State 所需的数据
    plan = []

    for step in result.steps:
        step_data = step.model_dump()

        # 每个步骤初始状态为 pending
        step_data["status"] = "pending"

        plan.append(step_data)

    logger.info(
        "Planner 规划完成 | thread_id=%s | steps=%d | tools=%s",
        thread_id,
        len(plan),
        ",".join(step["tool"] for step in plan)
    )

    # 9. 返回需要更新的 AgentState 字段
    return {
        "plan": plan,
        "current_step": 0,
        "iteration_count": state.get("iteration_count", 0) + 1,
        "status": "running",
        "error": ""
    }


def router_node(state: AgentState) -> dict:
    """根据当前计划选择工具，并完成安全校验。"""

    allowed_tools = {
        "calculator",
        "text_summarizer",
        "file_writer",
    }

    #获取会话 ID,用于追踪同一个任务的完整执行任务
    thread_id = state.get("thread_id","unknown")

    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)

    # 记录 Router 开始路由
    router_logger.info(
        "Router 开始路由 | thread_id=%s | step_index=%s | total_steps=%d",
        thread_id,
        current_step,
        len(plan)
    )

    # 检查任务计划是否存在
    if not plan:
        router_logger.warning(
            "Router 路由失败：任务计划为空 | thread_id=%s",
            thread_id
        )
        raise ValueError("任务计划为空，无法进行工具路由")

    # 检查当前步骤索引是否合法
    if current_step < 0 or current_step >= len(plan):
        router_logger.warning(
            "Router 路由失败：步骤索引无效 | thread_id=%s | step_index=%s | total_steps=%d",
            thread_id,
            current_step,
            len(plan)
        )
        raise ValueError(
            f"当前步骤索引无效：{current_step}，计划总步骤数：{len(plan)}"
        )

    # 获取当前步骤
    step = plan[current_step]

    tool_name = step.get("tool")
    tool_input = step.get("tool_input", {})
    step_status = step.get("status")

    # 检查工具白名单
    if tool_name not in allowed_tools:
        router_logger.warning(
            "Router 路由失败：工具不在白名单中 | thread_id=%s | tool=%s",
            thread_id,
            tool_name
        )
        raise ValueError(f"不允许调用该工具：{tool_name}")

    # 检查工具参数
    if not isinstance(tool_input, dict):
        router_logger.warning(
            "Router 路由失败：工具参数类型错误 | thread_id=%s | tool=%s",
            thread_id,
            tool_name
        )
        raise ValueError("工具参数必须是字典类型")

    # 检查步骤状态
    if step_status != "pending":
        router_logger.warning(
            "Router 路由失败：步骤状态异常 | thread_id=%s | step_index=%s | status=%s",
            thread_id,
            current_step,
            step_status
        )
        raise ValueError(
            f"当前步骤状态不正确：{step_status}，预期为 pending"
        )

    # 记录工具选择结果
    router_logger.info(
        "Router 选择工具 | thread_id=%s | step_index=%s | tool=%s | status=%s",
        thread_id,
        current_step,
        tool_name,
        step_status
    )

    return {
        "current_tool": tool_name,
        "current_tool_input": tool_input,
    }


def _format_previous_tool_results(tool_results: list[dict]) -> str:
    """
    将前面步骤的工具执行结果转换为可读的 Markdown 文本。
    """

    sections = []

    for item in tool_results:
        step_id = item.get("step_id")
        tool_name = item.get("tool_name")
        output = item.get("output", {})

        if not isinstance(output, dict):
            output = {}

        if tool_name == "calculator":
            numbers = output.get("numbers", [])
            numbers_text = "、".join(
                str(number) for number in numbers
            )

            section = (
                f"### 步骤 {step_id}：数学计算\n"
                f"- 运算类型：{output.get('operation')}\n"
                f"- 输入数字：{numbers_text}\n"
                f"- 计算结果：{output.get('result')}"
            )

        elif tool_name == "text_summarizer":
            section = (
                f"### 步骤 {step_id}：文本摘要\n"
                f"{output.get('summary', '')}"
            )

        elif tool_name == "file_writer":
            section = (
                f"### 步骤 {step_id}：文件生成\n"
                f"- 文件名称：{output.get('filename', '')}\n"
                f"- 文件路径：{output.get('file_path', '')}"
            )

        else:
            section = f"### 步骤 {step_id}\n{output}"

        sections.append(section)

    return "\n\n".join(sections)

def _is_retryable_error(exc: Exception) -> bool:
    """
    判断异常是否属于可以重试的错误。

    网络异常、超时、API限流和服务端错误允许重试。
    参数校验等业务错误不允许重试。
    """

    retryable_error_names = {
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
        "WriteTimeout",
        "PoolTimeout",
        "ReadError",
        "WriteError",
        "RemoteProtocolError",
        "NetworkError",
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "InternalServerError",
    }

    # 获取异常类及其父类的名称
    error_class_names = {
        cls.__name__
        for cls in type(exc).__mro__
    }

    # 常见 Python 网络和超时异常
    if isinstance(exc,(TimeoutError,ConnectionError)):
        return True

    # 识别HTTP客户端和OpenAI SDK 的异常
    return bool(error_class_names & retryable_error_names)

async def tool_executor_node(state: AgentState) -> dict:
    """
    根据 Router 选择的工具执行任务，
    并将执行结果写入 AgentState。
    """

    allowed_tools = {
        "calculator",
        "text_summarizer",
        "file_writer",
    }

    # 获取会话 ID，用于追踪任务执行日志
    thread_id = state.get("thread_id", "unknown")

    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)

    # 1. 检查当前步骤
    if not plan:
        raise ValueError("任务计划为空")

    if current_step < 0 or current_step >= len(plan):
        raise ValueError("当前步骤索引无效")

    step = plan[current_step]

    # 2. 获取 Router 选择的工具
    tool_name = state.get("current_tool")
    tool_input = state.get("current_tool_input")

    # 3. 安全校验
    if tool_name not in allowed_tools:
        raise ValueError(f"不允许执行该工具：{tool_name}")

    if tool_name != step.get("tool"):
        raise ValueError("Router 选择的工具与任务计划不一致")

    if not isinstance(tool_input, dict):
        raise ValueError("工具参数必须是字典类型")

    if tool_input != step.get("tool_input"):
        raise ValueError("工具参数与任务计划不一致")

    if step.get("status") != "pending":
        raise ValueError("当前任务步骤不是待执行状态")

    # 4. 标记当前步骤为执行中
    updated_plan = [dict(item) for item in plan]
    updated_plan[current_step]["status"] = "running"

    step_id = step["step_id"]

    # 创建本次实际执行参数副本
    execution_input = dict(tool_input)

    # 如果当前执行的是 File Writer，
    # 则自动注入前面步骤产生的真实结果
    if tool_name == "file_writer":

        previous_results = state.get("tool_results", [])

        previous_results_text = _format_previous_tool_results(
            previous_results
        )

        if previous_results_text:
            original_content = execution_input.get("content", "")

            execution_input["content"] = (
                "# 任务执行报告\n\n"
                "## 任务说明\n"
                f"{original_content}\n\n"
                "## 前序步骤执行结果\n\n"
                f"{previous_results_text}\n"
            )

    # 5. 执行对应工具
    execution_started_at = perf_counter()

    tool_executor_logger.info(
        "Tool Executor 开始执行 | thread_id=%s | step_index=%s | step_id=%s | tool=%s",
        thread_id,
        current_step,
        step_id,
        tool_name
    )

    try:
        if tool_name == "calculator":
            result = calculator(**tool_input)

        elif tool_name == "text_summarizer":
            result = await text_summarizer(**tool_input)

        elif tool_name == "file_writer":
            result = file_writer(**execution_input)

    # 6. 执行成功
        updated_plan[current_step]["status"] = "completed"

        elapsed_ms = (perf_counter() - execution_started_at) * 1000

        tool_executor_logger.info(
            "Tool Executor 执行成功 | thread_id=%s | step_index=%s | step_id=%s | tool=%s | elapsed_ms=%.2f",
            thread_id,
            current_step,
            step_id,
            tool_name,
            elapsed_ms
        )

        return {
            "plan": updated_plan,
            "current_step": current_step + 1,
            "current_tool": "",
            "current_tool_input": {},
            "retry_count": 0,
            "tool_calls": [
                {
                    "step_id": step_id,
                    "tool_name": tool_name,
                    "arguments": execution_input,
                    "status": "success",
                }
            ],
            "tool_results": [
                {
                    "step_id": step_id,
                    "tool_name": tool_name,
                    "output": result,
                    "error": "",
                }
            ],
            "status": "running",
            "error": "",
        }

    except Exception as exc:
        elapsed_ms = (perf_counter() - execution_started_at) * 1000
        # 1. 获取重试配置

        retry_count = state.get("retry_count", 0)

        max_retries = state.get("max_retries", 2)

        # 校验重试配置

        if (

                isinstance(retry_count, bool)

                or not isinstance(retry_count, int)

                or retry_count < 0

        ):
            retry_count = 0

        if (

                isinstance(max_retries, bool)

                or not isinstance(max_retries, int)

                or max_retries < 0

        ):
            max_retries = 2

        # 2. 判断错误是否允许重试

        retryable = _is_retryable_error(exc)

        # retry_count 是已经进行过的额外重试次数

        can_retry = (

                retryable

                and retry_count < max_retries

        )

        # 记录工具失败及重试决策
        if can_retry:
            tool_executor_logger.warning(
                "Tool Executor 执行失败，准备重试 | "
                "thread_id=%s | step_index=%s | step_id=%s | tool=%s | "
                "retry=%d/%d | error_type=%s | elapsed_ms=%.2f",
                thread_id,
                current_step,
                step_id,
                tool_name,
                retry_count + 1,
                max_retries,
                type(exc).__name__,
                elapsed_ms
            )

        else:
            tool_executor_logger.error(
                "Tool Executor 最终执行失败 | "
                "thread_id=%s | step_index=%s | step_id=%s | tool=%s | "
                "retryable=%s | retry_count=%d/%d | "
                "error_type=%s | elapsed_ms=%.2f",
                thread_id,
                current_step,
                step_id,
                tool_name,
                retryable,
                retry_count,
                max_retries,
                type(exc).__name__,
                elapsed_ms
            )
        # 3. 根据重试结果更新计划状态

        if can_retry:

            updated_plan[current_step]["status"] = "pending"

            next_retry_count = retry_count + 1

            next_status = "retrying"

            error_message = (

                f"{type(exc).__name__}: {exc}；"

                f"准备进行第 {next_retry_count} 次重试"

            )


        else:

            updated_plan[current_step]["status"] = "failed"

            next_retry_count = retry_count

            next_status = "failed"

            if retryable:

                error_message = (

                    f"{type(exc).__name__}: {exc}；"

                    f"已达到最大重试次数 {max_retries}"

                )

            else:

                error_message = (

                    f"{type(exc).__name__}: {exc}；"

                    "该错误不支持自动重试"

                )

        # 4. 返回失败记录和重试状态

        return {

            "plan": updated_plan,

            "retry_count": next_retry_count,

            "current_tool": "",

            "current_tool_input": {},

            "tool_calls": [

                {

                    "step_id": step_id,

                    "tool_name": tool_name,

                    "arguments": execution_input,

                    "status": "failed",

                }

            ],

            "tool_results": [

                {

                    "step_id": step_id,

                    "tool_name": tool_name,

                    "output": None,

                    "error": error_message,

                }

            ],

            "status": next_status,

            "error": f"{tool_name} 执行失败：{error_message}",

        }

def finalize_node(state: AgentState) -> dict:
    """
    整理工具执行结果，生成 Agent 最终回答。
    """

    # 获取会话 ID，用于追踪任务最终处理日志
    thread_id = state.get("thread_id", "unknown")

    finalizer_logger.info(
        "Finalizer 开始整理最终结果 | thread_id=%s | status=%s",
        thread_id,
        state.get("status", "unknown")
    )

    # 用户拒绝操作，返回取消结果
    if state.get("status") == "cancelled":
        error = state.get("error", "用户取消了操作")

        finalizer_logger.info(
            "Finalizer 结束任务 | thread_id=%s | status=cancelled",
            thread_id
        )

        return {
            "status": "cancelled",
            "final_answer": f"任务已取消：{error}",
        }

    # 1. 检查是否执行失败
    if state.get("status") == "failed":
        error = state.get("error", "未知错误")

        finalizer_logger.error(
            "Finalizer 结束任务 | thread_id=%s | status=failed",
            thread_id
        )

        return {
            "status": "failed",
            "final_answer": f"任务执行失败：{error}",
        }

    # 2. 检查所有步骤是否已经完成
    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)

    if current_step < len(plan):
        finalizer_logger.warning(
            "Finalizer 无法生成最终回答 | thread_id=%s | current_step=%d | total_steps=%d",
            thread_id,
            current_step,
            len(plan)
        )
        raise ValueError("仍有任务步骤未执行，不能生成最终回答")

    if any(step.get("status") != "completed" for step in plan):
        finalizer_logger.warning(
            "Finalizer 无法生成最终回答 | thread_id=%s | 存在未完成的任务步骤",
            thread_id
        )
        raise ValueError("存在未完成的任务步骤")

    # 3. 获取所有工具执行结果
    tool_results = state.get("tool_results", [])

    result_lines = []

    for item in tool_results:
        step_id = item.get("step_id")
        tool_name = item.get("tool_name")
        output = item.get("output")

        if not isinstance(output, dict):
            output = {}

        # 根据工具类型整理结果
        if tool_name == "calculator":
            description = (
                f"计算结果：{output.get('result')}"
            )

        elif tool_name == "text_summarizer":
            description = (
                f"摘要内容：{output.get('summary', '')}"
            )

        elif tool_name == "file_writer":
            description = (
                f"报告已生成：{output.get('filename', '')}，"
                f"保存路径：{output.get('file_path', '')}"
            )

        else:
            description = str(output)

        result_lines.append(
            f"{step_id}. {description}"
        )

    # 4. 生成最终回答
    if result_lines:
        final_answer = (
            "任务已完成。\n\n"
            "执行结果：\n"
            + "\n".join(result_lines)
        )
    else:
        final_answer = "任务已完成，但没有可展示的工具执行结果。"

    finalizer_logger.info(
        "Finalizer 整理完成 | thread_id=%s | status=completed | "
        "steps=%d | results=%d | answer_length=%d",
        thread_id,
        len(plan),
        len(tool_results),
        len(final_answer)
    )

    return {
        "status": "completed",
        "final_answer": final_answer,
    }

def reviewer_node(state: AgentState) -> dict:
    """
    审核 Agent 的任务执行结果。
    检查：
    1. 任务步骤是否全部完成
    2. 工具调用记录是否完整
    3. 工具执行结果是否有效
    4. 文件是否真正生成
    """

    # 获取会话 ID，用于追踪任务审核日志
    thread_id = state.get("thread_id", "unknown")

    # 统一处理审核失败
    def reject(reason: str) -> dict:
        reviewer_logger.warning(
            "Reviewer 审核未通过 | thread_id=%s | reason=%s",
            thread_id,
            reason
        )

        return {
            "review_passed": False,
            "review_feedback": reason,
            "status": "failed",
            "error": f"任务审核未通过：{reason}",
        }

    # 1. 获取执行状态
    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)
    tool_calls = state.get("tool_calls", [])
    tool_results = state.get("tool_results", [])

    reviewer_logger.info(
        "Reviewer 开始审核 | thread_id=%s | plan_steps=%d | tool_calls=%d | tool_results=%d",
        thread_id,
        len(plan),
        len(tool_calls),
        len(tool_results)
    )

    # 2. 检查任务计划
    if not plan:
        return reject("任务计划为空")

    if current_step != len(plan):
        return reject("仍存在未执行的任务步骤")

    if any(step.get("status") != "completed" for step in plan):
        return reject("存在未完成的任务步骤")

    # 3. 建立步骤与执行记录的对应关系
    calls_by_step = {
        item.get("step_id"): item
        for item in tool_calls
    }

    results_by_step = {
        item.get("step_id"): item
        for item in tool_results
    }

    # 4. 逐步检查执行结果
    for step in plan:
        step_id = step["step_id"]
        tool_name = step["tool"]

        call = calls_by_step.get(step_id)
        result = results_by_step.get(step_id)

        # 检查工具调用记录
        if not call:
            return reject(f"步骤 {step_id} 缺少工具调用记录")

        if call.get("status") != "success":
            return reject(f"步骤 {step_id} 工具调用未成功")

        if call.get("tool_name") != tool_name:
            return reject(f"步骤 {step_id} 工具调用记录不匹配")

        # 检查工具执行结果
        if not result:
            return reject(f"步骤 {step_id} 缺少工具执行结果")

        if result.get("error"):
            return reject(f"步骤 {step_id} 存在执行错误")

        if result.get("tool_name") != tool_name:
            return reject(f"步骤 {step_id} 执行结果与计划不匹配")

        output = result.get("output")

        if not isinstance(output, dict):
            return reject(f"步骤 {step_id} 返回结果格式不正确")

        # 5. 根据不同工具进行专项检查
        if tool_name == "calculator":
            value = output.get("result")

            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                return reject(f"步骤 {step_id} 计算结果无效")

        elif tool_name == "text_summarizer":
            summary = output.get("summary")

            if not isinstance(summary, str) or not summary.strip():
                return reject(f"步骤 {step_id} 摘要内容为空")

        elif tool_name == "file_writer":
            if output.get("success") is not True:
                return reject(f"步骤 {step_id} 文件生成失败")

            file_path = output.get("file_path")

            if not file_path or not Path(file_path).is_file():
                return reject(f"步骤 {step_id} 文件不存在")

        else:
            return reject(f"步骤 {step_id} 使用了未知工具")

    # 6. 全部检查通过
    feedback = (
        f"审核通过：共检查 {len(plan)} 个任务步骤，"
        "工具调用、执行结果及必要的文件均符合要求。"
    )

    reviewer_logger.info(
        "Reviewer 审核通过 | thread_id=%s | checked_steps=%d",
        thread_id,
        len(plan)
    )

    return {
        "review_passed": True,
        "review_feedback": feedback,
        "status": "running",
        "error": "",
    }


def confirmation_node(state: AgentState) -> dict:
    """
    人工确认节点。
    需要确认的工具暂停工作流，
    等待用户批准或拒绝。
    """

    # 1. 获取会话 ID 和当前工具
    thread_id = state.get("thread_id", "unknown")
    tool_name = state.get("current_tool", "")
    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)

    confirmation_logger.info(
        "Confirmation 开始检查 | thread_id=%s | step_index=%s | tool=%s",
        thread_id,
        current_step,
        tool_name
    )

    # 2. 不需要确认的工具直接通过
    if not requires_confirmation(tool_name):
        confirmation_logger.info(
            "Confirmation 无需人工确认 | thread_id=%s | step_index=%s | tool=%s",
            thread_id,
            current_step,
            tool_name
        )

        return {
            "needs_confirmation": False,
            "confirmation_request": "",
            "confirmation_result": None,
        }

    # 3. 检查当前任务步骤
    if current_step < 0 or current_step >= len(plan):
        confirmation_logger.error(
            "Confirmation 失败：当前任务步骤无效 | thread_id=%s | step_index=%s | total_steps=%s",
            thread_id,
            current_step,
            len(plan)
        )
        raise ValueError("当前任务步骤无效")

    step = dict(plan[current_step])

    # 4. 使用 Router 实际选择的工具和参数
    step["tool"] = tool_name
    step["tool_input"] = state.get("current_tool_input", {})

    # 5. 构建确认信息
    request = build_confirmation_request(step)

    confirmation_logger.info(
        "Confirmation 等待人工审批 | thread_id=%s | step_index=%s | tool=%s",
        thread_id,
        current_step,
        tool_name
    )

    # 6. 暂停工作流，等待用户决定
    # 注意：不要使用 try/except 包裹 interrupt()
    approved = interrupt({
        "type": "tool_confirmation",
        "tool": tool_name,
        "request": request,
        "message": "该操作需要用户授权，是否批准执行？",
    })

    # 7. 只有明确返回 True 才允许执行
    if approved is True:
        confirmation_logger.info(
            "Confirmation 用户已批准 | thread_id=%s | step_index=%s | tool=%s",
            thread_id,
            current_step,
            tool_name
        )

        return {
            "needs_confirmation": False,
            "confirmation_request": request,
            "confirmation_result": "approved",
            "status": "running",
            "error": "",
        }

    # 8. 用户拒绝或返回了无效审批结果
    confirmation_logger.warning(
        "Confirmation 用户拒绝或审批结果无效 | thread_id=%s | step_index=%s | tool=%s",
        thread_id,
        current_step,
        tool_name
    )

    return {
        "needs_confirmation": False,
        "confirmation_request": request,
        "confirmation_result": "rejected",
        "status": "cancelled",
        "error": "用户拒绝了当前工具操作",
    }



















