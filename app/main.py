from uuid import uuid4
from app.agent.graph import build_agent_graph
from fastapi import FastAPI,HTTPException
from app.config import settings
from langgraph.types import Command
from app.schemas.api import AgentRunRequest,AgentConfirmRequest
from app.utils.logger import get_logger


logger = get_logger("agent_api")

# 创建 FastAPI 应用实例
app = FastAPI(
    title=settings.app_name,
    description="基于 LangGraph 的多工具智能任务执行 Agent",
    version="0.1.0",
    debug=settings.debug,
)


# 创建全局Agent工作流
# 只能在这里创建一次，不能在每次 API 请求时重新创建
# 后续人工确认接口需要使用同一个 Graph 和 Checkpointer 恢复任务
agent_graph = build_agent_graph()

def build_initial_state(goal: str,thread_id: str) -> dict:
    """构建 Agent 初始状态。"""

    return{
        "message": [],
        "user_goal": goal,
        "thread_id": thread_id,
        "plan": [],
        "current_step": 0,
        "current_tool": "",
        "current_tool_input": {},
        "tool_calls": [],
        "tool_results": [],
        "needs_confirmation": False,
        "confirmation_request": "",
        "confirmation_result": None,
        "final_answer": "",
        "review_passed": False,
        "review_feedback": "",
        "error": "",
        "iteration_count": 0,
        "retry_count": 0,
        "max_retries": 2,
        "status": "pending",
    }

# 健康检查接口
@app.get("/health", tags=["Health"])
async def health_check():
    """
    检查 FastAPI 服务是否正常运行。
    """
    return {
        "status": "ok",
        "message": "Agent API is running",
        "app_name": settings.app_name,
    }

# 启动 Agent 任务接口
@app.post("/api/agent/run", tags=["Agent"])
async def run_agent(request: AgentRunRequest):
    """接收用户任务并启动 Agent 工作流"""

    # 为本次任务生成唯一会话 ID
    thread_id = str(uuid4())

    logger.info(
        "收到 Agent 任务 | thread_id=%s | goal_length=%d",
        thread_id,
        len(request.goal)
    )

    # LangGraph Checkpointer 使用 thread_id 保存工作流状态
    config = {
        "configurable": {
            "thread_id": thread_id,
        },
        "recursion_limit": 20,
    }

    # 构建初始状态
    initial_state = build_initial_state(
        request.goal,
        thread_id
    )

    try:
        # 执行Agent
        result = await agent_graph.ainvoke(
            initial_state,
            config=config,
        )

    except Exception:
        logger.exception(
            "Agent 执行异常 | thread_id=%s",
            thread_id
        )

        raise HTTPException(
            status_code=500,
            detail="Agent 任务执行失败，请稍后重试。",
        ) from None

    # 检查工作流是否因为人工确认而暂停
    if result.get("__interrupt__"):

        # 获取 LangGraph 中断信息
        interrupts = result["__interrupt__"]

        interrupt_value = (
            getattr(interrupts[0], "value", {})
            if interrupts
            else {}
        )

        # 提取人工确认请求
        if isinstance(interrupt_value, dict):
            confirmation_request = interrupt_value.get("request")
        else:
            confirmation_request = None

        # 设置兜底信息，避免返回空字符串
        if not confirmation_request:
            confirmation_request = "Agent 请求执行需要人工确认的操作"

        logger.info(
            "Agent 任务等待人工确认 | thread_id=%s",
            thread_id
        )

        return {
            "status": "waiting_for_confirmation",
            "requires_confirmation": True,
            "thread_id": thread_id,
            "confirmation_request": confirmation_request,
        }
    logger.info(
        "Agent 任务执行结束 | thread_id=%s | status=%s",
        thread_id,
        result.get("status")
    )
    # 没有发生中断，直接返回执行结果
    return {
        "status": result.get("status"),
        "requires_confirmation": False,
        "thread_id": thread_id,
        "final_answer": result.get("final_answer", ""),
        "review_passed": result.get("review_passed", False),
        "tool_results": result.get("tool_results", []),
        "error": result.get("error", ""),
    }

# 人工确认接口
@app.post("/api/agent/confirm", tags=["Agent"])
async def confirm_agent(request: AgentConfirmRequest):
    """接收用户的批准或拒绝，并恢复 Agent 工作流。"""

    # 使用原任务的 thread_id 找回暂停的工作流
    config = {
        "configurable": {
            "thread_id": request.thread_id,
        },
        "recursion_limit": 20,
    }
    logger.info(
        "收到人工确认 | thread_id=%s | approved=%s",
        request.thread_id,
        request.approved
    )

    # 1. 查询当前会话状态
    try:
        snapshot = await agent_graph.aget_state(config)

    except Exception:
        logger.exception(
            "读取 Agent 会话状态异常 | thread_id=%s",
            request.thread_id
        )

        raise HTTPException(
            status_code=500,
            detail="读取 Agent 会话状态失败，请稍后重试。",
        ) from None

    # 2. 检查会话是否存在
    if snapshot is None or not snapshot.values:
        logger.warning(
            "人工确认失败：会话不存在 | thread_id=%s",
            request.thread_id
        )

        raise HTTPException(
            status_code=404,
            detail="未找到对应的 Agent 会话。",
        )

    # 3. 获取当前仍待处理的中断信息
    pending_interrupts = getattr(snapshot, "interrupts", ())

    # 兼容通过任务信息读取中断的情况
    if not pending_interrupts:
        pending_interrupts = tuple(
            interrupt
            for task in snapshot.tasks
            for interrupt in getattr(task, "interrupts", ())
        )

    # 4. 只允许恢复人工工具确认类型的中断
    has_pending_confirmation = any(
        isinstance(getattr(interrupt, "value", None), dict)
        and interrupt.value.get("type") == "tool_confirmation"
        for interrupt in pending_interrupts
    )

    if not has_pending_confirmation:
        logger.warning(
            "人工确认失败：当前会话没有待处理的确认操作 | thread_id=%s",
            request.thread_id
        )

        raise HTTPException(
            status_code=409,
            detail="当前会话没有待处理的人工确认，可能已经完成或取消。",
        )

    try:
        # 根据用户选择恢复工作流
        result = await agent_graph.ainvoke(
            Command(resume=request.approved),
            config=config,
        )

    except Exception:
        logger.exception(
            "Agent 恢复执行异常 | thread_id=%s",
            request.thread_id
        )

        raise HTTPException(
            status_code=500,
            detail="Agent 恢复执行失败，请稍后重试。",
        ) from None

    # 检查工作流是否再次暂停
    if result.get("__interrupt__"):
        logger.info(
            "Agent 再次等待人工确认 | thread_id=%s",
            request.thread_id
        )

        return {
            "status":"waiting_for_confirmation",
            "requires_confirmation": True,
            "thread_id": request.thread_id,
            "confirmation_request": result.get(
                "confirmation_request",
                "Agent 需要进一步确认",
            ),
        }
    logger.info(
        "人工确认后工作流执行结束 | thread_id=%s | status=%s",
        request.thread_id,
        result.get("status")
    )
    # 返回最终执行结果
    return {
        "status": result.get("status"),
        "approved": request.approved,
        "requires_confirmation": False,
        "thread_id": request.thread_id,
        "final_answer": result.get("final_answer", ""),
        "review_passed": result.get("review_passed", False),
        "tool_results": result.get("tool_results", []),
        "error": result.get("error", ""),
    }