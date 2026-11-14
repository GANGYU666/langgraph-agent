from operator import add
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


# 单个任务步骤的数据结构
class PlanStep(TypedDict):
    step_id: int
    description: str
    tool: str
    tool_input: dict[str, Any]
    status: Literal["pending", "running", "completed", "failed"]


# 工具调用记录
class ToolCallRecord(TypedDict):
    step_id: int
    tool_name: str
    arguments: dict[str, Any]
    status: Literal["success", "failed"]


# 工具执行结果
class ToolResultRecord(TypedDict):
    step_id: int
    tool_name: str
    output: Any
    error: str


# Agent 共享状态
class AgentState(TypedDict):

    # 用户与 Agent 的消息
    messages: Annotated[list[AnyMessage], add_messages]

    #当前 Agent 任务的唯一会话标识
    thread_id: str

    # 用户原始任务
    user_goal: str

    # Planner 生成的任务计划
    plan: list[PlanStep]

    # 当前执行步骤，从 0 开始
    current_step: int

    # Router 当前选择的工具及其参数
    current_tool: str
    current_tool_input: dict[str, Any]

    # 工具调用记录，采用追加方式更新
    tool_calls: Annotated[list[ToolCallRecord], add]

    # 工具执行结果，采用追加方式更新
    tool_results: Annotated[list[ToolResultRecord], add]

    # 人机协同
    needs_confirmation: bool
    confirmation_request: str
    confirmation_result: Literal["approved", "rejected"] | None

    # 最终回答
    final_answer: str

    # Reviewer 审核结果
    review_passed: bool
    review_feedback: str

    # 当前步骤已经重试的次数
    retry_count: int

    # 当前步骤允许的最大重试次数
    max_retries: int

    # 错误信息
    error: str

    # 最大循环次数控制
    iteration_count: int

    # 当前任务状态
    status: Literal[
        "pending",
        "running",
        "retrying",
        "waiting_confirmation",
        "completed",
        "failed",
        "cancelled"
    ]
