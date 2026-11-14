from typing import Literal

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from app.agent.state import AgentState
from app.agent.confirmation import requires_confirmation

from app.agent.nodes import (
    planner_node,
    router_node,
    confirmation_node,
    tool_executor_node,
    reviewer_node,
    finalize_node,
)


def route_after_confirmation(
    state: AgentState,
) -> Literal["tool_executor", "finalize"]:
    """判断人工确认后的执行方向。"""

    status = state.get("status")

    if status in ("failed", "cancelled"):
        return "finalize"

    tool_name = state.get("current_tool", "")

    # 需要确认的工具必须获得明确批准
    if requires_confirmation(tool_name):
        if state.get("confirmation_result") != "approved":
            return "finalize"

    return "tool_executor"


def route_after_executor(
    state: AgentState,
) -> Literal["router", "reviewer", "finalize"]:
    """判断工具执行后的下一步。"""

    status = state.get("status")

    # 可重试异常：返回 Router
    if status == "retrying":
        return "router"

    # 执行失败或取消：结束
    if status in ("failed", "cancelled"):
        return "finalize"

    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)

    # 还有未执行的步骤
    if current_step < len(plan):
        return "router"

    # 所有步骤执行完成
    return "reviewer"


def build_agent_graph():
    """构建主 Agent 工作流。"""

    workflow = StateGraph(AgentState)

    # 注册节点
    workflow.add_node("planner", planner_node)
    workflow.add_node("router", router_node)
    workflow.add_node("confirmation", confirmation_node)
    workflow.add_node("tool_executor", tool_executor_node)
    workflow.add_node("reviewer", reviewer_node)
    workflow.add_node("finalize", finalize_node)

    # 工作流入口
    workflow.add_edge(START, "planner")

    # Planner -> Router -> Confirmation
    workflow.add_edge("planner", "router")
    workflow.add_edge("router", "confirmation")

    # 人工确认后的条件路由
    workflow.add_conditional_edges(
        "confirmation",
        route_after_confirmation,
        {
            "tool_executor": "tool_executor",
            "finalize": "finalize",
        },
    )

    # 工具执行后的条件路由
    workflow.add_conditional_edges(
        "tool_executor",
        route_after_executor,
        {
            "router": "router",
            "reviewer": "reviewer",
            "finalize": "finalize",
        },
    )

    # Reviewer -> Finalizer -> END
    workflow.add_edge("reviewer", "finalize")
    workflow.add_edge("finalize", END)

    # 添加 checkpointer，支持 interrupt 暂停与恢复
    checkpointer = InMemorySaver()

    return workflow.compile(checkpointer=checkpointer)
