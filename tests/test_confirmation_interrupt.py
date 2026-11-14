
import pytest

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver

from app.agent.state import AgentState
from app.agent.nodes import confirmation_node


def build_confirmation_graph():
    """创建一个专门测试人工确认的 LangGraph。"""

    workflow = StateGraph(AgentState)

    workflow.add_node("confirmation", confirmation_node)

    workflow.add_edge(START, "confirmation")
    workflow.add_edge("confirmation", END)

    # 使用内存 checkpointer 保存暂停状态
    checkpointer = InMemorySaver()

    return workflow.compile(checkpointer=checkpointer)


def build_initial_state(tool="file_writer"):
    """构建测试初始状态。"""

    if tool == "file_writer":
        tool_input = {
            "filename": "approval_test.md",
            "content": "# 人工确认测试\n\n计算结果：20.0",
        }
    else:
        tool_input = {
            "operation": "average",
            "numbers": [10, 20, 30],
        }

    return {
        "messages": [],
        "user_goal": "测试人工确认机制",
        "plan": [
            {
                "step_id": 1,
                "description": "执行测试操作",
                "tool": tool,
                "tool_input": tool_input,
                "status": "pending",
            }
        ],
        "current_step": 0,
        "current_tool": tool,
        "current_tool_input": tool_input,
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
        "status": "running",
    }


def test_confirmation_pause_and_approve():
    """测试工作流暂停并获得批准后恢复。"""

    graph = build_confirmation_graph()

    config = {
        "configurable": {
            "thread_id": "approval-test-001",
        }
    }

    # 第一次执行：应该暂停
    paused_result = graph.invoke(
        build_initial_state(),
        config = config,
    )

    assert "__interrupt__" in paused_result

    interrupt_info = paused_result["__interrupt__"][0]
    payload = interrupt_info.value

    assert payload["tool"] == "file_writer"
    assert "approval_test.md" in payload["request"]

    print("\n测试一通过：File Writer 触发人工暂停")

    # 第二次执行：用户批准
    resumed_result = graph.invoke(
        Command(resume=True),
        config=config,
    )

    assert resumed_result["confirmation_result"] == "approved"
    assert resumed_result["status"] == "running"
    assert resumed_result["needs_confirmation"] is False

    print("\n测试二通过：用户批准后工作流恢复")


def test_confirmation_pause_and_reject():
    """测试用户拒绝后取消操作。"""

    graph = build_confirmation_graph()

    config = {
        "configurable": {
            "thread_id": "approval-test-002",
        }
    }

    # 第一次执行：暂停
    paused_result = graph.invoke(
        build_initial_state(),
        config=config,
    )

    assert "__interrupt__" in paused_result

    print("\n测试三通过：拒绝测试成功触发暂停")

    # 第二次执行：用户拒绝
    result = graph.invoke(
        Command(resume=False),
        config=config,
    )

    assert result["confirmation_result"] == "rejected"
    assert result["status"] == "cancelled"
    assert "用户拒绝" in result["error"]

    print("\n测试四通过：用户拒绝后操作取消")


def test_confirmation_not_required():
    """测试不需要确认的工具直接通过。"""

    graph = build_confirmation_graph()

    config = {
        "configurable": {
            "thread_id": "approval-test-003",
        }
    }

    result = graph.invoke(
        build_initial_state(tool="calculator"),
        config=config,
    )

    assert "__interrupt__" not in result
    assert result["needs_confirmation"] is False
    assert result["confirmation_result"] is None

    print("\n测试五通过：Calculator 不触发人工暂停")


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
