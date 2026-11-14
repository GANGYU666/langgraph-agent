from app.agent.nodes import finalize_node
from app.agent.graph import(
    build_agent_graph,
    route_after_executor,
)

def test_graph_compile():
    graph = build_agent_graph()

    assert graph is not None
    assert "reviewer" in graph.get_graph().nodes

    print("\n测试一通过：LangGraph 工作流编译成功,Reviewer 已注册")

def test_route_continue():
    state = {
        "status": "running",
        "plan": [
            {"step_id: 1"},
            {"step_id: 2"},
            {"step_id: 3"},
        ],
        "current_step": 1,
    }

    result = route_after_executor(state)

    assert result == "router"

    print("\n测试二通过：未完成任务返回 Router")


def test_route_finish():
    state = {
        "status": "running",
        "plan": [
            {"step_id": 1},
            {"step_id": 2},
        ],
        "current_step": 2,
    }

    result = route_after_executor(state)

    assert result == "reviewer"

    print("\n测试三通过：全部完成后进入 Reviewer")


def test_route_failure():
    state = {
        "status": "failed",
        "plan": [{"step_id": 1}],
        "current_step": 0,
    }

    result = route_after_executor(state)

    assert result == "finalize"

    print("\n测试四通过：工具失败后进入 Finalizer")


def test_finalize_success():
    state = {
        "status": "running",
        "plan": [
            {"step_id": 1, "status": "completed"}
        ],
        "current_step": 1,
        "tool_results": [
            {
                "step_id": 1,
                "tool_name": "calculator",
                "output": {
                    "operation": "average",
                    "numbers": [10, 20, 30],
                    "result": 20,
                },
                "error": "",
            }
        ],
    }

    result = finalize_node(state)

    assert result["status"] == "completed"
    assert "任务已完成" in result["final_answer"]
    assert "20" in result["final_answer"]

    print("\n测试五通过：成功生成最终回答")


def test_finalize_failure():
    state = {
        "status": "failed",
        "error": "除数不能为零",
        "plan": [],
        "current_step": 0,
        "tool_results": [],
    }

    result = finalize_node(state)

    assert result["status"] == "failed"
    assert "除数不能为零" in result["final_answer"]

    print("\n测试六通过：正确处理失败结果")

def test_route_retrying():
    """测试重试状态是否返回 Router。"""

    state = {
        "status": "retrying",
        "plan": [
            {
                "step_id": 1,
                "tool": "text_summarizer",
                "status": "pending",
            }
        ],
        "current_step": 0,
        "retry_count": 1,
        "max_retries": 2,
    }

    result = route_after_executor(state)

    assert result == "router"

    print("\n测试七通过：重试状态自动返回 Router")

if __name__ == "__main__":
    import pytest
    pytest.main([__file__,"-s"])