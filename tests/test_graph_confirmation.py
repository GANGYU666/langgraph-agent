
import pytest

from langgraph.types import Command

from app.agent.graph import build_agent_graph


@pytest.fixture
def anyio_backend():
    return "asyncio"


def build_initial_state():
    return {
        "messages": [],
        "user_goal": "生成一份人工确认测试报告",
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


def fake_planner(state):
    return {
        "plan": [
            {
                "step_id": 1,
                "description": "生成测试报告",
                "tool": "file_writer",
                "tool_input": {
                    "filename": "approval_test.md",
                    "content": "# 人工审批测试\n\n审批成功。",
                },
                "status": "pending",
            }
        ],
        "current_step": 0,
        "iteration_count": 1,
        "status": "running",
        "error": "",
    }


@pytest.mark.anyio
async def test_main_graph_approval(monkeypatch, tmp_path):
    """测试主工作流暂停并在批准后生成文件。"""

    monkeypatch.setattr(
        "app.agent.graph.planner_node",
        fake_planner,
    )

    monkeypatch.setattr(
        "app.tools.file_writer.OUTPUT_DIR",
        tmp_path,
    )

    monkeypatch.setattr(
        "app.tools.file_writer.PROJECT_ROOT",
        tmp_path.parent,
    )

    graph = build_agent_graph()

    config = {
        "configurable": {
            "thread_id": "main-approval-test",
        },
        "recursion_limit": 20,
    }

    # 第一次执行：应该暂停
    paused = await graph.ainvoke(
        build_initial_state(),
        config=config,
    )

    assert "__interrupt__" in paused

    assert not (tmp_path / "approval_test.md").exists()

    print("\n测试一通过：主工作流成功暂停")

    # 用户批准
    result = await graph.ainvoke(
        Command(resume=True),
        config=config,
    )

    assert result["status"] == "completed"
    assert result["confirmation_result"] == "approved"
    assert result["review_passed"] is True

    assert (tmp_path / "approval_test.md").exists()

    print("\n测试二通过：批准后文件生成成功")
    print("\n测试三通过：Reviewer 审核通过")


@pytest.mark.anyio
async def test_main_graph_rejection(monkeypatch, tmp_path):
    """测试用户拒绝后不执行文件生成。"""

    monkeypatch.setattr(
        "app.agent.graph.planner_node",
        fake_planner,
    )

    monkeypatch.setattr(
        "app.tools.file_writer.OUTPUT_DIR",
        tmp_path,
    )

    graph = build_agent_graph()

    config = {
        "configurable": {
            "thread_id": "main-rejection-test",
        },
        "recursion_limit": 20,
    }

    # 第一次执行：暂停
    paused = await graph.ainvoke(
        build_initial_state(),
        config=config,
    )

    assert "__interrupt__" in paused

    # 用户拒绝
    result = await graph.ainvoke(
        Command(resume=False),
        config=config,
    )

    assert result["status"] == "cancelled"
    assert result["confirmation_result"] == "rejected"
    assert "任务已取消" in result["final_answer"]

    # 文件不应该生成
    assert not (tmp_path / "approval_test.md").exists()

    # Reviewer 不应该审核未执行的任务
    assert result["review_passed"] is False

    print("\n测试四通过：拒绝后任务取消")
    print("\n测试五通过：拒绝后没有生成文件")


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
