import pytest

from app.agent.nodes import router_node


def build_state(
    tool="calculator",
    tool_input=None,
    current_step=0,
    status="pending",
):
    if tool_input is None:
        tool_input = {
            "operation": "average",
            "numbers": [10, 20, 30],
        }

    return {
        "plan": [
            {
                "step_id": 1,
                "description": "计算平均值",
                "tool": tool,
                "tool_input": tool_input,
                "status": status,
            }
        ],
        "current_step": current_step,
    }


def test_router_calculator():
    state = build_state()
    result = router_node(state)

    assert result["current_tool"] == "calculator"
    assert result["current_tool_input"]["operation"] == "average"

    print("\n测试一通过：Calculator 路由")


def test_router_summarizer():
    state = build_state(
        tool="text_summarizer",
        tool_input={
            "text": "人工智能正在快速发展",
            "task": "summarize",
        },
    )

    result = router_node(state)

    assert result["current_tool"] == "text_summarizer"
    print("\n测试二通过：Summarizer 路由")


def test_router_file_writer():
    state = build_state(
        tool="file_writer",
        tool_input={
            "filename": "report.md",
            "content": "测试报告",
        },
    )

    result = router_node(state)

    assert result["current_tool"] == "file_writer"
    print("\n测试三通过：File Writer 路由")


def test_router_reject_unknown_tool():
    state = build_state(tool="unknown_tool")

    with pytest.raises(ValueError, match="不允许调用"):
        router_node(state)

    print("\n测试四通过：非法工具拦截")


def test_router_reject_invalid_step():
    state = build_state(current_step=5)

    with pytest.raises(ValueError, match="当前步骤索引无效"):
        router_node(state)

    print("\n测试五通过：非法步骤拦截")


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
