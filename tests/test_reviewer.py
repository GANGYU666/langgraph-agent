import pytest

from app.agent.nodes import reviewer_node


def build_state():
    return {
        "plan": [
            {
                "step_id": 1,
                "description": "计算平均值",
                "tool": "calculator",
                "tool_input": {
                    "operation": "average",
                    "numbers": [10, 20, 30],
                },
                "status": "completed",
            }
        ],
        "current_step": 1,
        "tool_calls": [
            {
                "step_id": 1,
                "tool_name": "calculator",
                "arguments": {
                    "operation": "average",
                    "numbers": [10, 20, 30],
                },
                "status": "success",
            }
        ],
        "tool_results": [
            {
                "step_id": 1,
                "tool_name": "calculator",
                "output": {
                    "operation": "average",
                    "numbers": [10, 20, 30],
                    "result": 20.0,
                },
                "error": "",
            }
        ],
    }


def test_reviewer_success():
    state = build_state()

    result = reviewer_node(state)

    assert result["review_passed"] is True
    assert "审核通过" in result["review_feedback"]

    print("\n测试一通过：任务审核成功")


def test_reviewer_incomplete_plan():
    state = build_state()
    state["current_step"] = 0

    result = reviewer_node(state)

    assert result["review_passed"] is False
    assert "未执行" in result["review_feedback"]

    print("\n测试二通过：未完成步骤拦截")


def test_reviewer_missing_result():
    state = build_state()
    state["tool_results"] = []

    result = reviewer_node(state)

    assert result["review_passed"] is False
    assert "缺少工具执行结果" in result["review_feedback"]

    print("\n测试三通过：缺失结果拦截")


def test_reviewer_invalid_calculation():
    state = build_state()

    state["tool_results"][0]["output"]["result"] = float("nan")

    result = reviewer_node(state)

    assert result["review_passed"] is False
    assert "计算结果无效" in result["review_feedback"]

    print("\n测试四通过：无效计算结果拦截")


def test_reviewer_file_exists(tmp_path):
    state = build_state()

    report_path = tmp_path / "report.md"
    report_path.write_text("# 测试报告", encoding="utf-8")

    state["plan"][0]["tool"] = "file_writer"
    state["tool_calls"][0]["tool_name"] = "file_writer"

    state["tool_results"][0] = {
        "step_id": 1,
        "tool_name": "file_writer",
        "output": {
            "success": True,
            "filename": "report.md",
            "file_path": str(report_path),
            "file_size": report_path.stat().st_size,
        },
        "error": "",
    }

    result = reviewer_node(state)

    assert result["review_passed"] is True

    print("\n测试五通过：文件存在性审核")


def test_reviewer_file_missing(tmp_path):
    state = build_state()

    report_path = tmp_path / "missing.md"

    state["plan"][0]["tool"] = "file_writer"
    state["tool_calls"][0]["tool_name"] = "file_writer"

    state["tool_results"][0] = {
        "step_id": 1,
        "tool_name": "file_writer",
        "output": {
            "success": True,
            "filename": "missing.md",
            "file_path": str(report_path),
            "file_size": 100,
        },
        "error": "",
    }

    result = reviewer_node(state)

    assert result["review_passed"] is False
    assert "文件不存在" in result["review_feedback"]

    print("\n测试六通过：不存在的文件被拦截")


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
