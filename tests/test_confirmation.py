import pytest

from app.agent.confirmation import (
    requires_confirmation,
    build_confirmation_request,
)


def test_calculator_no_confirmation():
    assert requires_confirmation("calculator") is False

    print("\n测试一通过：Calculator 不需要人工确认")


def test_summarizer_no_confirmation():
    assert requires_confirmation("text_summarizer") is False

    print("\n测试二通过：Text Summarizer 不需要人工确认")


def test_file_writer_requires_confirmation():
    assert requires_confirmation("file_writer") is True

    print("\n测试三通过：File Writer 需要人工确认")


def test_build_confirmation_request():
    step = {
        "step_id": 2,
        "description": "生成计算结果报告",
        "tool": "file_writer",
        "tool_input": {
            "filename": "report.md",
            "content": "# 测试报告\n计算结果：20.0",
        },
        "status": "pending",
    }

    request = build_confirmation_request(step)

    assert "file_writer" in request
    assert "report.md" in request
    assert "计算结果：20.0" in request
    assert "请批准或拒绝" in request

    print("\n测试四通过：确认请求生成正确")


def test_confirmation_preview_limit():
    step = {
        "step_id": 1,
        "description": "生成测试报告",
        "tool": "file_writer",
        "tool_input": {
            "filename": "long_report.md",
            "content": "A" * 500,
        },
        "status": "pending",
    }

    request = build_confirmation_request(step)

    assert "内容预览已截断" in request

    print("\n测试五通过：文件预览长度限制正确")


def test_non_confirmation_tool_rejected():
    step = {
        "step_id": 1,
        "description": "执行计算",
        "tool": "calculator",
        "tool_input": {
            "operation": "add",
            "numbers": [10, 20],
        },
        "status": "pending",
    }

    with pytest.raises(ValueError, match="不需要人工确认"):
        build_confirmation_request(step)

    print("\n测试六通过：非确认工具请求被拦截")


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
