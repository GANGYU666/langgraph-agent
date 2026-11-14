from unittest import result

import pytest
from openai import max_retries

from app.agent.nodes import tool_executor_node

@pytest.fixture
def anyio_backend():
    return "asyncio"

def build_state(tool,tool_input):
    return{
        "plan":[
            {
                "step_id":1,
                "description":"执行测试任务",
                "tool":tool,
                "tool_input":tool_input,
                "status":"pending",
            }
        ],
        "current_step":0,
        "current_tool":tool,
        "current_tool_input":tool_input,
    }
@pytest.mark.anyio
async def test_calculator_execution():
    state = build_state(
        "calculator",
        {
            "operation": "average",
            "numbers":[10,20,30],
        },
    )

    result = await tool_executor_node(state)

    assert result["status"] == "running"
    assert result["current_step"] == 1
    assert result["plan"][0]["status"] == "completed"

    output = result["tool_results"][0]["output"]
    assert output["result"] == 20

    assert result["tool_calls"][0]["status"] == "success"

    print("\n测试一通过：Calculator 工具执行")

@pytest.mark.anyio
async def test_summarizer_execution(monkeypatch):
    async def fake_summarizer(text,task="summarize"):
        return {
            "task":task,
            "original_text":text,
            "summary":"这是一段模拟摘要",
        }
    monkeypatch.setattr(
        "app.agent.nodes.text_summarizer",
        fake_summarizer,
    )

    state = build_state(
        "text_summarizer",
        {
            "text":"人工智能正在快速发展",
            "task":"summarize",
        },
    )

    result = await tool_executor_node(state)

    assert result["status"] == "running"
    assert result["tool_results"][0]["output"]["summary"] == "这是一段模拟摘要"

    print("\n测试二通过：Text Summarizer 工具执行")

@pytest.mark.anyio
async def test_file_writer_execution(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "app.tools.file_writer.OUTPUT_DIR",
        tmp_path,
    )

    monkeypatch.setattr(
        "app.tools.file_writer.PROJECT_ROOT",
        tmp_path.parent,
    )

    state = build_state(
        "file_writer",
        {
            "filename": "executor_test.md",
            "content": "# Agent 执行测试\n\n测试成功。",
        },
    )

    result = await tool_executor_node(state)

    assert result["status"] == "running"
    assert result["plan"][0]["status"] == "completed"

    output = result["tool_results"][0]["output"]

    assert output["success"] is True
    assert (tmp_path / "executor_test.md").exists()

    print("\n测试三通过：File Writer 工具执行")


@pytest.mark.anyio
async def test_tool_execution_failure():
    state = build_state(
        "calculator",
        {
            "operation": "divide",
            "numbers": [10, 0],
        },
    )

    result = await tool_executor_node(state)

    assert result["status"] == "failed"
    assert result["plan"][0]["status"] == "failed"
    assert "除数不能为零" in result["error"]

    print("\n测试四通过：工具执行异常处理")


@pytest.mark.anyio
async def test_multi_tool_result_transfer(tmp_path, monkeypatch):
    """
    测试 Calculator 的结果能否传递给 File Writer。
    """

    # 将报告输出目录设置为临时目录
    monkeypatch.setattr(
        "app.tools.file_writer.OUTPUT_DIR",
        tmp_path,
    )

    monkeypatch.setattr(
        "app.tools.file_writer.PROJECT_ROOT",
        tmp_path.parent,
    )

    # 当前步骤是第二步：生成报告
    state = build_state(
        "file_writer",
        {
            "filename": "calculation_report.md",
            "content": "整理前面的计算过程和结果",
        },
    )

    state["plan"][0]["step_id"] = 2

    # 模拟第一步 Calculator 已经执行成功
    state["tool_results"] = [
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
    ]

    # 执行当前步骤
    result = await tool_executor_node(state)

    # 检查工具执行状态
    assert result["status"] == "running"

    # 检查文件是否生成
    report_path = tmp_path / "calculation_report.md"

    assert report_path.exists()

    # 读取实际生成的 Markdown 报告
    content = report_path.read_text(encoding="utf-8")

    # 检查上一工具的结果是否成功传递
    assert "计算结果：20.0" in content
    assert "10、20、30" in content
    assert "前序步骤执行结果" in content

    # 检查实际执行参数是否包含历史结果
    actual_arguments = result["tool_calls"][0]["arguments"]

    assert "计算结果：20.0" in actual_arguments["content"]

    print("\n测试五通过：多工具结果传递")

@pytest.mark.anyio
async def test_retryable_error():
    """ 测试可重试错误 """

    async def fake_summarizer(text,task="summarize"):
        raise ConnectionError("模拟网络连接失败")

    from app.agent import nodes

    # 模拟摘要工具发生网络异常
    original_tool = nodes.text_summarizer
    nodes.text_summarizer = fake_summarizer

    try:
        state = build_state(
            "text_summarizer",
            {
                "text": "测试文本",
                "task": "summarize",
            },
        )
        state["retry_count"] = 0
        state["max_retries"] = 2

        result = await tool_executor_node(state)

        assert result["status"] == "retrying"
        assert result["retry_count"] == 1
        assert result["plan"][0]["status"] == "pending"

        assert "准备进行第 1 次重试" in result["error"]

        print("\n测试六通过：网络异常触发自动重试")
    finally:
        nodes.text_summarizer = original_tool

@pytest.mark.anyio
async def test_non_retryable_error():
    """ 测试不可重试错误 """

    state = build_state(
        "calculator",
        {
            "operation": "divide",
            "numbers": [10, 0],
        },
    )

    state["retry_count"] = 0
    state[max_retries] = 2

    result = await tool_executor_node(state)

    assert result["status"] == "failed"
    assert result["retry_count"] == 0
    assert result["plan"][0]["status"] == "failed"

    assert "不支持自动重试" in result["error"]

    print("\n测试七通过：业务错误不会触发重试")


@pytest.mark.anyio
async def test_max_retries_exceeded():
    """测试达到最大重试次数后停止。"""

    from app.agent import nodes

    async def fake_summarizer(text, task="summarize"):
        raise ConnectionError("模拟网络连接失败")

    original_tool = nodes.text_summarizer
    nodes.text_summarizer = fake_summarizer

    try:
        state = build_state(
            "text_summarizer",
            {
                "text": "测试文本",
                "task": "summarize",
            },
        )

        state["retry_count"] = 2
        state["max_retries"] = 2

        result = await tool_executor_node(state)

        assert result["status"] == "failed"
        assert result["retry_count"] == 2
        assert result["plan"][0]["status"] == "failed"
        assert "已达到最大重试次数" in result["error"]

        print("\n测试八通过：达到重试上限后停止")

    finally:
        nodes.text_summarizer = original_tool

if __name__ == "__main__":
    pytest.main([__file__, "-s"])