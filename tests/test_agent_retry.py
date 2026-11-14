import pytest
from app.agent.graph import build_agent_graph

@pytest.fixture
def anyio_backend():
    return "asyncio"

@pytest.mark.anyio
async def test_agent_automatic_retry(monkeypatch):
    """
    测试 LangGraph 自动重试：

    第一次：网络异常
    第二次：执行成功
    最终：Reviewer 审核通过
    """

    # 1. 模拟 Planner，避免调用真实大模型
    async def fake_planner(state):
        return {
            "plan": [
                {
                    "step_id": 1,
                    "description": "对文本进行摘要",
                    "tool": "text_summarizer",
                    "tool_input": {
                        "text": "人工智能正在快速发展",
                        "task": "summarize",
                    },
                    "status": "pending",
                }
            ],
            "current_step": 0,
            "iteration_count": 1,
            "status": "running",
            "error": "",
        }

    # 替换 LangGraph 模块中的 Planner 节点
    monkeypatch.setattr(
        "app.agent.graph.planner_node",
        fake_planner,
    )
    # 2. 模拟摘要工具第一次失败、第二次成功
    attempts = {"count": 0}

    async def fake_summarizer(text, task="summarize"):
        attempts["count"] += 1

        if attempts["count"] == 1:
            raise ConnectionError("模拟第一次网络连接失败")

        return {
            "task": task,
            "original_text": text,
            "summary": "人工智能技术正在快速发展。",
        }

    # 替换 nodes.py 中的摘要工具
    monkeypatch.setattr(
        "app.agent.nodes.text_summarizer",
        fake_summarizer,
    )

    # 3. 构造 Agent 初始状态
    initial_state = {
        "messages": [],
        "user_goal": "请总结人工智能的发展情况",
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

    # 4. 创建并执行 LangGraph
    graph = build_agent_graph()

    result = await graph.ainvoke(
        initial_state,
        config={
            "configurable": {
                "thread_id": "agent-retry-success-test",
            },
            "recursion_limit": 20,
        }
    )

    # 5. 检查工具是否真正执行了两次
    assert attempts["count"] == 2

    print("\n测试一通过：工具第一次执行失败")

    # 6. 检查两次工具调用记录
    calls = result["tool_calls"]

    assert len(calls) == 2
    assert calls[0]["status"] == "failed"
    assert calls[1]["status"] == "success"

    print("\n测试二通过：LangGraph 自动重新执行工具")

    # 7. 检查最终任务状态
    assert result["status"] == "completed"
    assert result["current_step"] == 1

    print("\n测试三通过：重试成功后任务继续完成")

    # 8. 检查 Reviewer
    assert result["review_passed"] is True
    assert "审核通过" in result["review_feedback"]

    print("\n测试四通过：Reviewer 审核通过")

    # 9. 检查最终结果
    assert "任务已完成" in result["final_answer"]
    assert "人工智能技术正在快速发展" in result["final_answer"]

    print("\n测试五通过：最终回答正确")

    # 10. 检查重试次数已经重置
    assert result["retry_count"] == 0

    print("\n测试六通过：成功后重试次数重置")

    print("\n========== 自动重试测试结果 ==========")
    print("工具实际执行次数：", attempts["count"])
    print("工具调用状态：", [item["status"] for item in calls])
    print("最终状态：", result["status"])
    print("Reviewer：", result["review_feedback"])
    print(result["final_answer"])


@pytest.mark.anyio
async def test_agent_stops_after_max_retries(monkeypatch):
    """
    测试连续失败达到最大重试次数后，
    LangGraph 能否安全结束任务。
    """

    # 1. 模拟 Planner
    async def fake_planner(state):
        return {
            "plan": [
                {
                    "step_id": 1,
                    "description": "对文本进行摘要",
                    "tool": "text_summarizer",
                    "tool_input": {
                        "text": "测试文本",
                        "task": "summarize",
                    },
                    "status": "pending",
                }
            ],
            "current_step": 0,
            "iteration_count": 1,
            "status": "running",
            "error": "",
        }

    monkeypatch.setattr(
        "app.agent.graph.planner_node",
        fake_planner,
    )

    # 2. 模拟摘要工具始终发生网络错误
    attempts = {"count": 0}

    async def fake_summarizer(text, task="summarize"):
        attempts["count"] += 1
        raise ConnectionError("模拟持续网络连接失败")

    monkeypatch.setattr(
        "app.agent.nodes.text_summarizer",
        fake_summarizer,
    )

    # 3. 构建初始状态
    initial_state = {
        "messages": [],
        "user_goal": "请总结一段测试文本",
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

    # 4. 执行完整 LangGraph 工作流
    graph = build_agent_graph()

    config = {
        "configurable": {
            "thread_id": "agent-retry-failure-test"
        },
        "recursion_limit": 20,
    }

    result = await graph.ainvoke(
        initial_state,
        config=config,
    )

    # 5. 检查实际调用次数
    assert attempts["count"] == 3

    print("\n测试七通过：工具最多执行三次")

    # 6. 检查所有调用均失败
    calls = result["tool_calls"]

    assert len(calls) == 3
    assert all(
        call["status"] == "failed"
        for call in calls
    )

    print("\n测试八通过：三次失败记录完整")

    # 7. 检查重试次数
    assert result["retry_count"] == 2

    print("\n测试九通过：最大重试次数限制正确")

    # 8. 检查最终任务状态
    assert result["status"] == "failed"

    assert result["plan"][0]["status"] == "failed"

    print("\n测试十通过：任务安全终止")

    # 9. 检查最终错误信息
    assert "任务执行失败" in result["final_answer"]
    assert "模拟持续网络连接失败" in result["final_answer"]

    print("\n测试十一通过：Finalizer 正确返回失败原因")

    # 10. 检查失败后没有进入 Reviewer
    assert result["review_passed"] is False

    print("\n测试十二通过：失败任务没有进入 Reviewer")

    print("\n========== 连续失败测试结果 ==========")
    print("工具实际执行次数：", attempts["count"])
    print("工具调用状态：", [item["status"] for item in calls])
    print("最终状态：", result["status"])
    print("最终错误：", result["final_answer"])
