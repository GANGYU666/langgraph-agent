import asyncio

from app.agent.nodes import planner_node
from app.services.llm_service import get_chat_model

def build_initial_state(user_goal: str) -> dict:
    return {
        "messages": [],
        "user_goal": user_goal,
        "plan": [],
        "current_step": 0,
        "tool_calls": [],
        "tool_results": [],
        "needs_confirmation": False,
        "confirmation_request": "",
        "confirmation_result": None,
        "final_answer": "",
        "error": "",
        "iteration_count": 0,
        "status": "pending",
    }


async def run_planner(user_goal: str) -> dict:
    initial_state = build_initial_state(user_goal)
    return await planner_node(initial_state)


async def run_all_tests():
    # 测试一：数学计算任务
    result1 = await run_planner("计算100、200、300的平均值")

    assert result1["status"] == "running"
    assert any(step["tool"] == "calculator" for step in result1["plan"])
    print("\n测试一通过：数学计算任务")

    # 测试二：文本摘要任务
    result2 = await run_planner(
        "请对以下文本进行摘要："
        "人工智能技术正在快速发展，"
        "大语言模型已经应用于教育、医疗、金融和软件开发等领域。"
        "RAG技术能够将外部知识库与大语言模型结合，"
        "提高回答的准确性和可追溯性。"
    )

    assert any(
        step["tool"] == "text_summarizer"
        for step in result2["plan"]
    )
    print("\n测试二通过：文本摘要任务")

    # 测试三：多工具组合任务
    result3 = await run_planner(
        "计算10、20、30的平均值，"
        "然后将计算过程和结果整理成Markdown报告。"
    )

    tools = [step["tool"] for step in result3["plan"]]

    assert "calculator" in tools
    assert "file_writer" in tools

    step_ids = [step["step_id"] for step in result3["plan"]]
    assert step_ids == list(range(1, len(step_ids) + 1))

    print("\n测试三通过：多工具组合任务")


def test_planner_all_scenarios():
    # 清除之前测试遗留的模型客户端缓存
    get_chat_model.cache_clear()

    asyncio.run(run_all_tests())

    # 清除当前测试创建的模型客户端缓存
    get_chat_model.cache_clear()


if __name__ == "__main__":
    test_planner_all_scenarios()
    print("\n所有 Planner 测试完成！")
