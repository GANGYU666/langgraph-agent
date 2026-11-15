import asyncio
from pathlib import Path

from app.agent.graph import build_agent_graph
from langgraph.types import Command


async def run_agent():
    """运行完整的 LangGraph Agent。"""

    # 1. 创建初始状态
    initial_state = {
        "messages": [],
        "user_goal": (
            "请先计算10、20、30的平均值，"
            "然后把计算过程和结果整理成名为"
            "e2e_report.md的Markdown报告。"
            "报告必须包含输入数字、运算类型和最终计算结果。"
        ),
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
        "status": "pending",
    }

    # 2. 创建 Agent 工作流
    graph = build_agent_graph()

    # 3. 异步执行整个工作流
    config = {
        "configurable": {
            "thread_id": "agent-e2e-confirmation-test"
        }
    }

    # 第一次执行：如果遇到需要人工确认的操作，工作流会暂停
    result = await graph.ainvoke(
        initial_state,
        config=config,
    )

    # 检查是否发生了人工确认中断
    if result.get("__interrupt__"):
        print("Agent 已暂停，模拟用户批准操作……")

        # 恢复工作流，并模拟用户批准
        result = await graph.ainvoke(
            Command(resume=True),
            config=config,
        )

    return result

def test_agent_end_to_end(tmp_path, monkeypatch):
    """测试完整的 Agent 执行流程。"""

    # 将文件输出目录设置为 pytest 临时目录，避免污染真实 outputs
    monkeypatch.setattr(
        "app.tools.file_writer.OUTPUT_DIR",
        tmp_path,
    )
    monkeypatch.setattr(
        "app.tools.file_writer.PROJECT_ROOT",
        tmp_path.parent,
    )

    # 4. 运行 Agent
    result = asyncio.run(run_agent())

    # 5. 检查最终执行状态
    assert result["status"] == "completed"

    print("\n测试一通过：Agent 执行成功")

    # 6. 检查任务计划
    plan = result["plan"]

    assert len(plan) >= 2

    tools = [step["tool"] for step in plan]

    assert "calculator" in tools
    assert "file_writer" in tools

    assert tools.index("calculator") < tools.index("file_writer")

    assert all(
        step["status"] == "completed"
        for step in plan
    )

    print("\n测试二通过：任务规划与执行步骤正确")

    # 7. 检查计算结果
    calculator_result = next(
        (
            item for item in result["tool_results"]
            if item["tool_name"] == "calculator"
        ),
        None,
    )

    assert calculator_result is not None

    calculation_output = calculator_result["output"]

    assert abs(
        calculation_output["result"] - 20.0
    ) < 1e-9

    print("\n测试三通过：Calculator 计算结果正确")

    # 8. 检查报告生成结果
    writer_result = next(
        (
            item for item in result["tool_results"]
            if item["tool_name"] == "file_writer"
        ),
        None,
    )

    assert writer_result is not None

    writer_output = writer_result["output"]

    assert writer_output["success"] is True
    assert writer_output["filename"] == "e2e_report.md"

    # 9. 检查实际生成的文件
    report_path = Path(writer_output["file_path"])

    assert report_path.exists()

    report_content = report_path.read_text(
        encoding="utf-8"
    )

    assert "20.0" in report_content
    assert "计算结果" in report_content
    assert "10、20、30" in report_content

    print("\n测试四通过：Markdown 报告生成成功")

    # 10. 检查工具调用记录
    assert len(result["tool_calls"]) >= 2

    assert all(
        call["status"] == "success"
        for call in result["tool_calls"]
    )

    print("\n测试五通过：工具调用记录正确")

    # 11. 检查最终回答
    assert result["final_answer"]
    assert "任务已完成" in result["final_answer"]

    print("\n测试六通过：Finalizer 最终结果正确")
    # 12. 检查Reviewer是否真正执行
    assert result["review_passed"] is True
    assert "审核通过" in result["review_feedback"]

    print("\n测试七通过：Reviewer已执行并审核通过")
    #13. 检查最终状态
    assert result["status"] == "completed"

    print("\n测试八通过：Agent 最终状态正确")

    print("\n========== Agent 最终执行结果 ==========")
    print(result["final_answer"])

    print("\n报告保存位置：")
    print(report_path)


if __name__ == "__main__":
    test_agent_end_to_end()
    print("\n端到端测试全部完成！")
