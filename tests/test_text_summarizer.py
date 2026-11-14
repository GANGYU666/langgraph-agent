import pytest

from app.tools.text_summarizer import text_summarizer


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_text_summarizer():
    text = (
        "人工智能技术正在快速发展。"
        "大语言模型已经应用于教育、医疗、金融和软件开发等领域。"
        "RAG技术能够将外部知识库与大语言模型结合，"
        "通过检索相关知识提高回答的准确性和可追溯性。"
    )

    result = await text_summarizer(text)

    assert result["task"] == "summarize"
    assert result["original_text"] == text
    assert isinstance(result["summary"], str)
    assert len(result["summary"]) > 0

    print("\n测试通过：文本摘要功能")
    print("模型生成的摘要：")
    print(result["summary"])


@pytest.mark.anyio
async def test_empty_text():
    with pytest.raises(ValueError, match="待处理文本不能为空"):
        await text_summarizer("   ")


@pytest.mark.anyio
async def test_invalid_task():
    with pytest.raises(ValueError, match="暂不支持"):
        await text_summarizer(
            text="这是一段测试文本",
            task="translate",
        )


@pytest.mark.anyio
async def test_text_too_long():
    with pytest.raises(ValueError, match="不能超过12000个字符"):
        await text_summarizer("人工智能" * 4000)


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
