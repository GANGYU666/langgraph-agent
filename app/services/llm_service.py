from functools import lru_cache

from langchain_deepseek import ChatDeepSeek

from app.config import settings


@lru_cache(maxsize=1)
def get_chat_model() -> ChatDeepSeek:
    """
    创建并缓存 DeepSeek 大模型客户端。
    整个项目复用同一个模型客户端。
    """
    model = ChatDeepSeek(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key.get_secret_value(),
        base_url=settings.deepseek_base_url,
        temperature=0.2,
        timeout=settings.tool_timeout,
        max_retries=settings.max_retries,
    )

    return model


async def ask_llm(prompt: str) -> str:
    """
    向 DeepSeek 发送文本请求，并返回模型回答。
    """
    if not prompt or not prompt.strip():
        raise ValueError("prompt 不能为空")

    # 获取模型客户端
    model = get_chat_model()

    # 异步调用大模型
    response = await model.ainvoke(prompt)

    # 检查返回内容
    if not isinstance(response.content, str):
        raise TypeError("模型返回的内容不是字符串")

    return response.content.strip()
