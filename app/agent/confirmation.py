from typing import Any


# 需要人工确认的工具白名单
CONFIRMATION_REQUIRED_TOOLS = {
    "file_writer",
}

# 文件内容预览最大字符数
MAX_PREVIEW_CHARS = 300


def requires_confirmation(tool_name: str) -> bool:
    """
    判断工具是否需要人工确认。
    """

    return tool_name in CONFIRMATION_REQUIRED_TOOLS


def build_confirmation_request(step: dict[str, Any]) -> str:
    """
    根据当前任务步骤生成用户确认请求。
    """

    tool_name = step.get("tool")

    # 1. 检查是否属于需要确认的工具
    if not requires_confirmation(tool_name):
        raise ValueError(f"工具 {tool_name} 不需要人工确认")

    # 2. 检查工具参数
    tool_input = step.get("tool_input", {})

    if not isinstance(tool_input, dict):
        raise ValueError("工具参数必须是字典类型")

    # 3. 获取任务信息
    step_id = step.get("step_id", "未知")
    description = step.get("description", "无任务描述")

    filename = tool_input.get("filename", "")
    content = tool_input.get("content", "")

    if not isinstance(content, str):
        raise ValueError("文件内容必须是字符串")

    if not isinstance(filename, str):
        raise ValueError("文件名必须是字符串")

    # 4. 限制预览内容长度
    preview = content[:MAX_PREVIEW_CHARS]

    if len(content) > MAX_PREVIEW_CHARS:
        preview += "\n……（内容预览已截断）"

    # 5. 生成确认信息
    request = (
        "需要用户确认后才能执行以下操作：\n\n"
        f"任务步骤：{step_id}\n"
        f"操作工具：{tool_name}\n"
        f"任务描述：{description}\n"
        f"目标文件名：{filename}\n\n"
        f"文件内容预览：\n{preview}\n\n"
        "请批准或拒绝此次操作。"
    )

    return request
