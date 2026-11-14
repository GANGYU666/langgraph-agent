from uuid import UUID
from pydantic import BaseModel, Field, field_validator

class AgentRunRequest(BaseModel):
    """启动 Agent 任务的请求模型。"""

    goal: str = Field(
        min_length=1,
        max_length=2000,
        description="用户希望 Agent 完成的任务",
    )

    @field_validator("goal")
    @classmethod
    def validate_goal(cls, value: str) -> str:
        """去除首尾空格，并拒绝纯空白任务。"""

        value = value.strip()

        if not value:
            raise ValueError("任务内容不能为空或仅包含空白字符")

        return value


class AgentConfirmRequest(BaseModel):
    """人工确认操作的请求模型。"""

    thread_id: str = Field(
        min_length=1,
        max_length=100,
        description="Agent 会话标识，必须是合法 UUID",
    )

    approved: bool = Field(
        strict=True,
        description="是否批准执行操作，必须为布尔值",
    )

    @field_validator("thread_id")
    @classmethod
    def validate_thread_id(cls, value: str) -> str:
        """检查会话 ID 是否为合法 UUID。"""

        value = value.strip()

        if not value:
            raise ValueError("thread_id 不能为空")

        try:
            return str(UUID(value))
        except ValueError as exc:
            raise ValueError("thread_id 必须是合法的 UUID") from exc