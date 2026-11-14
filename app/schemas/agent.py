from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

class PlannedStep(BaseModel):
    """单个任务步骤的数据模型"""

    model_config = ConfigDict(extra="forbid")

    step_id: int = Field(
        ge=1,
        description="任务步骤编号，从1开始"
    )

    description: str = Field(
        min_length=1,
        description="当前步骤的具体任务描述"
    )

    tool: Literal[
        "calculator",
        "text_summarizer",
        "file_writer"
    ] = Field(
        description="当前步骤需要调用的工具名称"
    )

    tool_input: dict[str, Any] = Field(
        default_factory=dict,
        description="调用工具所需的参数"
    )


class PlannerOutput(BaseModel):
    """Planner 生成的完整任务计划"""

    model_config = ConfigDict(extra="forbid")

    steps: list[PlannedStep] = Field(
        min_length=1,
        max_length=5,
        description="按照执行顺序排列的任务步骤"
    )
