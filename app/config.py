from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# 获取项目根目录下的 .env 文件
ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


class Settings(BaseSettings):
    """项目全局配置"""

    # 配置文件读取规则
    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # 应用配置
    app_name: str = "LangGraph Multi-Tool Agent"
    debug: bool = False

    # DeepSeek 模型配置
    deepseek_api_key: SecretStr

    deepseek_base_url: str = "https://api.deepseek.com"

    deepseek_model: str = "deepseek-chat"

    # Agent 执行限制
    max_iterations: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Agent 最大执行步骤数",
    )

    max_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        description="工具执行最大重试次数",
    )

    tool_timeout: int = Field(
        default=15,
        ge=1,
        le=120,
        description="工具执行超时时间，单位为秒",
    )

    # 检查 API Key 是否为空
    @field_validator("deepseek_api_key")
    @classmethod
    def validate_api_key(cls, value: SecretStr) -> SecretStr:
        secret = value.get_secret_value().strip()

        if not secret:
            raise ValueError("DEEPSEEK_API_KEY 不能为空")

        return SecretStr(secret)

    # 检查 API 地址是否合法
    @field_validator("deepseek_base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        value = value.strip()

        parsed_url = urlsplit(value)

        if (
            parsed_url.scheme not in {"http", "https"}
            or not parsed_url.netloc
        ):
            raise ValueError(
                "DEEPSEEK_BASE_URL 必须是合法的 HTTP 或 HTTPS 地址"
            )

        return value.rstrip("/")

    # 检查模型名称是否为空
    @field_validator("deepseek_model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("DEEPSEEK_MODEL 不能为空")

        return value


# 缓存配置对象，避免重复创建
@lru_cache
def get_settings() -> Settings:
    return Settings()


# 全局配置对象，供其他模块使用
settings = get_settings()