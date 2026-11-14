import logging
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path


# 项目目录
BASE_DIR = Path(__file__).resolve().parents[2]

# 日志文件目录
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# 日志文件路径
LOG_FILE = LOG_DIR / "agent.log"


class SensitiveDataFormatter(logging.Formatter):
    """自动过滤日志中的常见敏感信息。"""

    SECRET_PATTERNS = [
        # API Key、密码、Token 等键值
        re.compile(
            r"(?i)\b(api[_-]?key|secret|password|access[_-]?token|"
            r"refresh[_-]?token)\b\s*[:=]\s*[^\s,;]+"
        ),

        # Bearer Token
        re.compile(
            r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+"
        ),

        # 常见 sk- 格式的 API Key
        re.compile(
            r"\bsk-[A-Za-z0-9_-]{16,}\b"
        ),
    ]

    def format(self, record: logging.LogRecord) -> str:
        """格式化日志后，对敏感信息进行脱敏。"""

        message = super().format(record)

        for pattern in self.SECRET_PATTERNS:
            message = pattern.sub("[REDACTED]", message)

        return message


def setup_logger() -> None:
    """
    初始化项目统一日志配置。
    同时输出到控制台和日志文件。
    """

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    formatter = SensitiveDataFormatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 检查是否已经添加项目控制台 Handler
    has_console_handler = any(
        getattr(handler, "_agent_console_handler", False)
        for handler in root_logger.handlers
    )

    if not has_console_handler:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        # 标记，防止重复添加
        console_handler._agent_console_handler = True

        root_logger.addHandler(console_handler)

    # 检查是否已经添加项目文件 Handler
    has_file_handler = any(
        getattr(handler, "_agent_file_handler", False)
        for handler in root_logger.handlers
    )

    if not has_file_handler:
        file_handler = RotatingFileHandler(
            LOG_FILE,
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8"
        )

        file_handler.setFormatter(formatter)

        # 标记，防止重复添加
        file_handler._agent_file_handler = True

        root_logger.addHandler(file_handler)


def get_logger(name: str = "agent_app") -> logging.Logger:
    """获取项目统一 Logger。"""

    setup_logger()

    return logging.getLogger(name)