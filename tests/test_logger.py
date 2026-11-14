from app.utils.logger import get_logger,LOG_FILE

def test_logger():
    logger = get_logger("test_logger")
    logger.info("日志模块测试成功")

    assert LOG_FILE.exists()
    assert logger.name == "test_logger"