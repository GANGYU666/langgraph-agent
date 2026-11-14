import pytest

from app.tools.file_writer import file_writer, OUTPUT_DIR


def test_write_markdown():
    result = file_writer(
        filename="test_report.md",
        content="# 测试报告\n\n这是一个 Markdown 测试文件。",
    )

    assert result["success"] is True
    assert result["filename"] == "test_report.md"
    assert result["file_size"] > 0

    file_path = OUTPUT_DIR / "test_report.md"

    assert file_path.exists()
    assert file_path.read_text(encoding="utf-8").startswith("# 测试报告")

    print("\n测试一通过：Markdown 文件生成")


def test_empty_filename():
    with pytest.raises(ValueError, match="文件名不能为空"):
        file_writer("", "测试内容")


def test_empty_content():
    with pytest.raises(ValueError, match="文件内容不能为空"):
        file_writer("test.md", "  ")


def test_invalid_extension():
    with pytest.raises(ValueError, match="只允许生成 Markdown 文件"):
        file_writer("test.txt", "测试内容")


def test_path_traversal():
    with pytest.raises(ValueError):
        file_writer("../outside.md", "不允许写入")

    print("\n测试五通过：路径穿越拦截")


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
