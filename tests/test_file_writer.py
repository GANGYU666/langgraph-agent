import pytest

from app.tools.file_writer import file_writer, OUTPUT_DIR


def test_write_markdown(tmp_path, monkeypatch):
    # 将文件输出目录设置为 pytest 临时目录
    monkeypatch.setattr(
        "app.tools.file_writer.OUTPUT_DIR",
        tmp_path,
    )
    monkeypatch.setattr(
        "app.tools.file_writer.PROJECT_ROOT",
        tmp_path.parent,
    )

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

def test_existing_file_not_overwritten(tmp_path, monkeypatch):
    # 使用 pytest 临时目录
    monkeypatch.setattr(
        "app.tools.file_writer.OUTPUT_DIR",
        tmp_path,
    )
    monkeypatch.setattr(
        "app.tools.file_writer.PROJECT_ROOT",
        tmp_path.parent,
    )

    # 预先创建一个同名文件
    file_path = tmp_path / "existing_report.md"
    original_content = "# 原始报告\n\n这部分内容不能被覆盖。"
    file_path.write_text(original_content, encoding="utf-8")

    # 再次写入同名文件，应该抛出 FileExistsError
    with pytest.raises(FileExistsError):
        file_writer(
            filename="existing_report.md",
            content="# 新报告\n\n尝试覆盖原文件。",
        )

    # 验证原文件内容保持不变
    assert file_path.read_text(encoding="utf-8") == original_content

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

@pytest.mark.parametrize(
    "filename",
    [
        "CON.md",
        "PRN.md",
        "AUX.md",
        "NUL.md",
        "COM1.md",
        "LPT1.md",
    ],
)
def test_windows_reserved_filename(filename):
    with pytest.raises(ValueError, match="Windows 系统保留名称"):
        file_writer(filename, "测试内容")

if __name__ == "__main__":
    pytest.main([__file__, "-s"])
