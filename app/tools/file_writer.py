from pathlib import Path


# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 所有报告统一保存到项目的 outputs 目录
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def file_writer(
    filename: str,
    content: str,
) -> dict:
    """
    将内容保存为 Markdown 文件。

    参数：
        filename: 文件名称
        content: 文件内容

    返回：
        文件路径、文件名和文件大小
    """

    # 1. 参数校验
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("文件名不能为空")

    if not isinstance(content, str) or not content.strip():
        raise ValueError("文件内容不能为空")

    filename = filename.strip()

    # 2. 文件名安全校验
    # 同时检查 Windows 和 Linux 路径分隔符
    if "/" in filename or "\\" in filename:
        raise ValueError("文件名不能包含目录路径")

    if Path(filename).name != filename:
        raise ValueError("文件名不能包含目录路径")

    # 检查 Windows 不允许的文件名字符
    invalid_chars = '<>:"|?*'

    if any(char in filename for char in invalid_chars):
        raise ValueError("文件名包含不允许的字符")

    if any(ord(char) < 32 for char in filename):
        raise ValueError("文件名不能包含控制字符")

    if filename.endswith((" ", ".")):
        raise ValueError("文件名不能以空格或句点结尾")

    # 3. 只允许 Markdown 文件
    if not filename.lower().endswith(".md"):
        raise ValueError("目前只允许生成 Markdown 文件")

    # 4. 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 5. 检查输出目录本身是否位于项目根目录内
    project_root = PROJECT_ROOT.resolve()
    output_root = OUTPUT_DIR.resolve()

    if not output_root.is_relative_to(project_root):
        raise ValueError("输出目录不能指向项目目录以外的位置")

    # 6. 生成目标文件路径
    output_path = (output_root / filename).resolve()

    # 7. 再次检查目标文件是否位于输出目录内
    if not output_path.is_relative_to(output_root):
        raise ValueError("不允许写入输出目录以外的位置")

    # 8. 写入文件
    output_path.write_text(
        content,
        encoding="utf-8",
    )

    # 9. 返回执行结果
    return {
        "success": True,
        "filename": filename,
        "file_path": str(output_path),
        "file_size": output_path.stat().st_size,
    }