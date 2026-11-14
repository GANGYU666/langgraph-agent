import math
from typing import Union

Number = Union[int, float]

# 允许的运算类型
ALLOWED_OPERATIONS = {
    "add",
    "subtract",
    "multiply",
    "divide",
    "average",
}

# 单次计算最多允许的数字数量
MAX_NUMBERS = 100

# 输入和计算结果允许的最大绝对值
MAX_ABS_NUMBER = 1e100


def calculator(
    operation: str,
    numbers: list[Number],
) -> dict:
    """
    执行基础数学计算。

    支持：
        add       加法
        subtract  减法
        multiply  乘法
        divide    除法
        average   平均值
    """

    # 1. 检查运算类型
    if not isinstance(operation, str):
        raise ValueError("operation 必须是字符串")

    if operation not in ALLOWED_OPERATIONS:
        raise ValueError(f"不支持的运算类型：{operation}")

    # 2. 检查数字列表
    if not isinstance(numbers, list) or not numbers:
        raise ValueError("numbers 必须是非空数字列表")

    if len(numbers) > MAX_NUMBERS:
        raise ValueError(
            f"单次计算最多允许 {MAX_NUMBERS} 个数字"
        )

    # 3. 检查数字类型和范围
    for number in numbers:

        if isinstance(number, bool) or not isinstance(
            number, (int, float)
        ):
            raise ValueError("所有计算参数必须是数字")

        # 浮点数检查
        if isinstance(number, float) and not math.isfinite(number):
            raise ValueError("不允许使用无穷大或 NaN")

        # 防止超大数字
        if abs(number) > MAX_ABS_NUMBER:
            raise ValueError("输入数字超出允许范围")

    # 4. 根据运算类型执行计算
    if operation == "add":
        result = sum(numbers)

    elif operation == "subtract":
        if len(numbers) < 2:
            raise ValueError("减法至少需要两个数字")

        result = numbers[0]

        for number in numbers[1:]:
            result -= number

    elif operation == "multiply":
        result = math.prod(numbers)

    elif operation == "divide":
        if len(numbers) < 2:
            raise ValueError("除法至少需要两个数字")

        result = numbers[0]

        for number in numbers[1:]:
            if number == 0:
                raise ValueError("除数不能为零")

            result /= number

    elif operation == "average":
        result = sum(numbers) / len(numbers)

    # 5. 检查最终计算结果
    if isinstance(result, float) and not math.isfinite(result):
        raise ValueError("计算结果超出有效数值范围")

    if abs(result) > MAX_ABS_NUMBER:
        raise ValueError("计算结果超出允许范围")

    # 6. 返回计算结果
    return {
        "operation": operation,
        "numbers": numbers,
        "result": result,
    }