import pytest

from app.tools.calculator import calculator


def test_add():
    result = calculator("add", [10, 20, 30])
    assert result["result"] == 60
    print("\n测试一通过：加法")


def test_subtract():
    result = calculator("subtract", [100, 20, 10])
    assert result["result"] == 70
    print("\n测试二通过：减法")


def test_multiply():
    result = calculator("multiply", [2, 3, 4])
    assert result["result"] == 24
    print("\n测试三通过：乘法")


def test_divide():
    result = calculator("divide", [100, 2, 5])
    assert result["result"] == 10
    print("\n测试四通过：除法")


def test_average():
    result = calculator("average", [10, 20, 30])
    assert result["result"] == 20
    print("\n测试五通过：平均值")


def test_divide_by_zero():
    with pytest.raises(ValueError, match="除数不能为零"):
        calculator("divide", [10, 0])

    print("\n测试六通过：除零异常处理")


def test_invalid_operation():
    with pytest.raises(ValueError, match="不支持的运算类型"):
        calculator("power", [2, 3])

    print("\n测试七通过：非法运算拦截")


def test_invalid_input():
    with pytest.raises(ValueError):
        calculator("add", [])

    with pytest.raises(ValueError):
        calculator("add", [10, "abc"])

    print("\n测试八通过：非法参数拦截")


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
