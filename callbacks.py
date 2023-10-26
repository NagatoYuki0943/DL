from typing import Any, Callable


def func1():
    print("func1")


def func2():
    print("func2")


def func3():
    print("func3")


class Func4:
    def __init__(self, num: int = 0):
        self.num = num

    def __call__(self):
        print(f"Func4: {self.num}")
        self.num += 1


# 调用函数
def callbacks(funcs: list[Callable]):
    for func in funcs:
        func()


if __name__ == "__main__":
    func4 = Func4()
    fs = [func1, func2, func3, func4, func4, func4]
    callbacks(fs)
    # func1
    # func2
    # func3
    # Func4: 0
    # Func4: 1
    # Func4: 2
