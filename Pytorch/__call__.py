"""
__call__ 类似于C++的运算符重载
"""

class Add():
    def __call__(self, *args, **kwargs):
        self.add(args[0], args[1])


    def add(self, x, y):
        print(str(x + y))


# 两种方式
add = Add()
add(1, 2)       # 3

Add()(1, 4)     # 5