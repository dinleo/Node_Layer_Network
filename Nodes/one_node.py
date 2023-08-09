import numpy as np


class GetValue:
    def __init__(self, v):
        self.v = v
        self.dv = None

    def forward(self):
        # 순전파시 grad 초기화
        self.dv = None
        # 순전파시 value return
        return self.v.copy()

    def backward(self, y):
        # 역전파시 흘러들어오는 grad 모두 더함
        if self.dv is None:
            self.dv = y.copy()
        else:
            self.dv += y

class OneNode:
    def __init__(self, x_node):
        self.x_node = x_node
        self.out = None


class ConstNode:
    def __init__(self, x_node, c):
        self.x_node = x_node
        self.c = c

    def forward(self):
        pass

    def backward(self, y):
        pass


class Exp(OneNode):
    def __init__(self, x_node):
        super().__init__(x_node)

    def forward(self):
        x = self.x_node.forward()
        self.out = np.exp(x)
        y = self.out.copy()

        return y

    def backward(self, y):
        dx = y * self.out
        # print("Exp backward:\n", dx)
        self.x_node.backward(dx)


class Log(OneNode):
    def __init__(self, x_node, eta=0):
        super().__init__(x_node)
        self.x = None
        self.eta = eta

    def forward(self):
        x = self.x_node.forward()
        self.x = x + self.eta
        y = np.log(self.x)

        return y

    def backward(self, y):
        dx = y / self.x
        # print("Log backward:\n", dx)
        self.x_node.backward(dx)


class Power(OneNode):
    def __init__(self, x_node, p, eta=0):
        super().__init__(x_node)
        self.p = p
        self.x = None
        self.eta = eta
        self.out = None

    def forward(self):
        x = self.x_node.forward()
        self.x = x + self.eta
        y = np.power(self.x, self.p)
        self.out = y.copy()

        return y

    def backward(self, y):
        if self.p == 0.5:
            dx = self.p * (1 / self.out)
        elif self.p == 2:
            dx = 2 * self.x
        else:
            dx = self.p * np.power(self.x, self.p - 1)
        dx = y * dx
        # print("Power backward\n", dx)
        self.x_node.backward(dx)


class Reciprocal(OneNode):
    def __init__(self, x_node, eta=0):
        super().__init__(x_node)
        self.eta = eta

    def forward(self):
        x = self.x_node.forward()
        self.out = 1 / (x + self.eta)
        y = self.out.copy()

        return y

    def backward(self, y):
        dx = (-1) * y * (self.out * self.out)

        self.x_node.backward(dx)


class AddConst(ConstNode):
    def __init__(self, x_node, c=1):
        super().__init__(x_node, c)

    def forward(self):
        x = self.x_node.forward()
        y = x + self.c

        return y

    def backward(self, y):
        # print("AddConst backward:\n", y)
        self.x_node.backward(y)


class MulConst(ConstNode):
    def __init__(self, x_node, c=-1):
        super().__init__(x_node, c)
        self.x_node = x_node
        self.c = c

    def forward(self):
        x = self.x_node.forward()
        y = x * self.c

        return y

    def backward(self, y):
        dx = y * self.c
        # print("MulConst backward:\n", dx)
        self.x_node.backward(dx)


class NormByMax:
    def __init__(self, x_node):
        self.x_node = x_node

    def forward(self):
        x = self.x_node.forward()
        y = x.T
        y = y - np.max(y, axis=0)
        y = y.T

        return y

    def backward(self, y):
        # print("NormByMax backward:\n", dx)
        self.x_node.backward(y)
