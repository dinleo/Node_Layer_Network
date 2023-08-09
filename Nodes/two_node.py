import numpy as np

class TwoNode:
    def __init__(self, a_node, b_node):
        self.a_node = a_node
        self.b_node = b_node

    def forward(self):
        pass

    def backward(self, y):
        pass


class Dot(TwoNode):
    def __init__(self, a_node, b_node):
        super().__init__(a_node, b_node)
        self.aT = None
        self.bT = None

    def forward(self):
        a = self.a_node.forward()
        b = self.b_node.forward()
        self.aT = a.T  # [128, 100] -> [100, 128]
        self.bT = b.T  # [100, 10] -> [10, 100]
        y = np.dot(a, b)  # [128, 100] * [100, 10] -> [128, 10]

        return y

    def backward(self, y):
        da = np.dot(y, self.bT)  # [128, 10] * [10, 100] -> [128, 100]
        db = np.dot(self.aT, y)  # [100, 128] * [128, 10] -> [100, 10]
        self.a_node.backward(da)
        self.b_node.backward(db)


class Mul(TwoNode):
    def __init__(self, a_node, b_node):
        super().__init__(a_node, b_node)
        self.a = None
        self.b = None

    def forward(self):
        self.a = self.a_node.forward()
        self.b = self.b_node.forward()
        y = self.a * self.b

        return y

    def backward(self, y):
        da = y * self.b
        db = self.a * y
        # print("Mul backward left:\n", da)
        # print("Mul backward right:\n", db)
        self.a_node.backward(da)
        self.b_node.backward(db)


class Add(TwoNode):
    def __init__(self, a_node, b_node):
        super().__init__(a_node, b_node)

    def forward(self):
        a = self.a_node.forward()
        b = self.b_node.forward()
        y = a + b

        return y

    def backward(self, y):
        da = y.copy()
        db = y.copy()
        # print("Add Backward\n:", y)
        self.a_node.backward(da)
        self.b_node.backward(db)
