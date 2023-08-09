import numpy as np


class Sum:
    def __init__(self, x_node, axis=0):
        self.x_node = x_node
        self.axis = axis
        self.r = None
        self.shape = None

    def forward(self):
        x = self.x_node.forward()
        self.r = x.shape[self.axis]
        y = np.sum(x, axis=self.axis)

        return y

    def backward(self, y):
        dx = np.expand_dims(y, axis=self.axis)
        dx = np.repeat(dx, self.r, axis=self.axis)
        # print("Sum backward:\n", dx)
        self.x_node.backward(dx)


class Repeat:
    def __init__(self, x_node, axis, r):
        self.x_node = x_node
        self.axis = axis
        self.r = r

    def forward(self):
        x = self.x_node.forward()
        y = np.expand_dims(x, axis=self.axis)
        y = np.repeat(y, repeats=self.r, axis=self.axis)

        return y

    def backward(self, y):
        dx = np.sum(y, axis=self.axis)
        # print("Repeat backward:\n", dx)
        self.x_node.backward(dx)


class Mean:
    def __init__(self, x_node, axis=0, name=""):
        self.x_node = x_node
        self.axis = axis
        self.r = None
        self.name = name
        self.out = None

    def forward(self):
        x = self.x_node.forward()
        self.r = x.shape[self.axis]
        y = np.sum(x, axis=self.axis) / self.r
        self.out = y.copy()

        return y

    def backward(self, y):
        dx = np.expand_dims(y, axis=self.axis)
        dx = np.repeat(dx, self.r, axis=self.axis) / self.r
        # print(self.name + " backward:\n", dx)
        self.x_node.backward(dx)


class ValueMask:
    def __init__(self, x_node, t=0):
        self.x_node = x_node
        self.t = t
        self.mask = None

    def forward(self):
        x = self.x_node.forward()
        self.mask = (x <= self.t)
        y = x.copy()
        y[self.mask] = 0

        return y

    def backward(self, y):
        dx = y.copy()
        dx[self.mask] = 0

        self.x_node.backward(dx)


class IndexMask:
    def __init__(self, x_node, t=0):
        self.x_node = x_node
        self.t = t
        self.mask = None

    def forward(self):
        x = self.x_node.forward()
        self.mask = np.random.rand(*x.shape) > self.t
        y = x * self.mask

        return y

    def backward(self, y):
        dx = y * self.mask

        self.x_node.backward(dx)


class Max:
    def __init__(self, x_node, axis=0):
        self.x_node = x_node
        self.axis = axis
        self.x_shape = None
        self.o_shape = None
        self.mask = None

    def forward(self):
        x = self.x_node.forward()
        y = np.max(x, axis=self.axis)

        self.x_shape = x.shape
        self.mask = np.argmax(x, axis=self.axis)
        self.o_shape = y.shape

        return y

    def backward(self, y):
        dx = np.zeros(self.x_shape)
        if len(self.x_shape) == 1:
            dx[self.mask] = y
        else:
            n_i = np.indices(self.o_shape)
            mask = np.insert(n_i, self.axis, self.mask, axis=0)
            dx[tuple(mask)] = y
        # print("Max backward:\n", dx)
        self.x_node.backward(dx)
