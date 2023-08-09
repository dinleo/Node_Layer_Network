import numpy as np

class Reshape:
    def __init__(self, x_node, shape):
        self.x_node = x_node
        self.shape = shape
        self.x_shape = None

    def forward(self):
        x = self.x_node.forward()
        self.x_shape = x.shape
        y = x.reshape(*self.shape)
        return y

    def backward(self, y):
        dx = y.reshape(self.x_shape)

        self.x_node.backward(dx)


class T:
    def __init__(self, x_node):
        self.x_node = x_node

    def forward(self):
        x = self.x_node.forward()
        y = x.T

        return y

    def backward(self, y):
        dx = y.T

        self.x_node.backward(dx)


class Transpose:
    def __init__(self, x_node, shape):
        self.x_node = x_node
        self.shape = shape

    def forward(self):
        x = self.x_node.forward()
        y = x.transpose(*self.shape)

        return y

    def backward(self, y):
        r_shape = np.argsort(self.shape)
        dx = y.transpose(r_shape)

        self.x_node.backward(dx)


class Img2Matrix:
    def __init__(self, x_node, f_shape):
        self.x_node = x_node
        self.x_shape = None
        self.f_shape = f_shape

    def forward(self):
        x = self.x_node.forward()
        self.x_shape = x.shape
        fh = self.f_shape['fh']
        fw = self.f_shape['fw']
        pad = self.f_shape['pad']
        stride = self.f_shape['stride']

        N, C, H, W = x.shape
        out_h = (H + 2 * pad - fh) // stride + 1
        out_w = (W + 2 * pad - fw) // stride + 1

        img = np.pad(x, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
        col = np.zeros((N, C, fh, fw, out_h, out_w))

        for i in range(fh):
            y_max = i + stride * out_h
            for j in range(fw):
                x_max = j + stride * out_w
                col[:, :, i, j, :, :] = img[:, :, i:y_max:stride, j:x_max:stride]

        y = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)

        return y

    def backward(self, y):
        fh = self.f_shape['fh']
        fw = self.f_shape['fw']
        pad = self.f_shape['pad']
        stride = self.f_shape['stride']

        N, C, H, W = self.x_shape
        out_h = (H + 2 * pad - fh) // stride + 1
        out_w = (W + 2 * pad - fw) // stride + 1
        col = y.reshape(N, out_h, out_w, C, fh, fw).transpose(0, 3, 4, 5, 1, 2)

        img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
        for y in range(fh):
            y_max = y + stride * out_h
            for x in range(fw):
                x_max = x + stride * out_w
                img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

        dx = img[:, :, pad:H + pad, pad:W + pad]

        self.x_node.backward(dx)
