import sys
from Nodes.one_node import *
from Nodes.two_node import *
from Nodes.structure_node import *
from Nodes.axis_node import *
from Nodes.functions import *


class Layer:
    def __init__(self):
        self.get_x = GetValue(None)
        self.dx = None
        self.last_node = None

    def forward(self, x):
        self.get_x.v = x

        return self.last_node.forward()

    def backward(self, y):
        self.last_node.backward(y)
        self.dx = self.get_x.dv

        return self.dx


class Affine(Layer):
    def __init__(self, w, b):
        super().__init__()
        # Leaf Node
        self.get_w = GetValue(w)
        self.get_b = GetValue(b)
        # gradient
        self.dW = None
        self.db = None
        # Tree 생성 (노드 연결)
        self.reshape_node = Reshape(self.get_x, None)
        self.dot_xw_node = Dot(self.reshape_node, self.get_w)
        self.rep_b_node = Repeat(self.get_b, 0, None)
        self.last_node = Add(self.dot_xw_node, self.rep_b_node)

    def forward(self, x):
        self.get_x.v = x
        self.reshape_node.shape = [x.shape[0], -1]
        self.rep_b_node.r = x.shape[0]

        return self.last_node.forward()

    def backward(self, y):
        self.last_node.backward(y)
        self.dW = self.get_w.dv
        self.db = self.get_b.dv
        self.dx = self.get_x.dv

        return self.dx


class Sigmoid(Layer):
    def __init__(self):
        super().__init__()

        self.neg_node = MulConst(self.get_x, -1)
        self.exp_node = Exp(self.neg_node)
        self.add1_node = AddConst(self.exp_node, 1)
        self.last_node = Reciprocal(self.add1_node)


class Relu(Layer):
    def __init__(self):
        super().__init__()
        self.last_node = ValueMask(self.get_x, 0)


class Softmax(Layer):
    def __init__(self):
        super().__init__()

        self.norm_node = NormByMax(self.get_x)
        self.exp_node = Exp(self.norm_node)
        self.sum_node = Sum(self.exp_node, 1)
        self.rec_node = Reciprocal(self.sum_node)
        self.rep_node = Repeat(self.rec_node, 1, None)
        self.last_node = Mul(self.exp_node, self.rep_node)

    def forward(self, x):
        self.get_x.v = x
        self.rep_node.r = x.shape[1]
        y = self.last_node.forward()
        # print("Y:\n", y)
        return y


class CrossEntropy:
    def __init__(self):
        self.get_x = GetValue(None)
        self.dx = None
        self.get_t = GetValue(None)

        self.mul_node = Mul(self.get_t, self.get_x)
        self.sum_node = Sum(self.mul_node, 1)
        self.log_node = Log(self.sum_node, 1e-7)
        self.neg_node = MulConst(self.log_node, -1)
        self.last_node = Mean(self.neg_node, 0)

    def forward(self, x, t):
        self.get_x.v = x
        if (x.ndim == 1 and len(t) == 1) or (x.ndim == 2 and t.ndim == 1):
            # not one-hot-vec
            z = np.zeros((t.shape[0], x.shape[x.ndim - 1]))
            for i, v in enumerate(t):
                z[i, v] = 1
            self.get_t.v = z

        else:
            # one-hot-vec
            self.get_t.v = t

        return self.last_node.forward()

    def backward(self, y):
        self.last_node.backward(y)
        self.dx = self.get_x.dv.copy()

        return self.dx


class SoftmaxWithLoss:
    def __init__(self):
        self.s = Softmax()
        self.c = CrossEntropy()
        self.y = None
        self.t = None

    def forward(self, x, t):
        y = self.s.forward(x)
        o = self.c.forward(y, t)
        self.y = y
        self.t = self.c.get_t.v

        return o

    def backward(self, y):
        dx = self.c.backward(1)
        dx = self.s.backward(dx)
        # 빠른 계산을 위한 공식 적용, 단 Log Node 에서 self.x = x + eta 가 아닌  self.x = x 로 적용됨 주의
        # dx = (self.y - self.t) / self.t.shape[0]
        return dx


class SigmoidWithLoss:
    def __init__(self):
        self.s = Sigmoid()
        self.c = CrossEntropy()

    def forward(self, x, t):
        y = self.s.forward(x)
        o = self.c.forward(y, t)

        return o

    def backward(self, y):
        dx = self.c.backward(1)
        dx = self.s.backward(dx)

        return dx


class Convolution(Layer):
    def __init__(self, w, b, stride=1, pad=0):
        super().__init__()
        self.get_w = GetValue(w)
        self.get_b = GetValue(b)

        self.dW = None
        self.db = None

        FN, C, FH, FW = self.get_w.v.shape
        self.f_shape = {
            'fh': FH,
            'fw': FW,
            'stride': stride,
            'pad': pad
        }

        self.im2col_node = Img2Matrix(self.get_x, self.f_shape)
        self.w_reshape_node = Reshape(self.get_w, [FN, -1])
        self.w_t_node = T(self.w_reshape_node)
        self.dot_node = Dot(self.im2col_node, self.w_t_node)
        self.rep_b_node = Repeat(self.get_b, 0, None)
        self.add_node = Add(self.dot_node, self.rep_b_node)
        self.y_reshape_node = Reshape(self.add_node, None)
        self.last_node = Transpose(self.y_reshape_node, [0, 3, 1, 2])

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2 * self.f_shape['pad'] - self.f_shape['fh']) / self.f_shape['stride'])
        out_w = 1 + int((W + 2 * self.f_shape['pad'] - self.f_shape['fw']) / self.f_shape['stride'])

        self.get_x.v = x
        self.rep_b_node.r = N * out_h * out_w
        self.y_reshape_node.shape = [N, out_h, out_w, -1]

        return self.last_node.forward()

    def backward(self, y):
        self.last_node.backward(y)

        self.dW = self.get_w.dv
        self.db = self.get_b.dv
        self.dx = self.get_x.dv

        return self.dx


class Pooling(Layer):
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        super().__init__()
        self.f_shape = {
            'fh': pool_h,
            'fw': pool_w,
            'stride': stride,
            'pad': pad
        }

        self.im2col_node = Img2Matrix(self.get_x, self.f_shape)
        self.reshape_node = Reshape(self.im2col_node, [-1, pool_h * pool_w])
        self.max_node = Max(self.reshape_node, axis=1)
        self.reshape_node2 = Reshape(self.max_node, None)
        self.last_node = Transpose(self.reshape_node2, [0, 3, 1, 2])

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.f_shape['fh']) / self.f_shape['stride'])
        out_w = int(1 + (W - self.f_shape['fw']) / self.f_shape['stride'])

        self.get_x.v = x
        self.reshape_node2.shape = [N, out_h, out_w, C]

        return self.last_node.forward()


class BatchNormalization(Layer):
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        super().__init__()
        self.get_r = GetValue(gamma)
        self.get_b = GetValue(beta)
        self.dgamma = None
        self.dbeta = None

        self.momentum = momentum
        self.running_mean = running_mean
        self.running_var = running_var
        self.input_shape = None
        self.first = True

        # Layer
        self.mean_node = Mean(self.get_x, axis=0, name="mean1")
        self.neg_node = MulConst(self.mean_node, -1)
        self.mu_node = Repeat(self.neg_node, axis=0, r=None)

        self.xc_node = Add(self.get_x, self.mu_node)

        self.sqr_node = Power(self.xc_node, 2)
        self.var_node = Mean(self.sqr_node, axis=0, name="var")
        self.std_node = Power(self.var_node, 0.5, 10e-7)
        self.recip_node = Reciprocal(self.std_node, 0)
        self.rep_node2 = Repeat(self.recip_node, axis=0, r=None)

        self.xn_node = Mul(self.xc_node, self.rep_node2)
        self.r_node = Repeat(self.get_r, axis=0, r=None)
        self.b_node = Repeat(self.get_b, axis=0, r=None)
        self.r_xn_node = Mul(self.xn_node, self.r_node)
        self.last_node = Add(self.r_xn_node, self.b_node)

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)
        N, D = x.shape

        if train_flg:
            if self.first:
                self.mu_node.r = N
                self.rep_node2.r = N
                self.r_node.r = N
                self.b_node.r = N
                self.first = False
            self.get_x.v = x

            out = self.last_node.forward()

            if self.running_mean is None:
                self.running_mean = np.zeros(D)
                self.running_var = np.zeros(D)

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.mean_node.out
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * np.power(self.std_node.out, 2)
        else:
            xc = x - self.running_mean
            xn = xc / (np.sqrt(self.running_var + 10e-7))
            out = self.get_r.v * xn + self.get_b.v

        return out.reshape(*self.input_shape)

    def backward(self, y):
        if y.ndim != 2:
            N, C, H, W = y.shape
            y = y.reshape(N, -1)

        self.last_node.backward(y)

        self.dgamma = self.get_r.dv
        self.dbeta = self.get_b.dv
        self.dx = self.get_x.dv
        # print(self.dx)
        dx = self.dx.reshape(*self.input_shape)

        return dx


class Dropout(Layer):
    def __init__(self, dropout_ratio=0.5):
        super().__init__()
        self.dropout_ratio = dropout_ratio

        self.last_node = IndexMask(self.get_x, dropout_ratio)

    def forward(self, x, train_flg=True):
        self.get_x.v = x
        if train_flg:
            return self.last_node.forward()
        else:
            return x * (1.0 - self.dropout_ratio)
