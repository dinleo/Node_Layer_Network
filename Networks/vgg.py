import pickle

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from Layers.layers import *


class VGG:
    """VGG 신경망

    네트워크 구성은 아래와 같음
        conv - relu - conv- relu - pool -
        conv - relu - conv- relu - pool -
        conv - relu - conv- relu - pool -
        affine - relu - dropout - affine - dropout - softmax
    """

    def __init__(self, input_dim=(1, 28, 28), output_size=10, dropout_ratio=0.5, back_eta=True):
        # 가중치 초기화===========
        # 각 층의 뉴런 하나당 앞 층의 몇 개 뉴런과 연결되는가（TODO: 자동 계산되게 바꿀 것）
        param_size = [
            {'filter_num': 64, 'filter_size': 3, 'pad': 2, 'stride': 1},  # 64, 30, 30
            {'filter_num': 64, 'filter_size': 3, 'pad': 2, 'stride': 1},  # 63, 32, 32
            {'pooling': 2, 'stride': 2},  # 64, 16, 16
            {'filter_num': 128, 'filter_size': 3, 'pad': 1, 'stride': 1},
            {'filter_num': 128, 'filter_size': 3, 'pad': 1, 'stride': 1},
            {'pooling': 2, 'stride': 2},  # 128, 8, 8
            {'filter_num': 256, 'filter_size': 3, 'pad': 1, 'stride': 1},
            {'filter_num': 256, 'filter_size': 3, 'pad': 1, 'stride': 1},
            {'filter_num': 256, 'filter_size': 3, 'pad': 1, 'stride': 1},
            {'pooling': 2, 'stride': 2},  # 128, 4, 4
            {'filter_num': 512, 'filter_size': 3, 'pad': 1, 'stride': 1},
            {'filter_num': 512, 'filter_size': 3, 'pad': 1, 'stride': 1},
            {'filter_num': 512, 'filter_size': 3, 'pad': 1, 'stride': 1},
            {'pooling': 2, 'stride': 2},  # 256, 2, 2
            {'filter_num': 512, 'filter_size': 3, 'pad': 1, 'stride': 1},
            {'filter_num': 512, 'filter_size': 3, 'pad': 1, 'stride': 1},
            {'filter_num': 512, 'filter_size': 3, 'pad': 1, 'stride': 1},
            {'pooling': 2, 'stride': 2},  # 512, 1, 1
            # dense
            {'unit_num': 4095},
            {'unit_num': 4095},
            {'unit_num': output_size}
        ]

        # 가중치 초기화
        height, width = input_dim[1], input_dim[2]
        pre_channel_num = input_dim[0]
        self.params = {}

        for idx, ps in enumerate(param_size):
            if 'filter_num' in ps:  # Convolutional Layer 초기화
                filter_size = ps['filter_size']
                filter_num = ps['filter_num']
                pad = ps['pad']
                stride = ps['stride']
                scale = np.sqrt(2.0 / (pre_channel_num * filter_size * filter_size))  # ReLU를 사용할 때의 권장 초깃값

                self.params['W' + str(idx + 1)] = scale * np.random.randn(filter_num, pre_channel_num,
                                                                          filter_size, filter_size)
                self.params['b' + str(idx + 1)] = np.zeros(filter_num)
                self.params['gamma' + str(idx + 1)] = np.ones(filter_num)
                self.params['beta' + str(idx + 1)] = np.zeros(filter_num)

                pre_channel_num = filter_num
                height = 1 + (height + 2 * pad - filter_size) // stride
                width = 1 + (width + 2 * pad - filter_size) // stride
            elif 'pooling' in ps:  # Max Pooling Layer 크기 변경
                height //= ps['pooling']
                width //= ps['pooling']
            elif 'unit_num' in ps:  # dense Layer 초기화
                if 'pooling' in param_size[idx - 1]:
                    pre_channel_num = pre_channel_num * height * width
                scale = np.sqrt(2.0 / pre_channel_num)
                self.params['W' + str(idx + 1)] = scale * np.random.randn(pre_channel_num, ps['unit_num'])
                self.params['b' + str(idx + 1)] = np.zeros(ps['unit_num'])
                pre_channel_num = ps['unit_num']

        # 계층 생성===========
        self.layers = []
        for idx, ps in enumerate(param_size):
            if 'filter_num' in ps:  # Convolutional Layer
                W, b = self.params['W' + str(idx + 1)], self.params['b' + str(idx + 1)]
                gamma, beta = self.params['gamma' + str(idx + 1)], self.params['beta' + str(idx + 1)]

                self.layers.append(Convolution(W, b, ps['stride'], ps['pad']))
                self.layers.append(BatchNormalization(gamma=gamma, beta=beta))
                self.layers.append(Relu())
            elif 'pooling' in ps:  # Max Pooling Layer
                self.layers.append(Pooling(ps['pooling'], ps['stride']))
            elif 'unit_num' in ps:  # dense Layer
                W, b = self.params['W' + str(idx + 1)], self.params['b' + str(idx + 1)]

                self.layers.append(Affine(W, b))
                if idx < len(param_size) - 1:  # 마지막 dense Layer가 아닌 경우에만 ReLU 적용
                    self.layers.append(Relu())
                    self.layers.append(Dropout(dropout_ratio))

        self.last_layer = SoftmaxWithLoss(back_eta=back_eta)

    def predict(self, x, train_flg=False):
        for layer in self.layers:
            if isinstance(layer, Dropout):
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t, train_flg=False):
        y = self.predict(x, train_flg)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=128):
        if t.ndim != 1: t = np.argmax(t, axis=1)

        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i * batch_size:(i + 1) * batch_size]
            tt = t[i * batch_size:(i + 1) * batch_size]
            y = self.predict(tx, train_flg=False)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]

    def acc_and_loss(self, x, t):
        y = self.predict(x, train_flg=False)
        loss = self.last_layer.forward(y, t)

        yp = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        acc = np.sum(yp == t) / float(x.shape[0])

        return acc, loss

    def gradient(self, x, t):
        # forward
        self.loss(x, t, train_flg=True)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        tmp_layers = self.layers.copy()
        tmp_layers.reverse()
        for layer in tmp_layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        for i, layer_idx in enumerate((0, 2, 5, 7, 10, 12, 15, 18)):
            grads['W' + str(i + 1)] = self.layers[layer_idx].dW
            grads['b' + str(i + 1)] = self.layers[layer_idx].db

        return grads

    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, layer_idx in enumerate((0, 2, 5, 7, 10, 12, 15, 18)):
            self.layers[layer_idx].W = self.params['W' + str(i + 1)]
            self.layers[layer_idx].b = self.params['b' + str(i + 1)]
