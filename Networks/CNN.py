import pickle

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from Layers.layers import *


class CNN:
    """Convolution 신경망
    기본설정:
        Batch Norm 사용 안 함
        DropOut 사용 안 함
    모델종류:
        CVD, VGG11, VGG16
    CVD:
        1 conv - batch - relu -
        2 conv - batch - relu - pooling
        3 conv - batch - relu -
        4 conv - batch - relu - pooling
        5 conv - batch - relu -
        6 conv - batch - relu - pooling
        7 affine - relu - dropout
        8 affine - softmax
    VGG11:
        1 conv - batch - relu - pooling
        2 conv - batch - relu - pooling
        3 conv - batch - relu -
        4 conv - batch - relu - pooling
        5 conv - batch - relu -
        6 conv - batch - relu - pooling
        7 conv - batch - relu -
        8 conv - batch - relu - pooling
        9 affine - relu - dropout
        10 affine - relu - dropout
        11 affine - softmax
    VGG16:
        1 conv - batch - relu -
        2 conv - batch - relu - pooling
        3 conv - batch - relu -
        4 conv - batch - relu - pooling
        5 conv - batch - relu -
        6 conv - batch - relu -
        7 conv - batch - relu - pooling
        8 conv - batch - relu -
        9 conv - batch - relu -
        10 conv - batch - relu - pooling
        11 conv - batch - relu -
        12 conv - batch - relu -
        13 conv - batch - relu - pooling
        14 affine - relu - dropout
        15 affine - relu - dropout
        16 affine - softmax
    """

    def __init__(self, input_dim=(1, 28, 28), output_size=10, dropout_ratio=0, model='VGG16', init_fn=64, act='relu',
                 use_batchnorm=False,
                 back_eta=True):
        # 가중치 초기화===========
        param_size = []
        if model == 'CVD':
            init_fn = init_fn//4
            param_size = [
                {'filter_num': init_fn, 'filter_size': 3, 'pad': 1, 'stride': 1},  # 16, 28, 28
                {'filter_num': init_fn, 'filter_size': 3, 'pad': 1, 'stride': 1},  # 16, 28, 28
                {'pooling': 2, 'stride': 2},  # 16, 14, 14
                {'filter_num': 2*init_fn, 'filter_size': 3, 'pad': 1, 'stride': 1},  # 32, 14, 14
                {'filter_num': 2*init_fn, 'filter_size': 3, 'pad': 2, 'stride': 1},  # 32, 16, 16
                {'pooling': 2, 'stride': 2},  # 32, 8, 8
                {'filter_num': 4*init_fn, 'filter_size': 3, 'pad': 1, 'stride': 1},  # 64, 8, 8
                {'filter_num': 4*init_fn, 'filter_size': 3, 'pad': 1, 'stride': 1},  # 64, 8, 8
                {'pooling': 2, 'stride': 2},  # 64, 4, 4
                {'flatten': True},  # pre_channel_num 용. 실제 Flatten 은 Affine 에 구현 되어있음
                # dense
                {'unit_num': 50},
                {'unit_num': output_size}
            ]
        elif model == 'VGG11':
            param_size = [
                {'filter_num': init_fn, 'filter_size': 3, 'pad': 1, 'stride': 1},  # 64, 28, 28
                {'pooling': 2, 'stride': 2},  # 64, 14, 14
                {'filter_num': 2*init_fn, 'filter_size': 3, 'pad': 2, 'stride': 1},  # 128, 16, 16
                {'pooling': 2, 'stride': 2},  # 128, 8, 8
                {'filter_num': 4*init_fn, 'filter_size': 3, 'pad': 1, 'stride': 1},  # 256, 8, 8
                {'filter_num': 4*init_fn, 'filter_size': 3, 'pad': 1, 'stride': 1},
                {'pooling': 2, 'stride': 2},  # 128, 4, 4
                {'filter_num': 8*init_fn, 'filter_size': 3, 'pad': 1, 'stride': 1},  # 512, 4, 4
                {'filter_num': 8*init_fn, 'filter_size': 3, 'pad': 1, 'stride': 1},
                {'pooling': 2, 'stride': 2},  # 256, 2, 2
                {'filter_num': 8*init_fn, 'filter_size': 3, 'pad': 1, 'stride': 1},  # 512, 2, 2
                {'filter_num': 8*init_fn, 'filter_size': 3, 'pad': 1, 'stride': 1},
                {'pooling': 2, 'stride': 2},  # 512, 1, 1
                {'flatten': True},  # pre_channel_num 용. 실제 Flatten 은 Affine 에 구현 되어있음
                # dense
                {'unit_num': init_fn*init_fn},
                {'unit_num': 4*init_fn},
                {'unit_num': output_size}
            ]
        elif model == 'VGG16':
            param_size = [
                {'filter_num': init_fn, 'filter_size': 3, 'pad': 1, 'stride': 1},  # 64, 28, 28
                {'filter_num': init_fn, 'filter_size': 3, 'pad': 1, 'stride': 1},  # 64, 28, 28
                {'pooling': 2, 'stride': 2},  # 64, 14, 14
                {'filter_num': 2*init_fn, 'filter_size': 3, 'pad': 1, 'stride': 1},  # 128, 14, 14
                {'filter_num': 2*init_fn, 'filter_size': 3, 'pad': 2, 'stride': 1},  # 128, 16, 16
                {'pooling': 2, 'stride': 2},  # 128, 8, 8
                {'filter_num': 4*init_fn, 'filter_size': 3, 'pad': 1, 'stride': 1},  # 256, 8, 8
                {'filter_num': 4*init_fn, 'filter_size': 3, 'pad': 1, 'stride': 1},
                {'filter_num': 4*init_fn, 'filter_size': 3, 'pad': 1, 'stride': 1},
                {'pooling': 2, 'stride': 2},  # 128, 4, 4
                {'filter_num': 8*init_fn, 'filter_size': 3, 'pad': 1, 'stride': 1},  # 512, 4, 4
                {'filter_num': 8*init_fn, 'filter_size': 3, 'pad': 1, 'stride': 1},
                {'filter_num': 8*init_fn, 'filter_size': 3, 'pad': 1, 'stride': 1},
                {'pooling': 2, 'stride': 2},  # 256, 2, 2
                {'filter_num': 8*init_fn, 'filter_size': 3, 'pad': 1, 'stride': 1},  # 512, 2, 2
                {'filter_num': 8*init_fn, 'filter_size': 3, 'pad': 1, 'stride': 1},
                {'filter_num': 8*init_fn, 'filter_size': 3, 'pad': 1, 'stride': 1},
                {'pooling': 2, 'stride': 2},  # 512, 1, 1
                {'flatten': True},  # pre_channel_num 용. 실제 Flatten 은 Affine 에 구현 되어있음
                # dense
                {'unit_num': init_fn*init_fn},
                {'unit_num': 4*init_fn},
                {'unit_num': output_size}
            ]
        else:
            assert 'No Model'

        # 가중치 초기화
        height, width = input_dim[1], input_dim[2]
        pre_channel_num = input_dim[0]
        self.params = {}
        std = 2.0
        if act == 'sigmoid':
            std = 1.0

        i = 1
        for _, ps in enumerate(param_size):
            if 'filter_num' in ps:  # Convolutional Layer 초기화
                filter_size = ps['filter_size']
                filter_num = ps['filter_num']
                pad = ps['pad']
                stride = ps['stride']
                scale = np.sqrt(std / (pre_channel_num * filter_size * filter_size))  # ReLU를 사용할 때의 권장 초깃값

                self.params['W' + str(i)] = scale * np.random.randn(filter_num, pre_channel_num, filter_size,
                                                                    filter_size)
                self.params['b' + str(i)] = np.zeros(filter_num)

                height = 1 + (height + 2 * pad - filter_size) // stride
                width = 1 + (width + 2 * pad - filter_size) // stride
                if use_batchnorm:
                    self.params['gamma' + str(i)] = np.ones((filter_num, height, width))
                    self.params['beta' + str(i)] = np.zeros((filter_num, height, width))
                pre_channel_num = filter_num
                i += 1
            elif 'pooling' in ps:  # Max Pooling Layer 크기 변경
                height //= ps['pooling']
                width //= ps['pooling']
            elif 'flatten' in ps:
                pre_channel_num = pre_channel_num * height * width
            elif 'unit_num' in ps:  # dense Layer 초기화
                scale = np.sqrt(std / pre_channel_num)
                self.params['W' + str(i)] = scale * np.random.randn(pre_channel_num, ps['unit_num'])
                self.params['b' + str(i)] = np.zeros(ps['unit_num'])
                pre_channel_num = ps['unit_num']
                i += 1

        # 계층 생성===========
        self.layers = []
        i = 1
        for idx, ps in enumerate(param_size):
            if 'filter_num' in ps:
                # Convolutional Layer
                W, b = self.params['W' + str(i)], self.params['b' + str(i)]
                self.layers.append(Convolution(W, b, ps['stride'], ps['pad']))

                # BatchNormalization Layer
                if use_batchnorm:
                    gamma, beta = self.params['gamma' + str(i)], self.params['beta' + str(i)]
                    self.layers.append(BatchNormalization(gamma=gamma, beta=beta))

                # Activation Layer
                if act == 'relu':
                    self.layers.append(Relu())
                else:
                    self.layers.append(Sigmoid())

                i += 1
            elif 'pooling' in ps:
                # Max Pooling Layer
                self.layers.append(Pooling(pool_h=ps['pooling'], pool_w=ps['pooling'], stride=ps['stride']))
            elif 'unit_num' in ps:
                # Affine(dense) Layer
                W, b = self.params['W' + str(i)], self.params['b' + str(i)]
                self.layers.append(Affine(W, b))

                if idx == len(param_size) - 1:
                    # last Layer 는 활성화 따로 구현
                    break

                # Activation Layer
                if act == 'relu':
                    self.layers.append(Relu())
                else:
                    self.layers.append(Sigmoid())

                # Dropout Layer
                if dropout_ratio != 0:
                    self.layers.append(Dropout(dropout_ratio))
                i += 1

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

    def accuracy(self, x, t):
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        if x.shape[0] != t.shape[0]:
            assert 'Batch size unmatched'

        y = self.predict(t, train_flg=False)
        yp = np.argmax(y, axis=1)
        acc = np.sum(yp == t) / float(x.shape[0])

        return acc / x.shape[0]

    def acc_and_loss(self, x, t):
        if x.shape[0] != t.shape[0]:
            assert 'Batch size unmatched'

        # loss
        y = self.predict(x, train_flg=False)
        loss = self.last_layer.forward(y, t)

        # acc
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
        i = 1
        bi = 1
        for layer in self.layers:
            if isinstance(layer, (Convolution, Affine)):
                grads['W' + str(i)] = layer.dW
                grads['b' + str(i)] = layer.db
                i += 1
            elif isinstance(layer, BatchNormalization):
                grads['gamma' + str(bi)] = layer.dgamma
                grads['beta' + str(bi)] = layer.dbeta
                bi += 1

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

        i = 1
        bi = 1
        for layer in self.layers:
            if isinstance(layer, (Convolution, Affine)):
                layer.get_w.v = params['W' + str(i)]
                layer.get_w.v = params['b' + str(i)]
                i += 1
            elif isinstance(layer, BatchNormalization):
                layer.get_r.v = params['gamma' + str(i)]
                layer.get_b.v = params['beta' + str(i)]
                bi += 1
