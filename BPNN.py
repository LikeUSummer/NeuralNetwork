import random
import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def d_sigmoid(y):
    return y * (1 - y)

class BPNNClassification:
    def __init__(self, sizes):
        self.layer_count = len(sizes)
        self.sizes = sizes
        # 初始化偏差，除输入层外，其它每层每个节点都有一个 biase 值
        self.biases = [np.random.randn(n) for n in sizes[1:]]
        # 初始化权重，除输入层外，其他每层都有一个 r 行 c 列 的权重矩阵
        # c 为上一层神经元个数/本层输入维数，r 为本层神经元个数/本层输出维数
        self.weights = [np.random.randn(r, c) for r, c in zip(sizes[1:], sizes[:-1])]

    def feed_forward(self, input, full = False):
        outputs = [input]
        for w, b in zip(self.weights, self.biases):
            outputs.append(sigmoid(w @ outputs[-1] + b))
        return outputs if full else outputs[-1]

    def back_propagation(self, outputs, y):
        d_biases = [np.zeros(b.shape) for b in self.biases]
        d_weights = [np.zeros(w.shape) for w in self.weights]

        delta = (outputs[-1] - y) * d_sigmoid(outputs[-1]) # 输出层 δ
        d_biases[-1] = delta
        d_weights[-1] = np.outer(delta, outputs[-2])
        for k in range(2, self.layer_count):
            # 利用 k + 1 层的 δ 计算 k 层的 δ 
            delta = (self.weights[-k + 1].T @ delta) * d_sigmoid(outputs[-k])
            d_biases[-k] = delta
            d_weights[-k] = np.outer(delta, outputs[-k - 1])
        return (d_biases, d_weights)

    def train_batch(self, batch, eta):
        d_biases_accum = [np.zeros(b.shape) for b in self.biases]
        d_weights_accum = [np.zeros(w.shape) for w in self.weights]

        for x, y in batch:
            # 前向传播，求得各层神经元的输出值
            outputs = self.feed_forward(x, full = True)
            # 后向传播，递推计算梯度
            d_biases, d_weights = self.back_propagation(outputs, y)
            d_biases_accum = [dba + db for dba, db in zip(d_biases_accum, d_biases)]
            d_weights_accum = [dwa + dw for dwa, dw in zip(d_weights_accum, d_weights)]
        # 根据累积梯度 d_biases, d_weights 更新 weights 和 biases
        self.weights = [w - dw * eta / len(batch)
                        for w, dw in zip(self.weights, d_weights_accum)]
        self.biases = [b - db * eta / len(batch)
                        for b, db in zip(self.biases, d_biases_accum)]

    def train(self, train_data, epochs, batch_size, eta, test_data = None, tolerance = 0.0):
        for i in range(epochs):
            # 打乱训练集顺序
            random.shuffle(train_data)
            # 划分 batch
            mini_batchs = [train_data[j : j + batch_size]
                            for j in range(0, len(train_data), batch_size)]
            # 逐 batch 训练
            for batch in mini_batchs:
                self.train_batch(batch, eta)

            error = self.evaluate(test_data) if test_data else self.evaluate(train_data)
            print("Epoch {0}: {1}".format(i, error))
            if error <= tolerance:
                break
        return error

    def evaluate(self, test_data):
        test_result = [int(np.argmax(self.feed_forward(x)) == np.argmax(y)) for (x, y) in test_data]
        return 1 - sum(test_result) / len(test_data)

class BPNNRegression(BPNNClassification):
    '''
    神经网络回归与分类的差别在于：
    1. 输出层不需要再经过激活函数
    2. 输出层的 w 和 b 更新量计算相应更改
    '''
    def __init__(self, sizes):
        BPNNClassification.__init__(self, sizes)

    def feed_forward(self, input, full = False):
        outputs = BPNNClassification.feed_forward(self, input, True)
        outputs[-1] = self.weights[-1] @ outputs[-2] + self.biases[-1] # 和 Classification 的差异
        return outputs if full else outputs[-1]

    def back_propagation(self, outputs, y):
        d_biases = [np.zeros(b.shape) for b in self.biases]
        d_weights = [np.zeros(w.shape) for w in self.weights]

        delta = (outputs[-1] - y)
        d_biases[-1] = delta
        d_weights[-1] = np.outer(delta, outputs[-2]) # 和 Classification 的差异
        for k in range(2, self.layer_count):
            delta = (self.weights[-k + 1].T @ delta) * d_sigmoid(outputs[-k])
            d_biases[-k] = delta
            d_weights[-k] = np.outer(delta, outputs[-k - 1])
        return (d_biases, d_weights)

    def evaluate(self, test_data):
        test_result = [[self.feed_forward(x), y] for x, y in test_data]
        return np.sum([0.5 * (x - y) ** 2 for (x, y) in test_result])
