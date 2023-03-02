from BPNN import BPNNClassification
from sklearn import datasets
import numpy as np

def iris_classification():
    # 从本地文件加载鸢尾花数据
    dataset = np.loadtxt('IrisDataSet.csv', dtype = float, delimiter=',')
    data = [[x, np.eye(3)[int(y)]] for x, y in zip(dataset[:, :-1], dataset[:, -1])]
    train_data = data[:-30]
    test_data = data[-30:]

    '''
    # 从 sklearn 提供的数据集加载鸢尾花数据
    iris = datasets.load_iris()
    state = np.random.get_state()
    np.random.shuffle(iris.data)
    np.random.set_state(state)
    np.random.shuffle(iris.target)
    data = [[x, np.eye(3)[y]] for x, y in zip(iris.data, iris.target)]
    train_data = data[:-30]
    test_data = data[-30:]
    '''

    nn = BPNNClassification([4, 15, 3])
    nn.train(train_data, 1000, 10, 0.05, test_data = test_data)

def douban_classification():
    # 从本地文件加载豆瓣评分数据
    dataset = np.loadtxt('douban_rate.csv', dtype = float, delimiter=',')
    data = [[x, np.eye(4)[int(y)]] for x, y in zip(dataset[:, :-2], dataset[:, -1])]
    train_data = data[:-50000]
    test_data = data[-50000:]

    nn = BPNNClassification([5, 15, 4])
    nn.train(train_data, 5000, 20, 0.2, test_data = test_data)

iris_classification()
