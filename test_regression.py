import BPNN
import numpy as np

def func(x):
    return 2 * x ** 2 + 3 * x + 5

def func_regression():
    xs = np.linspace(-4, 4, 50)
    ys = func(xs) + np.random.randn(50) * 0.01
    train_data = [[np.array([x]), np.array([y])] for x, y in zip(xs, ys)]

    xs = np.array([2.6, 1.3, 0.3, -2.5, -3.7])
    ys = func(xs)
    test_data = [[np.array([x]), np.array([y])] for x, y in zip(xs, ys)]

    nn = BPNN.BPNNRegression([1, 10, 1])
    nn.train(train_data, 5000, 10, 0.02, test_data = test_data, tolerance = 0.01)
    print([[nn.feed_forward(x), y] for x, y in test_data])

def douban_regression():
    interval = 10
    dataset = np.loadtxt('douban_rate.csv', dtype = float, delimiter=',')
    input = dataset[::interval, :-2]
    output = dataset[::interval, -2]
    data = [[sx, sy] for sx, sy in zip (input, output)]
    train_data = data[:-1000]
    test_data = data[-1000:]

    nn = BPNN.BPNNRegression([5, 10, 1])
    nn.train(train_data, 1000, 10, 0.1, test_data = test_data, tolerance = 0.01)

    '''
    errs = []
    for batch_size in [1, 10, 100, 1000, 10000, len(train_data)]:
        nn = BPNN.BPNNRegression([5, 7, 1])
        errs.append(nn.train(train_data, 1000, batch_size, 0.05))
    print(errs)
    '''

    '''
    errs = []
    for rate in [0.01, 0.05, 0.1, 0.25, 0.5, 0.75]:
        nn = BPNN.BPNNRegression([5, 7, 1])
        errs.append(nn.train(train_data, 1000, len(train_data), rate))
    print(errs)
    '''

func_regression()
# douban_regression()
