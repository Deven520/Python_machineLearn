import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


# 梯度下降算法
class gradientDescent:

    def __init__(self, poly__degree=1):
        self.alpha = 10 ** (-(poly__degree + 1))  # 步长
        self.poly__degree = poly__degree  # 线性几阶
        self.theta = []  # 训练结果
        self.TSS = 0  # 总平方和
        self.RSS = 0  # 残差平方和
        self.R2 = 0  # 拟合度量值

    # 梯度计算函数
    def gradient_function(self, theta, x, y):
        diff = np.dot(x, theta)  # 向量相乘
        m = len(y)
        return (1 / m) * np.dot(np.transpose(x), (diff - y))

    # 计算平方差和
    def calculateDiffOSqu(self, x1, x2):
        diff = (x1 - x2) ** 2
        return np.sum(diff)

    # 计算平均值
    def calculateAverage(self, x):
        N = len(x)
        return np.sum(x) / N

    # 计算拟合度
    def score(self, x, y):
        N = len(y)
        average = self.calculateAverage(y)
        averagelist = np.array([average for i in range(N)]).reshape(N, 1)
        self.TSS = self.calculateDiffOSqu(y, averagelist)
        y_hat = self.predict(x)
        self.RSS = self.calculateDiffOSqu(y, y_hat)
        self.R2 = 1 - self.RSS / self.TSS
        return self.R2

    # 拟合函数
    def fit(self, x, y):
        d = self.poly__degree + 1
        theta0 = np.array([1])
        N = len(y)
        X0 = np.ones((N, 1)).reshape(N, 1)  # 全1项
        X = X0
        for i in range(d - 1):
            theta0 = np.hstack((theta0, 1))  # 初始值
            X = np.hstack((X, x ** (i + 1)))
        self.theta = theta0.reshape(d, 1)
        gradient = self.gradient_function(self.theta, X, y)
        for times in range(10 ** (self.poly__degree + 2)):
            self.theta = self.theta - self.alpha * gradient
            gradient = self.gradient_function(self.theta, X, y)
        # 计算拟合度量值
        self.score(x, y)
        return self

    # 预测函数
    def predict(self, x):
        d = self.poly__degree + 1
        N = len(x)
        y = np.zeros((N, 1)).reshape(N, 1)  # 全1项
        for i in range(d):
            y = y + self.theta[i] * x ** i
        return y

    # 设置参数
    def set_params(self, poly__degree=1):
        self.poly__degree = poly__degree  # 线性几阶
        self.alpha = 10 ** (-(poly__degree + 1))  # 步长
        return self



if __name__ == '__main__':
    np.random.seed(0)
    N = 9
    X1 = np.linspace(0, 6, N) + np.random.randn(N)
    X1 = np.sort(X1)
    y = X1 ** 2 - 4 * X1 - 3 + np.random.randn(N)
    y1 = X1 ** 2 - 4 * X1 - 3
    X1 = X1.reshape(N, 1)
    y = y.reshape(N, 1)
    y1 = y1.reshape(N, 1)
    X0 = np.ones((N, 1))
    x = X1
    # print('x=' + str(x))
    # print('y=' + str(y))
    model = gradientDescent()
    model.set_params(poly__degree=3)
    model.fit(x=x, y=y)
    theta = model.__getattribute__('theta')
    alpha = model.__getattribute__('alpha')
    R2 = model.__getattribute__('R2')
    print('theta=' + str(theta))
    print('alpha=' + str(alpha))
    print('R2=' + str(R2))
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.legend(loc='gradient')
    x_hat = np.linspace(0, 7, 7).reshape(7, 1)
    y_hat = model.predict(x=x_hat)
    plt.plot(x_hat, y_hat, 'g-', linewidth=2)
    plt.plot(X1, y1, 'y-', linewidth=2)
    plt.plot(x, y, 'r*', ms=10)
    plt.grid()
    plt.show()
