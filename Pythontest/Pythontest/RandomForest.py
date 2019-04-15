import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import preprocessing
import re
from collections import defaultdict
from sklearn.model_selection import train_test_split
import DecisionTree as de

# 随机森林


class RandomForest:

   # 初始化
    def __init__(self, criterion='gini', max_depth=10, max_tree=20, random_sample=0.5):
        self.max_depth = max_depth  # 最大树深
        self.criterion = criterion  # 生成模式 ID3 或 ID4.5 或 gini
        self.max_tree = max_tree  # 最大生成树数
        self.random_sample = random_sample  # 随机样本比例
        self.forest = []  # 森林

    # 拟合函数
    def fit(self, x, y):
        data = np.hstack((x, y))
        for i in range(self.max_tree):
            ranData = self.randomSample(data)
            x2 = ranData[:, :-1]
            y2 = ranData[:, -1]
            model = de.DecisionTree(
                criterion=self.criterion, max_depth=self.max_depth)
            model.fit(x2, y2.reshape(len(y2), 1))
            self.forest.append(model)
        return self

    # 预测多个样本
    def predict(self, x):
        return np.array([self.hat(i) for i in x])

    # 预测单个样本
    def hat(self, x):
        result = 0
        account = 0
        ls = np.array([i.hat(x,i.tree) for i in self.forest])
        d = self.calculate_N(ls)
        for key, value in d.items():
            if value > account:
                account = value
                result = key
        return result

    # 随机样本选择器

    def randomSample(self, data):
        l = len(data)
        indexs = np.random.choice(l, int(l*self.random_sample))
        return np.array([data[i, :] for i in indexs])

        # 计算列表并分类
    def calculate_N(self, x):
        s = set(x.reshape(1, len(x)).tolist()[0])
        d = {}
        for i in s:
            d[i] = list(x).count(i)
        return d


iris_feature = u'花萼长度', u'花萼宽度', u'花瓣长度', u'花瓣宽度'
if __name__ == "__main__":

    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    path = u'D:\Python_machineLearn\Pythontest\Pythontest\8.iris.data'  # 数据文件路径
    df = pd.read_csv(path, header=0)
    x = df.values[:, :-1]
    y = df.values[:, -1]
    print('x = \n', x)
    print('y = \n', y)
    le = preprocessing.LabelEncoder()
    le.fit(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
    y = le.transform(y)
    # 为了可视化，仅使用前两列特征
    x = x[:, :2]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=1)

    # 决策树参数估计
    model = RandomForest(criterion='gini', max_depth=5)
    model = model.fit(x_train, y_train.reshape(len(y_train), 1))
    y_test_hat = model.predict(x_test)      # 测试数据

    # 保存
    # dot -Tpng -o 1.png 1.dot
    #f = open('.\\iris_tree.dot', 'w')
    #tree.export_graphviz(model.get_params('DTC')['DTC'], out_file=f)

    # 画图
    N, M = 100, 100  # 横纵各采样多少个值
    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()  # 第0列的范围
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()  # 第1列的范围
    t1 = np.linspace(x1_min, x1_max, N)
    t2 = np.linspace(x2_min, x2_max, M)
    x1, x2 = np.meshgrid(t1, t2)  # 生成网格采样点
    x_show = np.stack((x1.flat, x2.flat), axis=1)  # 测试点

    # # 无意义，只是为了凑另外两个维度
    # # 打开该注释前，确保注释掉x = x[:, :2]
    # x3 = np.ones(x1.size) * np.average(x[:, 2])
    # x4 = np.ones(x1.size) * np.average(x[:, 3])
    # x_test = np.stack((x1.flat, x2.flat, x3, x4), axis=1) # 测试点

    cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
    y_show_hat = model.predict(x_show)  # 预测值
    print("xshow=" + str(x_show))
    print("yshow=" + str(y_show_hat))
    y_show_hat = y_show_hat.reshape(x1.shape)  # 使之与输入的形状相同
    plt.figure(facecolor='w')
    plt.pcolormesh(x1, x2, y_show_hat, cmap=cm_light)  # 预测值的显示
    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test.ravel(),
                edgecolors='k', s=100, cmap=cm_dark, marker='o')  # 测试数据
    plt.scatter(x[:, 0], x[:, 1], c=y.ravel(),
                edgecolors='k', s=40, cmap=cm_dark)  # 全部数据
    plt.xlabel(iris_feature[0], fontsize=15)
    plt.ylabel(iris_feature[1], fontsize=15)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.grid(True)
    plt.title(u'鸢尾花数据的随机森林分类', fontsize=17)
    plt.show()

    # 训练集上的预测结果
    y_test = y_test.reshape(-1)
    print(str(y_test_hat))
    print(str(y_test))
    result = (y_test_hat == y_test)   # True则预测正确，False则预测错误
    acc = np.mean(result)
    print('准确度: %.2f%%' % (100 * acc))

    # 过拟合：错误率
    depth = np.arange(1, 15)
    err_list = []
    for d in depth:
        clf = RandomForest(criterion='gini', max_depth=d)
        clf = clf.fit(x_train, y_train.reshape(len(y_train), 1))
        y_test_hat = clf.predict(x_test)  # 测试数据
        result = (y_test_hat == y_test)  # True则预测正确，False则预测错误
        err = 1 - np.mean(result)
        err_list.append(err)
        print(d, ' 错误率: %.2f%%' % (100 * err))
    plt.figure(facecolor='w')
    plt.plot(depth, err_list, 'ro-', lw=2)
    plt.xlabel(u'决策树深度', fontsize=15)
    plt.ylabel(u'错误率', fontsize=15)
    plt.title(u'随机森林深度与过拟合', fontsize=17)
    plt.grid(True)
    plt.show()