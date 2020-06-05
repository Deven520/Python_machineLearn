
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import preprocessing
import re
from collections import defaultdict
from sklearn.model_selection import train_test_split
import sklearn.datasets as ds
import matplotlib.colors

#K-Mean
class KMean(object):
    """K-Mean 算法实现"""

    # 初始化
    def __init__(self,traindata=np.array([]),n_clusters=2, kernel='LINE',mode='RANDOM'):
        self.kernel = kernel # 计算距离方式
        self.traindata = traindata #训练数据
        self.n_clusters = n_clusters #簇数
        self.mode = mode
        self.centers = np.array([])
        self.y_hat = []

    # 拟合函数
    def fit_predict(self,data):
        self.traindata = data
        self.centers = self.cal_firstChoose()
        y_hat = self.cal_classify(self.centers)
        while self.y_hat != y_hat:
            self.y_hat = y_hat
            self.cal_centers()
            y_hat = self.cal_classify(self.centers)
        return self.y_hat

    # 预测多个样本
    def predict(self, x):
        y_hat = np.array([])
        for i in range(len(x)):
            y_hatone = self.predict_one(x[i])
            y_hat = np.append(y_hat,y_hatone)
        return y_hat

    #预测一个样本
    def predict_one(self,x):
        y_hat = 0.0
        for i in range(len(self.a)):
            y_hat += self.a[i] * self.yl[i] * self.cal_kenrel(x,self.xl[i])
        y_hat+=self.b
        if y_hat >= 0:
            y_hat = 1
        else:
            y_hat = -1
        return y_hat

    #计算两点距离
    def cal_distanc(self,x,y):
        distanc = 100000
        # 笛卡尔距离
        if self.kernel.upper() == 'LINE':
            distanc = np.sqrt(np.sum(np.square(x - y)))
        elif self.kernel.upper() == 'LINE':
            distanc = (np.inner(x , y)) ** 0.5
        return distanc
    
    #初始簇中心选择
    def cal_firstChoose(self):
        if self.mode == 'RANDOM':
            randomInt = np.random.randint(0,len(self.traindata),self.n_clusters)
            centers = self.traindata[randomInt]
            return centers
        elif self.mode == 'K++':
            return
        return

    #计算簇中心-one
    def cal_center(self,x=[]):
        if len(x) == 0:
            return (-100000,-100000)
        sum = np.sum(x,axis=0)
        center = sum / len(x)
        return center

    #计算簇中心-all
    def cal_centers(self):
        centers = np.array([])
        xs = []
        for j in range(self.n_clusters):
            xs.append([])
        for i in range(len(self.traindata)):
            xs[self.y_hat[i]].append(self.traindata[i])
        for k in range(self.n_clusters):
                l = self.cal_center(xs[k])
                l = np.array(l)
                centers = np.append(centers,l,0)
        self.centers = centers.reshape(self.n_clusters,2)
        return self

    #计算距离并分类
    def cal_classify(self,x):
        y_hat = []
        # 遍历计算距离
        for j in range(len(self.traindata)):
            y_dist = np.array([])
            for i in range(len(x)):
                y_dist = np.append(y_dist,self.cal_distanc(x[i],self.traindata[j]))
            g = np.argwhere(y_dist == y_dist.min())
            y_hat.append(int(g[0]))
        return y_hat

def expand(a, b):
    d = (b - a) * 0.1
    return a - d, b + d


if __name__ == "__main__":
    N = 400
    centers = 4
    data, y = ds.make_blobs(N, n_features=2, centers=centers, random_state=2)
    data2, y2 = ds.make_blobs(N, n_features=2, centers=centers, cluster_std=(1,2.5,0.5,2), random_state=2)
    data3 = np.vstack((data[y == 0][:], data[y == 1][:50], data[y == 2][:20], data[y == 3][:5]))
    y3 = np.array([0] * 100 + [1] * 50 + [2] * 20 + [3] * 5)

    cls = KMean(n_clusters=4)
    y_hat = cls.fit_predict(data)
    y2_hat = cls.fit_predict(data2)
    y3_hat = cls.fit_predict(data3)

    m = np.array(((1, 1), (1, 3)))
    data_r = data.dot(m)
    y_r_hat = cls.fit_predict(data_r)

    matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    cm = matplotlib.colors.ListedColormap(list('rgbm'))

    plt.figure(figsize=(9, 10), facecolor='w')
    plt.subplot(421)
    plt.title(u'原始数据')
    plt.scatter(data[:, 0], data[:, 1], c=y, s=30, cmap=cm, edgecolors='none')
    x1_min, x2_min = np.min(data, axis=0)
    x1_max, x2_max = np.max(data, axis=0)
    x1_min, x1_max = expand(x1_min, x1_max)
    x2_min, x2_max = expand(x2_min, x2_max)
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(True)

    plt.subplot(422)
    plt.title(u'KMeans++聚类')
    plt.scatter(data[:, 0], data[:, 1], c=y_hat, s=30, cmap=cm, edgecolors='none')
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(True)

    plt.subplot(423)
    plt.title(u'旋转后数据')
    plt.scatter(data_r[:, 0], data_r[:, 1], c=y, s=30, cmap=cm, edgecolors='none')
    x1_min, x2_min = np.min(data_r, axis=0)
    x1_max, x2_max = np.max(data_r, axis=0)
    x1_min, x1_max = expand(x1_min, x1_max)
    x2_min, x2_max = expand(x2_min, x2_max)
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(True)

    plt.subplot(424)
    plt.title(u'旋转后KMeans++聚类')
    plt.scatter(data_r[:, 0], data_r[:, 1], c=y_r_hat, s=30, cmap=cm, edgecolors='none')
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(True)

    plt.subplot(425)
    plt.title(u'方差不相等数据')
    plt.scatter(data2[:, 0], data2[:, 1], c=y2, s=30, cmap=cm, edgecolors='none')
    x1_min, x2_min = np.min(data2, axis=0)
    x1_max, x2_max = np.max(data2, axis=0)
    x1_min, x1_max = expand(x1_min, x1_max)
    x2_min, x2_max = expand(x2_min, x2_max)
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(True)

    plt.subplot(426)
    plt.title(u'方差不相等KMeans++聚类')
    plt.scatter(data2[:, 0], data2[:, 1], c=y2_hat, s=30, cmap=cm, edgecolors='none')
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(True)

    plt.subplot(427)
    plt.title(u'数量不相等数据')
    plt.scatter(data3[:, 0], data3[:, 1], s=30, c=y3, cmap=cm, edgecolors='none')
    x1_min, x2_min = np.min(data3, axis=0)
    x1_max, x2_max = np.max(data3, axis=0)
    x1_min, x1_max = expand(x1_min, x1_max)
    x2_min, x2_max = expand(x2_min, x2_max)
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(True)

    plt.subplot(428)
    plt.title(u'数量不相等KMeans++聚类')
    plt.scatter(data3[:, 0], data3[:, 1], c=y3_hat, s=30, cmap=cm, edgecolors='none')
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(True)

    plt.tight_layout(2)
    plt.suptitle(u'数据分布对KMeans聚类的影响', fontsize=18)
    # https://github.com/matplotlib/matplotlib/issues/829
    plt.subplots_adjust(top=0.92)
    plt.show()


        