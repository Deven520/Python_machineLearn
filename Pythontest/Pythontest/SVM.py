import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import preprocessing
import re
from collections import defaultdict
from sklearn.model_selection import train_test_split

#SVM
class SVM(object):
    """支持向量积算法简单实现"""

    # 初始化
    def __init__(self,traindata=[], kernel='kernel',maxIter=1000,C=1,epsilon=''):
        self.kernel = kernel #核函数
        self.traindata = traindata #训练数据
        self.maxIter = maxIter #最大项
        self.C = C #惩罚因子
        self.epsilon = epsilon
        self.w = np.array([0 for i in range(len(traindata[0]) - 1)]) #w参数
        self.a = np.array([0 for i in range(len(traindata))]) #a参数
        self.b = 0 #np.array([0 for i in range(len(traindata))]).reshape(len(traindata),1) #b参数
        self.xl = traindata[:,:-1] #训练数据x
        self.yl = traindata[:,-1:] #训练数据结果y

    # 拟合函数
    def fit(self, x, y):
        
        return self

    #smo算法
    def SMO(self):

        return self

    # 预测多个样本
    def predict(self, x):
        y_hat = np.dot(x,self.w) + self.b
        return y_hat

    # 根据a值得出w
    def update_w(self):
        for i in range(len(self.xl)):
            wi = self.a[i] * self.yl[i] * self.xl[i,:]
            self.w = self.w + wi
        return self

if __name__ == "__main__":
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    x = np.array([1,1,2,3,2,2,2,1])
    y = np.array([1,1,-1,-1])
    traindata = np.hstack((x.reshape(4,2), y.reshape(4,1)))
    print(str(traindata))
    model = SVM(traindata=traindata)
    print(str(model.w))
    print(str(model.a))
    print(str(model.xl))
    print(str(model.yl))
    y_hat = model.predict(model.xl)
    print(str(y_hat))
    model.update_w()
    print(str(model.w))
