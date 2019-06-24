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
    def __init__(self,traindata=[], kernel='kernel',maxIter=1000,C=1,epsilon=0,r=1):
        self.kernel = kernel #核函数
        self.traindata = traindata #训练数据
        self.maxIter = maxIter #最大项
        self.C = C #惩罚因子
        self.epsilon = epsilon
        self.w = np.array([0 for i in range(len(traindata[0]) - 1)]) #w参数
        self.a = np.array([0 for i in range(len(traindata))]) #a参数
        self.b = 0  #b参数
        self.xl = traindata[:,:-1] #训练数据x
        self.yl = traindata[:,-1:] #训练数据结果y
        self.eCache = np.array([0 for i in range(len(traindata))]) # e 的缓存
        self.r = r #高斯核函数超参数r

    # 拟合函数
    def fit(self):
        while self.SMO() == 0:
            i = 0
        return self

    # 预测多个样本
    def predict(self, x):
        y_hat = np.array([])
        for i in range(len(x)):
            y_hatone = self.predict_one(x[i])
            y_hat = y_hat + y_hatone
        return y_hat

    #预测一个样本
    def predict_one(self,x):
        y_hat = 0
        for i in range(len(self.a)):
            y_hat+=self.a[i] * self.yl[i] * self.cal_kenrel(x,self.xl[i])
        y_hat+=self.b
        if y_hat >= 0:
            y_hat = 1
        else:
            y_hat = -1
        return y_hat

    # 根据a值更新w
    def update_w(self):
        for i in range(len(self.xl)):
            wi = self.a[i] * self.yl[i] * self.xl[i,:]
            self.w = self.w + wi
        return self

    #根据kkt条件选择a1和a2并计算更新
    #smo算法
    def SMO(self):
        i_a1_sel = 0
        max_error1 = 0
        max_error2 = 0
        max_error3 = 0
        first_sel = 0
        #遍历选择符合kkt条件的a1
        for i in range(len(self.a)):
            xi = self.xl[i]
            yi = self.yl[i]
            ai = self.a[i]
            error = self.cal_gx(i) * yi #误差
            #优先选择违反0<ai<C => yi*gi=1
            if ai > 0 and ai < sself.C and error != 1 and (1 - error) ** 2 > max_error1:
                max_error1 = (1 - error) ** 2
                i_a1_sel = i
                first_sel = 1
            elif ai == 0 and error < 1 and (1 - error) ** 2 > max_error2 and first_sel == 0:
                max_error1 = (1 - error) ** 2
                i_a1_sel = i
            elif ai == self.C and error > 1 and (1 - error) ** 2 > max_error2 and first_sel == 0:
                max_error1 = (1 - error) ** 2
                i_a1_sel = i
        #遍历选择a2
        i_a2_sel = 0
        max_error4 = 0
        e1 = self.eCache[i_a1_sel]
        for i in range(len(self.a)):
            error = (e1 - self.eCache[i]) ** 2
            if error > max_error4:
                i_a2_sel = i
                max_error4 = error
        #判断是否可以跳出循环
        isover = 0
        if i_a1_sel == 0 and max_error1 == 0 and max_error2 == 0 and max_error2 == 0:
            isover = 1
        else:
            self.cal_b(i_a1_sel,i_a2_sel)
            self.cal_E()
        return isover

    #计算g(xi)
    def cal_gx(self,i):
        gi = 0
        xi = self.xl[i]
        for j in range(len(self.a)):
            gi+=self.a[j] * self.yl[j] * self.cal_kenrel(xi,self.xl[j])
        gi+=self.b
        return gi

    #计算a2new
    def cal_a2new(self,i_a1,i_a2):
        a2new = self.a[i_a2] + self.yl[i_a2] * (self.eCache[i_a1] - self.eCache[i_a2]) / (self.cal_kenrel(self.xl[i_a1],self.xl[i_a1]) + self.cal_kenrel(self.xl[i_a2],self.xl[i_a2]) - 2 * self.cal_kenrel(self.xl[i_a1],self.xl[i_a2]))
        return a2new

    #计算 ks
    def cal_ks(self,i_a1,i_a2):
        ks = 0
        for i in range(len(self.a)):
            if i != i_a1 and i != i_a2:
                ks+=self.yl[i] * self.a[i]
        return ks

    #计算a1new
    def cal_a1new(self,a2new,i_a1,i_a2):
        a1new = 0
        if self.yl[i_a1] == self.yl[i_a2]:
            a1new = self.cal_ks(i_a1,i_a2) - a2new
        else:
            a1new = self.cal_ks(i_a1,i_a2) + a2new
        return a1new

    #计算并更新a1,a2,b
    def cal_b(self,i_a1,i_a2):
        #计算
        a2new = self.cal_a2new(i_a1,i_a2)
        a1new = self.cal_a1new(a2new,i_a1,i_a2)
        b1new = self.yl[i_a1] - self.cal_ks(i_a1,i_a2) - a1new * self.yl[i_a1] * self.cal_kenrel(self.xl[i_a1],self.xl[i_a1]) - a2new * self.yl[i_a2] * self.cal_kenrel(self.xl[i_a2],self.xl[i_a1])
        b2new = self.b - self.eCache[i_a2] - self.yl[i_a1] * self.cal_kenrel(self.xl[i_a1],self.xl[i_a2]) * (a1new - self.a[i_a1]) - self.yl[i_a2] * self.cal_kenrel(self.xl[i_a2],self.xl[i_a2]) * (a2new - self.a[i_a2])
        bnew = (b1new + b2new) / 2
        #更新
        self.a[i_a1] = a1new
        self.a[i_a2] = a2new
        self.b = bnew
        return

    #计算核函数kenrel
    def cal_kenrel(self,x,z):
        result = 0
        #线性核函数
        if self.kernel.upper == 'LINE':
            result = x * z
        #高斯核函数
        elif self.kernel.upper == 'RBF':
            result = np.exp(-self.r * (x - z) ** 2)
        return result

    #计算并更新E
    def cal_E(self):
        for i in range(len(self.eCache)):
            self.eCache[i] = self.cal_gx(i) - self.yl[i]
        return

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
    model.fit()
    x_hat = np.array([1,3,2,4,2,3,2,2]).reshape(4,2)
    y_hat = model.predict(x_hat)
    print(str(y_hat))
