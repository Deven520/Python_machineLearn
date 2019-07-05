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
    def __init__(self,traindata=[], kernel='LINE',maxIter=1000,C=100,epsilon=0.00001,r=1):
        self.kernel = kernel #核函数
        self.traindata = traindata #训练数据
        self.maxIter = maxIter #最大项
        self.C = C #惩罚因子
        self.epsilon = epsilon #误差**2精度
        self.w = np.array([0.0 for i in range(len(traindata[0]) - 1)]) #w参数
        self.a = np.array([0.0 for i in range(len(traindata))]) #a参数
        self.b = 0  #b参数
        self.xl = traindata[:,:-1] #训练数据x
        self.yl = traindata[:,-1:] #训练数据结果y
        self.eCache = np.array([-self.yl[i] for i in range(len(traindata))]) # e 的缓存
        self.r = r #高斯核函数超参数r
        self.i_pre_a1 = 0 #上次迭代选择的a1索引
        self.i_pre_a2 = 0 #上次迭代选择的a2索引
    # 拟合函数
    def fit(self):
        while self.isover() == 1:
            self.SMO()
        return self

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
        first_sel = 0
        i_a2_sel = 0
        isover = 0
        rebuilt = 0
        while (i_a1_sel == i_a2_sel and isover == 0) or rebuilt == 1:
            max_error1 = 0
            max_error2 = 0
            max_error3 = 0
            max_error4 = 0
            #遍历选择符合kkt条件的a1
            for i in range(len(self.a)):
                xi = self.xl[i]
                yi = self.yl[i]
                ai = self.a[i]
                if i == self.i_pre_a1 and rebuilt == 1:
                    error = 1 #排除重复数
                else:
                    error = self.cal_gx(i) * yi #误差
                #优先选择违反0<ai<C => yi*gi=1
                if ai > 0 and ai < self.C and error != 1 and (1 - error) ** 2 > max_error1 and (i != self.i_pre_a1 or i_a2_sel != self.i_pre_a2):
                    max_error1 = (1 - error) ** 2
                    i_a1_sel = i
                    first_sel = 1
                elif ai == 0 and error < 1 and (1 - error) ** 2 > max_error2 and first_sel == 0 and (i != self.i_pre_a1 or i_a2_sel != self.i_pre_a2):
                    max_error2 = (1 - error) ** 2
                    i_a1_sel = i
                    first_sel = 2
                elif ai == self.C and error > 1 and (1 - error) ** 2 > max_error2 and first_sel == 0 and (i != self.i_pre_a1 or i_a2_sel != self.i_pre_a2):
                    max_error2 = (1 - error) ** 2
                    i_a1_sel = i
                    first_sel = 2
                elif first_sel == 0 and (ai < 0 or ai > self.C) and (i != self.i_pre_a1 or i_a2_sel != self.i_pre_a2):
                    i_a1_sel = i
            #遍历选择a2
            e1 = self.eCache[i_a1_sel]
            for i in range(len(self.a)):
                error = (e1 - self.eCache[i]) ** 2
                if error > max_error4 :
                    i_a2_sel = i
                    max_error4 = error
            if i_a1_sel == self.i_pre_a1 and i_a2_sel == self.i_pre_a2:
                rebuilt += 1
            else:
                rebuilt = 0
                self.i_pre_a1 = i_a1_sel
                self.i_pre_a2 = i_a2_sel
            self.cal_b(i_a1_sel,i_a2_sel)
            self.cal_E()
        return self

    #判断是否满足终止条件
    def isover(self):
        done = 0 #默认为是
        for i in range(len(self.a)):
            x = self.xl[i]
            y = self.yl[i]
            a = self.a[i]
            gi = self.cal_gx(i)
            error = (y * gi - 1) ** 2
            if a > 0 and a < self.C and error > self.epsilon ** 2:
                done = 1
                break
            elif a == 0 and gi * y < 1:
                done = 1
                break
            elif a == self.C and gi * y > 1:
                done = 1
                break
            elif a < 0 or a > self.C:
                done = 1
                break
        return done


    #计算a2new
    def cal_a2new(self,i_a1,i_a2):
        y1 = self.yl[i_a1]
        y2 = self.yl[i_a2]
        a1 = self.a[i_a1]
        a2 = self.a[i_a2]
        k = self.cal_kenrel(self.xl[i_a1],self.xl[i_a1]) + self.cal_kenrel(self.xl[i_a2],self.xl[i_a2]) - 2 * self.cal_kenrel(self.xl[i_a1],self.xl[i_a2])
        e = self.eCache[i_a1] - self.eCache[i_a2] 
        a2new = a2 + y2 * e / k
        L = 0 #下限
        H = self.C #上限
        if y1 != y2:
            a_min_gap = a2 - a1
            a_max_gap = self.C + a_min_gap
        else:
            a_min_gap = a2 + a1 - self.C
            a_max_gap = a2 + a1
        if a_min_gap >= L:
            L = a_min_gap
        if a_max_gap <= H:
            H = a_max_gap
        #根据上下限赋值
        if a2new <= L:
            a2new = L
        elif a2new >= H:
            a2new = H
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
        y1 = self.yl[i_a1]
        y2 = self.yl[i_a2]
        a1 = self.a[i_a1]
        a2 = self.a[i_a2]
        #ks = self.cal_ks(i_a1,i_a2)
        a1new = a1 + y1 * y2 * (a2 - a2new)
        L = 0 #下限
        H = self.C #上限
        #根据上下限赋值
        if a1new <= L:
            a2new += y1 * y2 * a1new
            a1new = L
        elif a1new >= H:
            t = a1new - self.C
            a2+=y1 * y2 * t
            a1new = H
        return a1new,a2new

    #计算并更新a1,a2,b
    def cal_b(self,i_a1,i_a2):
        #计算
        a2new = self.cal_a2new(i_a1,i_a2)
        a1new,a2new = self.cal_a1new(a2new,i_a1,i_a2)
        b1new = self.yl[i_a1] - self.cal_ks(i_a1,i_a2) - a1new * self.yl[i_a1] * self.cal_kenrel(self.xl[i_a1],self.xl[i_a1]) - a2new * self.yl[i_a2] * self.cal_kenrel(self.xl[i_a2],self.xl[i_a1])
        b2new = self.b - self.eCache[i_a2] - self.yl[i_a1] * self.cal_kenrel(self.xl[i_a1],self.xl[i_a2]) * (a1new - self.a[i_a1]) - self.yl[i_a2] * self.cal_kenrel(self.xl[i_a2],self.xl[i_a2]) * (a2new - self.a[i_a2])
        if a2new > 0 and a2new < self.C:
            bnew = b2new
        elif a1new > 0 and a1new < self.C:
            bnew = b1new
        else:
            bnew = (b1new + b2new) / 2
        print("a1new=" + str(a1new))
        print("a2new=" + str(a2new))
        print("a=" + str(self.a))
        print("e=" + str(self.eCache))
        #更新
        c = self.a[i_a1]
        self.a[i_a1] = a1new
        d = self.a[i_a1]
        self.a[i_a2] = a2new
        self.b = bnew
        return self

    #计算核函数kenrel
    def cal_kenrel(self,x,z):
        result = 0
        #线性核函数
        if self.kernel.upper() == 'LINE':
            #result = np.dot(x , z)
            result=x[0]*z[0]+x[1]*z[1]
        #高斯核函数
        elif self.kernel.upper() == 'RBF':
            d = np.dot((x - z),(x - z))
            result = np.exp(-self.r * d)
        return result

    #计算g(xi)
    def cal_gx(self,i):
        gi = 0.0
        xi = self.xl[i]
        for j in range(len(self.a)):
            gi+=self.a[j] * self.yl[j] * self.cal_kenrel(xi,self.xl[j])
        gi+=self.b
        return gi

    #计算并更新E
    def cal_E(self):
        for i in range(len(self.eCache)):
            self.eCache[i] = self.cal_gx(i) - self.yl[i]
        return self

def iris_type(s):
    it = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    return it[s]


iris_feature = u'花萼长度', u'花萼宽度', u'花瓣长度', u'花瓣宽度'

if __name__ == "__main__":
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    #x = np.array([1,1,2,3,2,2,2,1])
    #y = np.array([1,1,-1,-1])
    #traindata = np.hstack((x.reshape(4,2), y.reshape(4,1)))
    #print(str(traindata))
    #model = SVM(traindata=traindata)
    #print(str(model.w))
    #print(str(model.a))
    #print(str(model.xl))
    #print(str(model.yl))
    #model.fit()
    #x_hat = np.array([1,3,2,4,2,3,2,2]).reshape(4,2)
    #y_hat = model.predict(x_hat)
    #print(str(y_hat))


    ##############################
    path = u'8.iris.data'  # 数据文件路径
    df = pd.read_csv(path, header=0)
    df = df.values[0:99,:]
    x = df[:, :-1]
    y = df[:, -1]
    le = preprocessing.LabelEncoder()
    le.fit(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
    y = le.transform(y)
    # 为了可视化，仅使用前两列特征
    x = x[:,:2]
    y1 = np.array([])
    for i in y:
        if i == 0:
            y1 = np.append(y1,-1)
        else:
             y1 = np.append(y1,i)
    y = y1
    print('x = \n', x)
    print('y = \n', y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

    # 决策树参数估计
    traindata = np.hstack((x, y.reshape(len(x),1)))
    print('traindata = \n', traindata)
    model = SVM(traindata=traindata)
    model.fit()
    print(str(model.a))
    y_test_hat = []# model.predict(x_test) # 测试数据


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
    #x3 = np.ones(x1.size) * np.average(x[:, 2])
    #x4 = np.ones(x1.size) * np.average(x[:, 3])
    #x_test = np.stack((x1.flat, x2.flat, x3, x4), axis=1) # 测试点

    cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])    
    y_show_hat = model.predict(x_show)#np.array([0 for i in range(len(x_show))]) #model.predict(x_show) # 预测值
    print("xshow=" + str(x_show))
    print("yshow=" + str(y_show_hat))
    print("a=" + str(model.a))
    y_show_hat = y_show_hat.reshape(x1.shape) # 使之与输入的形状相同
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
    plt.title(u'鸢尾花数据的SVM分类', fontsize=17)
    plt.show()
