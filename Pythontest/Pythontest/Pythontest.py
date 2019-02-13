import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import preprocessing
import re
from collections import defaultdict
from sklearn.model_selection import train_test_split

#树结构
def tree(): 
    return defaultdict(tree)

#决策树
class DecisionTree:

    #初始化
    def __init__(self,criterion='ID4.5',max_depth=10):
        self.max_depth = max_depth#最大树深
        self.tree = tree()#树的生成结果
        self.criterion = criterion#生成模式 ID3 或 ID4.5


    #拟合函数
    def fit(self,x,y):
        self.tree = self.CreateTree(x,y,1)
        return self

    #预测多个样本
    def predict(self,x):
        return np.array([self.hat(i,self.tree) for i in x])

    #预测某个样本
    def hat(self,x,tree):
        index = tree['index']
        x_hat = x[index]
        key = self.select_close(x_hat,tree["A"])
        if tree["y_hat"].__len__() != 0:
            return float(tree['y_hat'])
        else:
            return self.hat(x,tree['child_' + str(key)])

    #选取最匹配的key值
    def select_close(self,index,dic):
        result = index
        close = 9999999
        for key,value in dic.items():
            i = (key - index) ** 2
            if i <= close:
                close = i
                result = key
        return result

    #递归生成树
    def CreateTree(self,x,y,depth):
        tree1 = tree()
        data = np.hstack((x, y))
        acount_A = len(x[0])
        d = []#熵列表
        #遍历获取熵值列表
        for i in range(acount_A):
            x1 = x[:,i]
            d.append(self.calculateEntropy(x1,y))
        max1 = max(d)#最大熵值
        max_index = d.index(max1)#获得最大熵的索引值
        A = self.calculate_N(x[:,max_index])#最大熵的特征值分类
        #节点符值
        tree1['Entropy'] = max1
        tree1['Sample'] = len(data)
        tree1['index'] = max_index
        tree1['A'] = A
        tree1['depth'] = depth
        #判定熵值是否有效
        if max1 > 0.001 and self.max_depth > depth:
            x1 = list(x[:,max_index])
            for key,value in A.items():
                indexs = self.calculate_indexInList(x1,key)
                data_new = np.array([data[i,:] for i in indexs]) #获取分类后的样本
                x2 = data_new[:,:-1]
                y2 = data_new[:,-1]
                tree1['child_' + str(key)] = self.CreateTree(x2,y2.reshape(len(y2),1),depth + 1)
        else:
            tree1['y_hat'] = y[0]
        return tree1

    #计算信息熵
    def calculateEntropy(self,x,y):
        result = 0.0
        k = self.calculate_N(y)#样本分类个数
        HD = -1 * self.calculate_H(k,len(y))
        HDA = self.calculate_HD(x,y)
        gDA = HD - HDA
        if self.criterion == "ID3":#ID3
            result = gDA
        else:#ID4.5
            a = self.calculate_N(x)#样本分类个数
            HA = -1 * self.calculate_H(a,len(x))
            if HA != 0:
                result = gDA / HA
            else:
                result = gDA
        return result

    #计算列表并分类
    def calculate_N(self,x):
        s = set(x.reshape(1,len(x)).tolist()[0])
        d = {}
        for i in s:
            d[i] = list(x).count(i)
        return d

    #合并相同长度的列表
    def calculate_M(self,x,y):
        l = []
        for i in range(len(x)):
            l.append(str(x[i]) + "," + str(y[i]))
        return np.array(l)

    #计算-H(D)
    def calculate_H(self, k,d):
        result = 0
        for key,value in k.items():
            result = result + (value / d) * np.log(value / d)
        return result

    #计算H(D|A)
    def calculate_HD(self,x,y):
        i = self.calculate_N(x)#特征值分类个数
        l = len(x)
        ik = self.calculate_N(self.calculate_M(x,y))
        result = 0
        for i_key,i_value in i.items():
            dic = self.calculate_Remove(i_key,ik)
            result = result + (i_value / l) * self.calculate_H(dic,i_value)
        return result * -1

    #移除不匹配的字典项
    def calculate_Remove(self,index,k):
        result = {}
        for i_key,i_value in k.items():
            list = i_key.split(',')
            if list[0].find(str(index)) != -1:
               result[i_key] = i_value
        return result

    #计算列表中匹配项的索引值
    def calculate_indexInList(self,x,y):
        l = []
        acount = x.count(y)
        for i in range(acount):
            index = x.index(y)
            add = 0
            if len(l) > 0:
                add = l[i - 1] + 1
            l.append(index + add)
            x = x[index + 1:]
        return l


def iris_type(s):
    it = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    return it[s]

iris_feature = u'花萼长度', u'花萼宽度', u'花瓣长度', u'花瓣宽度'

if __name__ == "__main__":
    ##x = np.array([1,2,2,3,3,3,1,1,1,2,1,3]).reshape(12,1)
    ##y = np.array([2,2,2,3,1,3,1,1,1,2,1,4]).reshape(12,1)
    #de = DecisionTree()
    ##s = set(x.reshape(1,12).tolist()[0])
    ##print("x="+str(x.reshape(1,12)))
    ##print("x="+str(x.reshape(1,12).tolist()[0]))
    ##print("s="+str(s))
    ##print("x_N=" + str(de.calculate_N(x)))
    ##print("y_N=" + str(de.calculate_N(y)))
    ##print("M=" + str(de.calculate_M(x,y)))
    ##print("M_n=" + str(de.calculate_N(de.calculate_M(x,y))))
    ##print("En=" + str(de.calculateEntropy(x,y)))

    ##N = 100
    ##x = np.random.rand(N) * 6 - 3 # [-3,3)
    ##x.sort()
    ##y = np.sin(x) + np.random.randn(N) * 0.05
    ##print("y="+str(y))
    ##x = x.reshape(-1, 1) # 转置后，得到N个样本，每个样本都是1维的
    ##print("x="+str(x))

    ##mpl.rcParams['font.sans-serif'] = [u'SimHei']
    ##mpl.rcParams['axes.unicode_minus'] = False

    #path = u'8.iris.data' # 数据文件路径
    #df = pd.read_csv(path, header=0)
    #x = df.values[:, :-1]
    #y = df.values[:, -1]
    #le = preprocessing.LabelEncoder()
    #le.fit(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
    ## print(str(le.classes_))
    #y = le.transform(y)
    #print("x=" + str(x))
    #print("y=" + str(y.reshape(len(y),1)))
    #de.fit(x,y.reshape(len(y),1))
    ##print("tree="+str(de.tree))

    #y_hat = de.predict(x)
    #print("y_hat=" + str(y_hat))
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    path = u'8.iris.data'  # 数据文件路径
    df = pd.read_csv(path, header=0)
    x = df.values[:, :-1]
    y = df.values[:, -1]
    print('x = \n', x) 
    print('y = \n', y) 
    le = preprocessing.LabelEncoder()
    le.fit(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
    # print(str(le.classes_))
    y = le.transform(y)
    # print('Last Version, y = \n', y)
    # data = np.loadtxt(path,
    # delimiter=',',converters={4:iris_type},dtype=float)
    # print(str(data))
    # x, y = np.split(data, (4,), axis=1)
    # 为了可视化，仅使用前两列特征
    x = x[:, 2:4]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=1)
    #ss = StandardScaler()
    #ss = ss.fit(x_train)

    # 决策树参数估计
    # min_samples_split = 10：如果该结点包含的样本数目大于10，则(有可能)对其分支
    # min_samples_leaf = 10：若将某结点分支后，得到的每个子结点样本数目都大于10，则完成分支；否则，不进行分支
    #model = Pipeline([
    #    ('ss', StandardScaler()),
    #    ('DTC', DecisionTreeClassifier(criterion='entropy', max_depth=6))])
    model = DecisionTree(criterion='ID4.5')
    # clf = DecisionTreeClassifier(criterion='entropy', max_depth=3)
    model = model.fit(x_train, y_train.reshape(len(y_train),1))
    y_test_hat = model.predict(x_test)      # 测试数据

    ## 保存
    ## dot -Tpng -o 1.png 1.dot
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
    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test.ravel(), edgecolors='k', s=100, cmap=cm_dark, marker='o')  # 测试数据
    plt.scatter(x[:, 0], x[:, 1], c=y.ravel(), edgecolors='k', s=40, cmap=cm_dark)  # 全部数据
    plt.xlabel(iris_feature[0], fontsize=15)
    plt.ylabel(iris_feature[1], fontsize=15)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.grid(True)
    plt.title(u'鸢尾花数据的决策树分类', fontsize=17)
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
        clf = DecisionTree(criterion='ID4.5', max_depth=d)
        clf = clf.fit(x_train, y_train.reshape(len(y_train),1))
        y_test_hat = clf.predict(x_test)  # 测试数据
        result = (y_test_hat == y_test)  # True则预测正确，False则预测错误
        err = 1 - np.mean(result)
        err_list.append(err)
        print(d, ' 错误率: %.2f%%' % (100 * err))
    plt.figure(facecolor='w')
    plt.plot(depth, err_list, 'ro-', lw=2)
    plt.xlabel(u'决策树深度', fontsize=15)
    plt.ylabel(u'错误率', fontsize=15)
    plt.title(u'决策树深度与过拟合', fontsize=17)
    plt.grid(True)
    plt.show()