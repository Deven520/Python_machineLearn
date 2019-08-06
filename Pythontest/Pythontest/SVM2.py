import numpy as np

class optStruct:
	#@dataMatIn: 数据集,type: mat
	#@classLabels: 类标签,type: mat
	#@C：自设调节参数
	#@toler: 自设容错大小
	#@kTup: 核函数类型
	def __init__(self, dataMatIn, classLabels, C, toler, kTup):
		self.X = dataMatIn
		self.labelMat = classLabels
		self.C = C
		self.toler = toler
		self.m = shape(dataMatIn)[0]
		self.alphas = mat(zeros((self.m, 1)))   #初始化一个m的列向量α
		self.b = 0
		self.eCache = mat(zeros((self.m, 2)))	#误差(Ei)缓存
		self.K = mat(zeros((self.m, self.m)))   #初始化一个存储核函数值得m*m维的K
		for i in range(self.m):					#获得核函数的值K(X,xi)
			self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)

    #@kTup: lin: 线性 
    #       rbf: 高斯核 公式：exp{(xj - xi)^2 / (2 * δ^2)} | j = 1,...,N
    #       δ：有用户自设给出kTup[1]
def kernelTrans(X, A, kTup):
    m, n = shape(X)
    K = mat(zeros(m, 1))
    if kTup[0] == 'lin': K = X * A.T
    elif kTup[0] == 'rbf': 
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = exp(K / (-1 * kTup[1] ** 2))
    else: 
        raise NameError('Houston we have a promble -- That kernel is not recoginzed')
    return K

#根据公式【4】，计算出Ei
def calcEk(oS, k):
	fXk = float(multiply(oS.alphas, oS.labelMat).T * oS.K[:, k] + oS.b)
	Ek = fXk - float(oS.labelMat[k])
	return Ek

#随机选择第一个α
def selectJrand(i, m):
	j = i
	while (j == i):
		j = int(random.uniform(0, m))
	return j
	
#根据最大步长选择第二个变量	
def selectJ(i, oS, Ei):
    maxK = -1
    maxDeltaE = 0; Ej = 0
    oS.eCache[i] = [1, Ei]        		 #Ei保存的两维中，第一维表示Ei的有效性，第二维才是真实值
    validEcachelist = nonzero(oS.eCache[:, 0].A)[0]
    if (len(validEcachelist)) > 1:
        for k in validEcachelist:
            if k == i: continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):      #选择具有最大步长(max|Ei - Ej|)的j,
                maxK = k; maxDeltaE = deltaE; Ej = Ek
        return maxK, Ej
    else:                                 #随机选择第一个α
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
        return j, Ej

def updatEk(oS, k):
	Ek = calcEk(oS, k)
	oS.eCache[k] = [1, Ek]

#调整大于H或小于L的α值
def clipAlpha(aj, H, L):
	if aj > H:
		aj = H
	if aj < L:
	    aj = L
	return aj
	
#选择第一个变量α
def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i] * Ei < -oS.toler) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i] * Ei > oS.toler) and (oS.alphas[i] > 0)):
        j, Ej = selectJ(i, oS, Ei)      									 #第二个α选择的启发式方法
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy()  	 #进行深复制处理
        #根据公式[6]
        if (oS.labelMat[i] != oS.labelMat[j])	:
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])	
        if L == H: 
            print("L == H")
            return 0
		#根据公式[5]
        eta = 2.0 *  oS.K[i, j] - oS.K[i, i] - oS.K[j, j]
        if eta >= 0: print("eta >= 0"); return 0                             #??为什么eta>=0就可表示剪辑过？
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updatEk(oS, j)														#跟新误差缓存
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            print("j not moving enough"); return 0
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
        updatEk(oS, i)
		#根据公式[3]
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, i] - oS.labelMat[j] * (oS.labelMat[j] - alphaJold) * oS.K[i, j]
        b2 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, j] - oS.labelMat[j] * (oS.labelMat[j] - alphaJold) * oS.K[j, j]
		#根据SMO中优化的部分选择b的公式
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif (o < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
        else: oS.b = (b1 + b2) / 2.0
        return 1
    else: return 0

#@maxIter:程序最大的循环次数
#其他变量的含义已经在数据结构中介绍了，这里就不在赘诉
def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup = ('lin', ):
    oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler, kTup);
    iter = 0;
    entireSet = True ;
    alphaPairsChanged = 0;
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
		alphaPairsChanged = 0
		if entireSet:															#遍历所有值
			for i in range(oS.m):
				alphaPairsChanged += innerL(i, oS)
			print "fullSet, iter: %d i: %d, pairs changed %d" % (iter, i, alphaPairsChanged)
			#iter += 1
		else:																	#遍历非边界值（任意一对α值发生改变，那么就会返回1）
			nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]      #
			for i in nonBoundIs:
				alphaPairsChanged += innerL(i, oS)
				print "non-bound, iter: %d i: %d, pairs changed %d" %(iter, i, alphaPairsChanged)
			#iter += 1
		iter += 1
		if entireSet: entireSet = False
		elif (alphaPairsChanged == 0): entireSet = True
		print "iteration number: %d" % iter
	return oS.b, oS.alphas

#@param: alphas, dataAr, classLabels --> Type: List		
def calWs(alphas, dataArr, classLabels):
	X = mat(dataArr); labelMat = mat(classLabels).transpose()
	m, n = shape(X)
	w = zeros((n, 1))
	for i in range(m):
		w += multiply(alphas[i] * labelMat[i], X[i, :].T)
	return w

def testRbf(k1=1.3):
    dataArr,labelArr = loadDataSet('testSetRBF.txt')
    b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1)) #C=200 important
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    svInd=nonzero(alphas.A>0)[0]
    sVs=datMat[svInd] #get matrix of only support vectors
    labelSV = labelMat[svInd];
    print "there are %d Support Vectors" % shape(sVs)[0]
    m,n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print "the training error rate is: %f" % (float(errorCount)/m)
    dataArr,labelArr = loadDataSet('testSetRBF2.txt')
    errorCount = 0
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    m,n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1    
    print "the test error rate is: %f" % (float(errorCount)/m)  