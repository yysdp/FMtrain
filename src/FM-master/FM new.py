import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import exp
import numpy as np
from numpy import *
from random import normalvariate  # 正态分布
from datetime import datetime
import pandas as pd
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
'''计算logit损失函数：L = log(1 + e**(y_hat * y))'''
def logit(y, y_hat):
    return np.log(1 + np.exp(-y * y_hat))
'''计算logit损失函数的外层偏导(不含y_hat的一阶偏导)'''
def df_logit(y, y_hat):
    return sigmoid(-y * y_hat) * (-y)
'''FM的模型方程：LR线性组合 + 交叉项组合 = 1阶特征组合 + 2阶特征组合'''
datas = pd.read_csv('../xg.csv').values
def sigmoid(inx):
    return exp(inx)/(1+exp(inx))

def preprocessData(datas):
    train = datas[0:600]
    test = datas[600:769]

    train_f = train[:, 0:-1]
    test_f = test[:, 0:-1]

    train_l = train[:, -1]
    test_l = test[:, -1]
    for i, l in enumerate(train_l):
        if (l == 0):
            train_l[i] = -1
        else:
            train_l[i] = 1
    for i,l in enumerate(test_l):
        if(l==0):test_l[i]=-1
        else:test_l[i] = 1



    #将数组按行进行归一化
    zmax= train_f.max(axis=0)
    zmin = train_f.min(axis=0)
    train_f = (train_f - zmin) / (zmax - zmin)

    zmax = test_f.max(axis=0)
    zmin = test_f.min(axis=0)
    test_f = (test_f - zmin) / (zmax - zmin)

    return train_f,train_l,test_f,test_l

def FM(Xi, w0, W, V):
    # 样本Xi的特征分量xi和xj的2阶交叉项组合系数wij = xi和xj对应的隐向量Vi和Vj的内积
    # 向量形式：Wij:= <Vi, Vj> * Xi * Xj
    interaction = np.sum((Xi.dot(V)) ** 2 - (Xi ** 2).dot(V ** 2))  # 二值硬核匹配->向量软匹配
    y_hat = w0 + Xi.dot(W) + interaction / 2  # FM预测函数
    return y_hat[0]
'''SGD更新FM模型的参数列表：[w0, W, V]'''
def FM_SGD(X, y, k=2, alpha=0.01, iter=50):
    m, n = np.shape(X)
    w0, W = 0, np.zeros((n, 1))  # 初始化wo=R、W=(n, 1)
    V = np.random.normal(loc=0, scale=1, size=(n, k))  # 初始化隐向量矩阵V=(n, k)~N(0, 1)，其中Vj是第j维特征的隐向量
    all_FM_params = []  # FM模型的参数列表：[w0, W, V]
    for it in range(iter):
        total_loss = 0  # 当前迭代模型的损失值
        for i in range(m):  # 遍历训练集
            y_hat = FM(Xi=X[i], w0=w0, W=W, V=V)  # FM的模型方程
            total_loss += logit(y=y[i], y_hat=y_hat)  # 计算logit损失函数值
            dloss = df_logit(y=y[i], y_hat=y_hat)  # 计算logit损失函数的外层偏导
            dloss_w0 = dloss * 1  # l(y, y_hat)中y_hat展开w0，求关于w0的内层偏导
            w0 = w0 - alpha * dloss_w0  # 梯度下降更新w0
            for j in range(n):  # 遍历n维向量X[i]
                if X[i, j] != 0:
                    dloss_Wj = dloss * X[i, j]  # l(y, y_hat)中y_hat展开y_hat，求关于W[j]的内层偏导
                    W[j] = W[j] - alpha * dloss_Wj  # 梯度下降更新W[j]
                    for f in range(k):  # 遍历k维隐向量Vj
                        # l(y, y_hat)中y_hat展开V[j, f]，求关于V[j, f]的内层偏导
                        dloss_Vjf = dloss * (X[i, j] * (X[i].dot(V[:, f])) - V[j, f] * X[i, j] ** 2)
                        V[j, f] = V[j, f] - alpha * dloss_Vjf  # 梯度下降更新V[j, f]
        print('FM第{}次迭代，当前损失值为：{:.4f}'.format(it + 1, total_loss / m))
        all_FM_params.append([w0, W, V])  # 保存当前迭代下FM的参数列表:[w0, W, V]
    return all_FM_params
'''FM模型预测测试集分类结果'''

if __name__ == '__main__':
    np.random.seed(123)
    train_f,train_l,test_f,test_l = preprocessData(datas)
    all_FM_params = FM_SGD(X=train_f, y=train_l, k=2, alpha=0.01, iter=45)  # SGD更新FM模型的参数列表：[w0, W, V]
    w0, W, V = all_FM_params[-1]  # FM模型的参数列表

