from math import exp
import numpy as np
from numpy import *
from random import normalvariate  # 正态分布
from datetime import datetime
import pandas as pd

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

train_f,train_l,test_f,test_l = preprocessData(datas)
# print(train_f.shape,train_f[0:5])
#train_l,test_f,test_l)
def SGD_FM(train_f, train_l, k=20, iter=100):
    m, n = shape(train_f)  # 矩阵的行列数，即样本数和特征数
    alpha = 0.01
    # 初始化参数
    # w = random.randn(n, 1)#其中n是特征的个数
    w = mat(normalvariate(0, 0.2) * ones((n, 1)))  # 一阶特征的系数
    w_0 = 1.
    v = mat(normalvariate(0, 0.2) * ones((n, k)))  # 即生成辅助向量，用来训练二阶交叉特征的系数
    # print(train_f.shape[0])
    # print(w_0, w.shape, v.shape)

    for it in range(iter):
        count=0
        loss = 0
        for i in range(m):
            i = random.randint(0, m)

            q = sum(multiply(train_f[i] * v, train_f[i] * v) - multiply(train_f[i], train_f[i]) * multiply(v, v)) / 2

            y = w_0 + train_f[i] * w + q
            # y = sigmoid(y[0, 0])

            #print("Y:",y ,"\ntrain[i]",train_l[0,i])
            if (y >= 0 and train_l[0,i] == 1):
                count += 1
            if (y < 0 and train_l[0,i] == -1):
                count += 1

            loss = 1 - sigmoid(train_l[0,i]* y[0,0])  # 计算损失
            #loss +=(train_l[0,i]- y[0,0])**2

            #print("??",train_l[0,i],loss.shape,train_l[0,i])
            d = alpha * loss * train_l[0,i]
           # print(train_l[0,i], y[0,0])
            #d = - 2*(y[0,0]- train_l[0, i]) * alpha
            w_0 += d
            #print(d.shape)
            for j in range(n):
                # print("d",d,"\ntrain:",train_f[i][j],"\nw[j]",w[j])
                w[j] += d * train_f[i, j]
                for l in range(k):
                    # print(v[j,l],)
                    # print(train_f[i,j])
                    # print(train_f.shape)
                    # print(v.shape)
                    # print(train_f[i,]*v)
                    # print(train_f[i,:])
                    # print(v[:,l])
                    v[j, l] += d * train_f[i, j] * (train_f[i, :] * v[:, l] - train_f[i, j] * v[j, l])

        print(it, "轮loss:", loss,count,m,d)
        print("精度", count / m)

    return (w,w_0,v )


w,w_0,v  = SGD_FM(mat(train_f),mat(train_l))
def pridict(w,w_0,v,test_f,test_l):
    m,n = (test_f.shape)
    count=0
    for i in range(m):

        q = sum(multiply(test_f[i] * v, test_f[i] * v) - multiply(test_f[i], test_f[i]) * multiply(v, v)) / 2

        y = w_0 + test_f[i] * w + q
        # y = sigmoid(y[0, 0])

        # print("Y:",y ,"\ntrain[i]",test_l[0,i])
        if (y >= 0 and test_l[0, i] == 1):
            count += 1
        if (y < 0 and test_l[0, i] == -1):
            count += 1
    print("测试集精度{:.2%}".format(count/m))


pridict(w, w_0,v, mat(test_f),mat(test_l))