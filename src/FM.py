# -*- coding: utf-8 -*-
# @Time    : 2018/8/15 10:27
# @Author  : Lemon_shark
# @Email   : jiping_cehn@163.com
# @File    : FM.py
# @Software: PyCharm Community Edition

# coding:UTF-8

from __future__ import division
from math import exp
import numpy as np
from numpy import *
from random import normalvariate  # 正态分布
from datetime import datetime
import pandas as pd

trainData = 'data/diabetes_train.txt'   #请换为自己文件的路径
testData = 'data/diabetes_test.txt'

def preprocessData(data):
    print(data.shape)
    print(data[0:5])
    feature=np.array(data.iloc[:,:-1])   #取特征
    print(feature[0:5])
    label=data.iloc[:,-1].map(lambda x: 1 if x==1 else -1) #取标签并转化为 +1，-1
    #将数组按行进行归一化
    zmax, zmin = feature.max(axis=0), feature.min(axis=0)
    print("zmax, zmin ",zmax, zmin )
    feature = (feature - zmin) / (zmax - zmin)
    print(feature[0:5])
    label=np.array(label)

    return feature,label

def sigmoid(inx):
    #return 1. / (1. + exp(-max(min(inx, 15.), -15.)))
    return exp(inx)/(1+exp(inx))
    #return 1.0 / (1 + exp(-inx))

def SGD_FM(dataMatrix, classLabels, k, iter):
    '''
    :param dataMatrix:  特征矩阵
    :param classLabels: 类别矩阵
    :param k:           辅助向量的大小
    :param iter:        迭代次数
    :return:
    '''

    # dataMatrix用的是mat, classLabels是列表
    m, n = shape(dataMatrix)   #矩阵的行列数，即样本数和特征数
    alpha = 0.01
    # 初始化参数
    # w = random.randn(n, 1)#其中n是特征的个数
    w = zeros((n, 1))      #一阶特征的系数
    w_0 = 0.
    #print(ones((n,k)).shape)
    v = normalvariate(0, 0.2) * ones((n, k))   #即生成辅助向量，用来训练二阶交叉特征的系数
    # print(w,m,n,k)
    # print(w_0,v.shape)
    #print(v)

    for it in range(iter):
        count = 0
        for x in range(m):  # 随机优化，每次只使用一个样本
            # 二阶项的计算
            inter_1 = dataMatrix[x] * v  #输入x*隐向量
            # print("shape:",dataMatrix[x].shape,v.shape)
            # print("inter_1",inter_1.shape)
            # print("multiply(inter_1, inter_1",multiply(inter_1, inter_1).shape)
            inter_2 = multiply(dataMatrix[x], dataMatrix[x]) * multiply(v, v)  #二阶交叉项的计算1*8 *8*20 = 1*20
            interaction = sum(multiply(dataMatrix[x] * v , dataMatrix[x] * v ) -
                              multiply(dataMatrix[x], dataMatrix[x]) * multiply(v, v)) / 2.       #二阶交叉项计算完成 1*8 8*20 1*20

            # print("dataMatrix[x]",dataMatrix[x] )
            # print("w",w)
            # print("dataMatrix[x] * w",dataMatrix[x] * w)
            p = w_0 + dataMatrix[x] * w + interaction  # 计算预测的输出，即FM的全部项之和,w0为常数，dataMatirx为1*8，w为8*1
            # 也就是等于sum(wi*xi)
            loss = 1-sigmoid(classLabels[x] * p[0, 0])    #计算损失
            # print(p,loss,classLabels[x],classLabels[x] * p[0, 0],sigmoid(classLabels[x] * p[0, 0]))
            # loss2 = -log(p[0, 0])*classLabels[x]

            # d1=(-1/p[0,0])*classLabels[x]*1
            # d2=(-1/p[0,0])*classLabels[x]*dataMatrix[x]
            # d3=(-1/p[0,0])*classLabels[x]*
            q = (alpha * loss * classLabels[x])/(sigmoid(classLabels[x]*p[0,0]))
            q = (alpha * loss * classLabels[x])
            print(p)
            w_0 = w_0 + q
            #break
            for i in range(n):
                if dataMatrix[x, i] != 0:
                    w[i, 0] = w[i, 0] +q * dataMatrix[x, i]
                    for j in range(k):
                        v[i, j] = v[i, j]+ q* (dataMatrix[x, i] * inter_1[0, j] -v[i, j] * dataMatrix[x, i] * dataMatrix[x, i])
                        # inter_1 = dataMatrix[x] * v  #输入x*隐向量
                        #print("for^:",dataMatrix[x, i], inter_1[0, j])
                        #i 、n 为 隐向量个数，一个隐向量对应k、j个参数，
        print("第{}次迭代后的损失为{}".format(it, loss))

    return w_0, w, v


def getAccuracy(dataMatrix, classLabels, w_0, w, v):
    m, n = shape(dataMatrix)
    allItem = 0
    error = 0
    result = []
    for x in range(m):   #计算每一个样本的误差
        allItem += 1
        inter_1 = dataMatrix[x] * v
        inter_2 = multiply(dataMatrix[x], dataMatrix[x]) * multiply(v, v)
        interaction = sum(multiply(inter_1, inter_1) - inter_2) / 2.
        p = w_0 + dataMatrix[x] * w + interaction  # 计算预测的输出

        pre = sigmoid(p[0, 0])
        result.append(pre)

        if pre < 0.5 and classLabels[x] == 1.0:
            error += 1
        elif pre >= 0.5 and classLabels[x] == -1.0:
            error += 1
        else:
            continue

    return float(error) / allItem


if __name__ == '__main__':
    train=pd.read_csv(trainData)
    test = pd.read_csv(testData)
    dataTrain, labelTrain = preprocessData(train)
    dataTest, labelTest = preprocessData(test)
    date_startTrain = datetime.now()
    print    ("开始训练")
    # print(dataTrain.shape,mat(dataTrain).shape),
    # print(labelTrain.shape,labelTrain)
    #SGD_FM(mat(dataTrain), labelTrain, 20, 200)

    w_0, w, v = SGD_FM(mat(dataTrain), labelTrain, 20, 100)
    print(
        "训练准确性为：%f" % (1 - getAccuracy(mat(dataTrain), labelTrain, w_0, w, v)))
    date_endTrain = datetime.now()
    print(
    "训练用时为：%s" % (date_endTrain - date_startTrain))
    print("开始测试")
    print(
        "测试准确性为：%f" % (1 - getAccuracy(mat(dataTest), labelTest, w_0, w, v)))
