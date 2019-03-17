# -*- coding: utf-8 -*-
from __future__ import division
from math import exp
# 正态分布
from random import normalvariate  
from datetime import datetime
import pandas as pd
import numpy as np


trainData = 'train.csv'   
testData = 'test.csv'

def preprocessData(data):
    feature = np.array(data.iloc[:,:-1])   #取特征
    label = data.iloc[:,-1].map(lambda x: 1 if x==1 else -1) #取标签并转化为 +1，-1
    #将数组按行进行归一化,选取每一列的最大和最小资
    zmax, zmin = feature.max(axis=0), feature.min(axis=0)
    print "zmax", zmax
    print "zmin", zmin
    feature = (feature - zmin) / (zmax - zmin)
    # 这个地方在归一化的时候，加入最大和最小一致的话，可能会导致nan的问题
    feature = np.nan_to_num(feature)

    label=np.array(label)
    return feature,label




def sigmoid(inx):
    #ToDo，这个地方需要再仔细debug一下math异常错误
    res = 0
    try:
        res = 1.0 / (1 + exp(-inx))
    except Exception as e:
        print "sigmoid计算错误",inx,e
    return res


def SGD_FM(dataMatrix, classLabels, k, iter_num):
    '''
    :param dataMatrix:  特征矩阵
    :param classLabels: 类别矩阵
    :param k:           辅助向量的大小，默认是4
    :param iter:        迭代次数
    :return:
    '''
    # dataMatrix用的是mat, classLabels是列表
    m, n = np.shape(dataMatrix)   #矩阵的行列数，即样本数和特征数
    print("行数:{0}, 特征数:{1}".format(m, n))
    alpha = 0.01
    # 初始化参数
    # w = random.randn(n, 1)#其中n是特征的个数
    w = np.zeros((n, 1))      #一阶特征的系数
    w_0 = 0.
    v = normalvariate(0, 0.2) * np.ones((n, k))   #即生成辅助向量，用来训练二阶交叉特征的系数

    for it in range(iter_num):
        for x in range(m):  # 随机优化，每次只使用一个样本
            # 二阶项的计算
            inter_1 = dataMatrix[x] * v   #点积
            inter_2 = np.multiply(dataMatrix[x], dataMatrix[x]) * np.multiply(v, v)  #二阶交叉项的计算
            # 要注意这个地方的sum是numpy的sum，这样才能对向量的值求和
            interaction = np.sum(np.multiply(inter_1, inter_1) - inter_2) / 2.       #二阶交叉项计算完成
            p = w_0 + dataMatrix[x] * w + interaction  # 计算预测的输出，即FM的全部项之和
            print "v", v

            value = classLabels[x] * p[0, 0]
            loss = 1-sigmoid(value)    #计算损失
            w_0 = w_0 +alpha * loss * classLabels[x]

            for i in range(n):
                if dataMatrix[x, i] != 0:
                    w[i, 0] = w[i, 0] +alpha * loss * classLabels[x] * dataMatrix[x, i]
                    for j in range(k):
                        v[i, j] = v[i, j]+ alpha * loss * classLabels[x] * (
                        dataMatrix[x, i] * inter_1[0, j] - v[i, j] * dataMatrix[x, i] * dataMatrix[x, i])

        print("第{}次迭代后的损失为{}".format(it, loss))

    return w_0, w, v


def getAccuracy(dataMatrix, classLabels, w_0, w, v):
    m, n = np.shape(dataMatrix)
    allItem = 0
    error = 0
    result = []
    for x in range(m):   #计算每一个样本的误差
        allItem += 1
        inter_1 = dataMatrix[x] * v
        inter_2 = np.multiply(dataMatrix[x], dataMatrix[x]) * np.multiply(v, v)
        interaction = sum(np.multiply(inter_1, inter_1) - inter_2) / 2.
        p = w_0 + dataMatrix[x] * w + interaction  # 计算预测的输出
        pre = sigmoid(p[0, 0])
        result.append(pre)
        print "pre:",pre, "classLabels[x]", classLabels[x]

        if pre < 0.5 and classLabels[x] == 1.0:
            error += 1
        elif pre >= 0.5 and classLabels[x] == -1.0:
            error += 1
        else:
            continue

    return float(error) / allItem


if __name__ == '__main__':
    train=pd.read_csv(trainData,header=0)
    if "is_click" in train.columns:
        print "is_click"
    test = pd.read_csv(testData,header=0)
    dataTrain, labelTrain = preprocessData(train)
    dataTest, labelTest = preprocessData(test)
    date_startTrain = datetime.now()
    print("开始训练")
    w_0, w, v = SGD_FM(np.mat(dataTrain), labelTrain, 4, 2)
    print("训练准确性为：%f" % (1 - getAccuracy(np.mat(dataTrain), labelTrain, w_0, w, v)))
    date_endTrain = datetime.now()
    print("训练用时为：%s" % (date_endTrain - date_startTrain))
    print("开始测试")
    print("测试准确性为：%f" % (1 - getAccuracy(np.mat(dataTest), labelTest, w_0, w, v)))
