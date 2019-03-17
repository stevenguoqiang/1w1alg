#! -*- coding:utf-8 -*-
import numpy as np
from random import normalvariate  


a = np.array([[1,2],[3,4]])
b = np.array([[1,2],[3,4]])
print np.dot(a,b)
n = 2899
k = 4

g = np.zeros((1,n))
#g[0,0] = None
v = normalvariate(0, 0.2) * np.ones((n, k))   #即生成辅助向量，用来训练二阶交叉特征的系数
print np.dot(g,v)
