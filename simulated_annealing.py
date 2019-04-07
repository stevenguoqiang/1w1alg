#! -*- coding:utf-8 -*-
from random import random

def simAnnealingMax(lst, howFar):
    '''
    lst:待确定最大值的列表
    howFar:爬山时能看到的“最远方”，越大越准确
    '''
    #由于切片是左闭右开区间，所以howFat必须大于1
    assert howFar>1, 'parameter "howFar" must >1'
    #从列表第一个元素开始爬
    #如果已经到达最后一个元素，或者已找到局部最大值，结束
    start = 0
    ll = len(lst)
    print "ll", ll
    #随机走动的次数
    times = 1
    while start <= ll:
        #当前局部最优解
        m = lst[start]
        #下一个邻域内的数字
        loc = lst[start+1:start+howFar]
        #如果已处理完所有数据，结束
        if not loc:
            return m
        #下一个邻域的局部最优解及其位置
        mm = max(loc)
        mmPos = loc.index(mm)
        #如果下一个邻域内有更优解，走过去
        if m <= mm:
            start += mmPos+1
        else:
            #如果下一个邻域内没有更优解，以一定的概率前进或结束
            delta = float(m-mm)/(m+mm)
            #print "delta", delta
            #随机走动次数越多，对概率要求越低
            #print times
            if delta <= random()/times:
                start += mmPos+1
                times += 1
            else:
                return m


if __name__ == '__main__':
    from random import randint
    lst = [randint(1, 100) for i in range(200)]
    k = 3
    print(simAnnealingMax(lst, k))
