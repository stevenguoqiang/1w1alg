#! -*- coding:utf-8 -*-

import networkx as nx

def Dijkstra(G, start, end):
    # 将图进行翻转，从终点出发来找
    # 每一步更新到终点距离最短的一个点
    RG = G.reverse()
    dist = {}                                 # 保存每个点到终点的距离
    previous = {}                             # 路径，保存每个节点的上一个点

    for v in RG.nodes():
        dist[v] = float('inf')
        previous[v] = 'none'
    dist[end] = 0                             # 终点取值0
    u = end                                   # 开始节点u
    while u!=start:                           # 一直找到起点为止
        u = min(dist, key=dist.get)           #寻找dist里面最小值
        distu = dist[u]                       #保存dist中u的值
        del dist[u]                           # 删掉dist中的key u
        for u, v in RG.edges(u):              # 在u相连接的点里面，更新v的值
            if v in dist:
                alt = distu + RG[u][v]['weight']
                if alt < dist[v]:
                    dist[v] = alt
                    previous[v] = u
    path = (start,)
    last = start
    while last != end:
        nxt = previous[last]
        path += (nxt,)
        last = nxt
    return path

if __name__ == '__main__':
    G=nx.DiGraph()
    G.add_edge(0,1,weight=6)
    G.add_edge(0,2,weight=3)
    G.add_edge(1,2,weight=2)
    G.add_edge(1,3,weight=5)
    G.add_edge(2,3,weight=3)
    G.add_edge(2,4,weight=4)
    G.add_edge(3,4,weight=2)
    G.add_edge(3,5,weight=3)
    G.add_edge(4,5,weight=5)

    rs=Dijkstra(G,0,5)
    print(rs)
