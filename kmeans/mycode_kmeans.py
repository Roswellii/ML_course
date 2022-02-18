# 周逸群 20191583 ML9 KMEANS

import math
from matplotlib import pyplot as plt
import random


# KMEANS类
class KMEANS:
    # 初始化参数
    def __init__(self, k,max_iter):
        self.data= [] # 数据集
        self.vector = 0 # 数据的维度
        self.k= k # 簇数
        self.c= [[] for i in range(k)] # 簇
        self.mean_vec= [] # 均值向量
        self.mvec_flag_updated =False # 均值向量的改变
        self.max_iter_time= max_iter # 循环次数上限

    def load(self, filename):
        data= []
        # 怎么打开文件？
        with open(filename) as file:
            for items in file:
                # 按行切割
                items= items.replace('\n', '')
                item= items.split(' ')
                data.append(item[1:])
        self.vector= len(data[0])
        return data

    def GetVecDist(self,x,y):
        dis = 0
        for i in range(self.vector):
            dis += pow((float(x[i])-float(y[i])),2)
        dis = math.sqrt(dis)
        return dis

    def kmeans_process(self):
        # 如何产生初始均值向量？
        mean_vecs= random.sample(range(0, len(self.data)), self.k) # 随机生成k个下标
        for mean_vec in mean_vecs:
            self.mean_vec.append(self.data[mean_vec])
        # 产生完初始均值向量后，该做什么？大循环
        for i in range(self.max_iter_time):
            # 聚类
            self.c= [[] for j in range(self.k)] # 清空聚类
            for k in range(len(self.data)): # 对每一个样本聚类
                min_dis= -1
                min_index= -1
                for l in range(self.k): # 计算到均值向量的距离
                    d= self.GetVecDist(self.data[k], self.mean_vec[l])
                    if min_dis== -1 or d< min_dis:
                        min_dis= d
                        min_index= l
                self.c[min_index].append(k) # 划入相应的簇
            # 更新均值向量
            self.mvec_flag_updated= False
            for i in range(self.k): # 对于每一个簇
                means= [] # ?
                for h in range(self.vector): # 对于每一个维度
                    sum = 0
                    for j in self.c[i]: # 对于某一个簇
                        sum+= float(self.data[j][h])
                    if len(self.c[i])== 0:
                        mean= 0
                    else:
                        mean= sum/ len(self.c[i])
                    means.append(mean)
                if means == self.mean_vec[i]:
                    continue
                else:
                    self.mean_vec[i]= list(means)
                    self.mvec_flag_updated= True
            if self.mvec_flag_updated== False:
                break
        print(self.c)
        # 怎么可视化？
        fig = plt.figure()

        colors = ['red', 'blue', 'green', 'black', 'pink', 'brown', 'gray','black','purple','orange']
        print("colors:",colors)
        for i in range(self.k): # 对于每一个簇
            for item_index in self.c[i]: # 对于簇内的每个样本
                x= float(self.data[item_index][0])
                y= float(self.data[item_index][1])
                plt.scatter(x, y, color=colors[i], marker='p')
        plt.show()

    def start(self):
        # 载入数据
        self.data= self.load('watermelon.data')
        # 下一步干什么？
        self.kmeans_process()

# 从这儿开始
kms= KMEANS(5, 10000)
kms.start()