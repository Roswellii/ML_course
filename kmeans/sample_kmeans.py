# -*- coding: utf-8 -*-
#author: w61

import random
import math
import matplotlib.pyplot as plt

class Kmeans():
    def __init__(self,k,mode):
        self.datasets = []
        self.vector = 0                             #记录数据集的维数
        self.k = k                                      #聚类簇数
        self.C = [[] for i in range(k)]                  #簇
        self.mode = 0                               #初始化方式
        self.mean_vector = []                             #均值向量
        self.update_flag = 0                        #记录均值向量是否改变
        self.count = 0                              #记录循环次数
        self.max_count = 500                        #最大运行轮数

    def LoadData(self,file_name):
        """
        读取数据集
        :param file_name:数据集的路径
        :return:null
        """
        with open(file_name) as f:
            for contents in f:
                contents = contents.replace('\n','')
                content = contents.split(' ')
                self.datasets.append(content[1:])
        self.vector = len(self.datasets[0])

    def Inialize(self):
        """
        初始化均值向量
        :return:null
        """
        if self.mode == 0:
            means = random.sample(range(0,len(self.datasets)),self.k)
            for mean in means:
                self.mean_vector.append(self.datasets[mean])

    def GetDistance(self,x,y):
        """
        计算两个向量(x和y)之间的距离
        :param x:
        :param y:
        :return: dis 距离
        """
        dis = 0
        for i in range(self.vector):
            dis += pow((float(x[i])-float(y[i])),2)
        dis = math.sqrt(dis)
        return dis

    def Clustering(self):
        """
        划分簇
        :return:null
        """
        self.C = [[] for m in range(self.k)]   #每一次都要先把C置零
        for j in range(0,len(self.datasets)):
            min_dis = 0
            min_index = 0
            for i in range(self.k):
                dis = self.GetDistance(self.datasets[j],self.mean_vector[i])
                if min_dis == 0 or dis < min_dis:
                    min_dis = dis
                    min_index = i
            self.C[min_index].append(j)

    def Update(self):
        """
        更新每个簇的均值向量
        :return:
        """
        self.update_flag = 0
        for i in range(self.k):
            means= []
            for h in range(self.vector):
                sum = 0
                for j in self.C[i]:
                    sum += float(self.datasets[j][h])
                if len(self.C[i]) == 0:
                    mean = 0
                else:
                    mean = sum / len(self.C[i])
                means.append(mean)
            if means == self.mean_vector[i]:
                continue
            else:
                self.mean_vector[i] = list(means)
                self.update_flag = 1

    def Plot(self):
        """
        绘图
        :return:
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        colors = ['red','blue','green','black','pink','brown','gray']

        for i in range(self.k):
            for j in self.C[i]:
                x = float(self.datasets[j][0])
                y = float(self.datasets[j][1])
                plt.scatter(x,y,color = colors[i],marker='+')

        for i in range(self.k):
            for j in self.mean_vector:
                x = float(j[0])
                y = float(j[1])
                plt.scatter(x,y,color = 'cyan',marker='p')
        plt.show()

    def Process(self):
        """
        执行函数
        :return:
        """
        self.LoadData('watermelon.data')
        self.Inialize()
        while self.count <= self.max_count:
            self.Clustering()
            self.Update()
            if self.update_flag == 0:
                break
            self.count += 1
        print('k-means分类结果如下：')
        print(self.C)
        self.Plot()

if __name__ == "__main__":
    kmeans = Kmeans(5,0)
    kmeans.Process()
