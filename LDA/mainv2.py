import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import matplotlib

def read_data():
    # 读入alpha数据集
    data = pd.read_table('watermelon_3.0alpha.txt', delimiter=',')
    x1 = data['密度'].tolist()
    x2 = data['含糖率'].tolist()
    y_in = data['好瓜'].tolist()

    # 存入array
    x = np.zeros([len(x1), 2])
    y = np.zeros([len(x1), 1])
    i = 0
    while (i < len(x1)):
        x[i][0] = x1[i]
        x[i][1] = x2[i]
        y[i] = y_in[i]
        i += 1

    #矩阵拼接生成xhat
    #xhat[N, 3]
    # xhat = np.hstack((x, np.ones([len(x1), 1])))
    #y[N,1]
    yshape= y.reshape(-1,1)

    y = np.array(y)
    plt.scatter(x1, x2)

    i = 0
    while (i < len(x1)):
        if(y[i]==1):
            plt.scatter(x1[i], x2[i], c='y')
        else:
            plt.scatter(x1[i], x2[i], c='b')
        i += 1

    return x, yshape

def lda(c1, c2):
    # 均值坐标
    c1mean= np.mean(c1, axis= 0)
    c2mean= np.mean(c2, axis= 0)
    allmean= np.mean(np.vstack((c1, c2)))

    # sw 类内散度矩阵
    c1n= c1.shape[0]
    c2n= c2.shape[0]
    s1= 0
    for i in range(0, c1n):
        s1+= (c1[i,:]- c1mean).T* (c1[i,:]-c1mean)
    s2 = 0
    for i in range(0, c1n):
        s2 += (c2[i, :] - c2mean).T * (c2[i, :] - c2mean)
    sw= (c1n*s1+ c2n*s2)/(c1n+ c2n) #需要用权重来调节吗？

    # sb 类间散度矩阵
    sb= np.sum((c1mean- c2mean).dot((c1mean-c2mean).T))

    # 求最大特征值
    eigvalue, eigvector= np.linalg.eig(np.mat(sw).I*sb) # 求特征值，特征向量
    indexVec= np.argsort(-eigvalue) # 特征值从大到小排序
    w= eigvector[:,indexVec[:1] ] # 最大特征值对应的特征向量
    return w


if __name__ =='__main__':
    matplotlib.rcParams['font.sans-serif'] = ['KaiTi']
    # 读入数据
    x, y= read_data()
    # 好坏分类
    xgood= []
    xbad= []
    i = 0
    while (i < len(x)):
        if (y[i] == 1):
           xgood.append(x[i])
        else:
            xbad.append(x[i])
        i += 1
    # LDA求权值
    w= lda(np.mat(xgood), np.mat(xbad))

    # 绘图
    # 直线起终点
    # w1x+w2y+b=0 y=-(w1x+b)/w2
    density_l = 0.1
    density_r = 0.9
    ymin = -(np.sum(w[0]) * density_l) / np.sum(w[1])
    ymax = -(np.sum(w[0]) * density_r) / np.sum(w[1])
    # plt作图
    plt.plot([density_l, density_r], [ymin, ymax], 'k-')
    plt.xlabel('Density')
    plt.ylabel('Sugar')
    plt.title("LDA")
    plt.show()




