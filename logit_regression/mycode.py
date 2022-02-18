#周逸群 20191583 CQU
#2021-10
import matplotlib
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

class log_regression(object):
    #建模并计算
    def model(self, x, y):
        # x[N,3] 数据
        # y[N,1] 标签
        #初始化
        beta= np.ones((1, 3)) # beta[1,3]
        #线性组合
        z= np.dot(beta, x.T)
        #损失函数
        loss_old= 0
        loss_new= self.loss(y, z) # 初始状态的损失
        err= 1e-5
        #迭代求解
        while(np.abs(loss_new- loss_old)> err):
            beta= self.update(x, y, beta)
            z= np.dot(beta, x.T) # 更新线性组合
            loss_old= loss_new
            loss_new= self.loss(y,z)
        return beta
    #损失函数
    def loss(self, y,z):
        return np.sum(-y*z+ np.log(1+np.exp(z)))
    #更新beta
    def update(self, x, y, beta):
        x=x.reshape(-1, 3)
        #计算线性组合
        z= np.dot(x,beta.T)
        #预测结果
        p1= np.exp(z)/(1+ np.exp(z))
        #对角阵
        p= np.diag((p1*(1-p1)).reshape(-1))
        #一阶导数
        d1= -np.sum(x*(y-p1), 0, keepdims=True)
        d1= d1.reshape(-1, 1)
        #二阶导数
        d2= x.T.dot(p).dot(x)
        beta -=np.dot(d1.T, np.linalg.inv(d2))
        return beta

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
    xhat = np.hstack((x, np.ones([len(x1), 1])))
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

    return xhat, yshape

if __name__ == '__main__':
    matplotlib.rcParams['font.sans-serif'] = ['KaiTi']

    #读入数据
    xtrain, ytrain= read_data()
    #建模
    M= log_regression()
    beta= M.model(xtrain, ytrain) #beta[1,3]
    #直线起终点
    #w1x+w2y+b=0 y=-(w1x+b)/w2
    density_l= 0.1
    density_r= 0.9
    ymin= -(beta[0][0]*density_l+ beta[0][2])/beta[0][1]
    ymax= -(beta[0][0]*density_r+ beta[0][2])/beta[0][1]
    #plt作图
    plt.plot([density_l, density_r], [ymin, ymax], 'k-')
    plt.xlabel('密度')
    plt.ylabel('含糖率')
    plt.title("对数几率回归")
    plt.show()




