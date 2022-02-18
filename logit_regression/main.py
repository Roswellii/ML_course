import math

import pandas as pd
import numpy as np

if __name__ == '__main__':
    #读入alpha数据集
    data = pd.read_table('watermelon_3.0alpha.txt', delimiter=',')
    x1= data['密度'].tolist()
    x2= data['含糖率'].tolist()
    y_in= data['好瓜'].tolist()

    #存入array
    x= np.zeros([len(x1), 2])
    y= np.zeros([len(x1), 1])
    i=0
    while(i<len(x1)):
        x[i][0]= x1[i]
        x[i][1]= x2[i]
        y[i]= y_in[i]
        i+=1

    #计算xhat和beta
    xhat= np.hstack((x, np.ones([len(x1), 1])))
    beta_old= np.ones([3, 1])*np.inf
    beta_new= np.ones([3, 1])
    threshold= 0.001
    while(np.linalg.norm(beta_old-beta_new)>=threshold):
        beta_old= beta_new
        #计算p0, p1
        pre_true= 0
        pre_false= 0


        p0= pre_false/pre_true+ pre_false#good
        p1= 1- p0 #bad
        first_d= np.zeros([1,len(xhat[0])])

        i=0
        while(i<len(x1)):
            first_d += xhat[i]*(y[i] - p1)
            i+=1
        first_d= -first_d

        #计算二阶导数
        p= np.diag((p1*p0).reshape(-1))
        second_d= np.zeros([len(xhat[0]), len(xhat[0])])
        i = 0
        while (i < len(x1)):
            atemp = xhat[i]
            atemp = atemp.reshape(1, -1)
            second_d+= np.matmul(atemp.T, atemp)*p1*p0
            i += 1

        #牛顿迭代法
        beta= np.ones([3,1])
        beta_new= beta_old- np.matmul(np.linalg.inv(second_d),first_d)


