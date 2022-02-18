#周逸群 20191583 CQU
#机器学习lab1-鸢尾花
#2021-11
import math
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class log_regression(object):
    targetClass= -1
    def __init__(self, targetClass):
        self.targetClass= targetClass

    #建模并计算
    def model(self, x, y):
        # x[N,4] 数据
        # y[N,1] 标签
        # 初始化
        beta= np.ones((1, 5))
        # 以OvR策略处理多分类
        y_model= []
        for i in range(len(y)):
            if y[i]==self.targetClass:
                y_model.append(1)
            else:
                y_model.append(0)
        y_model= np.array(y_model).reshape(-1, 1)

        # 线性组合
        z= np.dot(beta, x.T)
        # 损失函数
        loss_old= 0
        loss_new= self.loss(y_model, z) # 初始状态的损失
        err= 0.1
        # 迭代求解
        while(np.abs(loss_new- loss_old)> err):
            beta= self.update(x, y_model, beta)
            z= np.dot(beta, x.T) # 更新线性组合
            loss_old= loss_new
            loss_new= self.loss(y_model,z)
        return beta
    #损失函数
    def loss(self, y,z):
        return np.sum(-y*z+ np.log(1+np.exp(z)))
    #更新beta
    def update(self, x, y, beta):
        x=x.reshape(-1, 5)
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
    data = pd.read_table('iris.data', delimiter=',')
    x1 = data['sepal length'].tolist()
    x2 = data['sepal width'].tolist()
    x3 = data['petal length'].tolist()
    x4 = data['petal width'].tolist()
    y_in = data['class'].tolist() # 0Setosa,1Versicolour,2Virginica

    # 存入array
    x = np.zeros([len(x1), 4])
    y = np.zeros([len(x1), 1])
    i = 0
    while (i < len(x1)):
        x[i][0] = x1[i]
        x[i][1] = x2[i]
        x[i][2] = x3[i]
        x[i][3] = x4[i]
        y[i] = y_in[i]
        i += 1

    #矩阵拼接生成xhat
    xhat = np.hstack((x, np.ones([len(x1), 1])))
    yshape= y.reshape(-1,1)

    y = np.array(y)
    i = 0
    return xhat, yshape

def predict(scalar, beta, sample):
    temp= []
    for i in range(len(sample)):
        temp.append(sample[i])
    temp.append(1)
    temp= np.array(temp)
    temp/=scalar
    return 1/(1+ math.exp(-beta.dot(temp)))

def read_test_data():
    data = pd.read_table('iris_test.data', delimiter=',')
    x1 = data['sepal length'].tolist()
    x2 = data['sepal width'].tolist()
    x3 = data['petal length'].tolist()
    x4 = data['petal width'].tolist()
    y_in = data['class'].tolist()  # 0Setosa,1Versicolour,2Virginica

    # 存入array
    x = np.zeros([len(x1), 4])
    y = np.zeros([len(x1), 1])
    i = 0
    while (i < len(x1)):
        x[i][0] = x1[i]
        x[i][1] = x2[i]
        x[i][2] = x3[i]
        x[i][3] = x4[i]
        y[i] = y_in[i]
        i += 1
    return x, y


if __name__ == '__main__':
    # 中文输出
    matplotlib.rcParams['font.sans-serif'] = ['KaiTi']
    # 读入数据
    xtrain, ytrain= read_data()
    scalar= np.sum(xtrain)
    xtrain/=scalar
    # 训练得到三个分类器
    M0 = log_regression(0)
    beta0 = M0.model(xtrain, ytrain)
    M1 = log_regression(1)
    beta1 = M1.model(xtrain, ytrain)
    M2 = log_regression(2)
    beta2 = M2.model(xtrain, ytrain)
    # 读入测试集
    test_data_x, test_data_y= read_test_data()
    # 保存预测结果
    predict_result = []
    for i in range(len(test_data_x)):
        a_flower_x = test_data_x[i].tolist()
        print("第{}个测试样本".format(i))
        p0= predict(scalar, beta0,a_flower_x)
        p1 = predict(scalar, beta1, a_flower_x)
        p2 = predict(scalar, beta2, a_flower_x)
        print("p0={}, p1={}, p2={}".format(p0,p1,p2))
        # print(p2)
        if p0>p1 and p0>p2:
            predict_result.append(0)
            print("It is 0 Setosa")
        if p1>p2 and p1>p0:
            predict_result.append(1)
            print("It is 1 Versicolour")
        if p2>p1 and p2>p0:
            predict_result.append(2)
            print("It is 2 Virginica")
        print("-------")
    # 计算准确率
    accu_count= 0
    for i in range(len(predict_result)):
        if predict_result[i]== test_data_y[i]:
            accu_count+=1
    print("预测标签为："+ str(predict_result))
    print("预测准确率为:" + str(accu_count/len(predict_result)))
    plt.scatter(range(0, len(test_data_y), 1), test_data_y, marker='o')
    plt.scatter(range(0,len(test_data_y),1), predict_result, marker='*')
    plt.show()





