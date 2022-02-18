from __future__ import division
import math
import random
import pandas as pd
import numpy as np

flowerLables = {0: 'Iris-setosa',
                1: 'Iris-versicolor',
                2: 'Iris-virginica'}

random.seed(0)

def read_data():
    # 打开文件
    data_raw = pd.read_table('iris.data', delimiter=',')
    # 读取数据
    data_array= data_raw.values
    data= []
    for i in range(len(data_raw)):
        data_line=[]
        data_line.append(data_array[i][0:4])
        data_line.append(data_array[i][4])
        data.append((data_line))
    # 转化标记
    for i in range(len(data_raw)):
        if data[i][1] == 'setosa':
            data[i][1] = [1, 0, 0]
        elif data[i][1] == 'versicolor':
            data[i][1] = [0, 1, 0]
        else:
            data[i][1] = [0, 0, 1]
    return data


# 函数 sigmoid
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

# sigmoid 的导数
def dsigmoid(x): # x= f_sigmoid（x）
    return x * (1 - x) # 5.9


class myNeuralNetwork:
    # 神经网络初始化
    # 配置节点数
    # 初始化权重矩阵
    def __init__(self, ni, nh, no):
        # 输入层、隐藏层、输出层的节点（数）
        self.ni = ni + 1  # 增加一个偏差节点
        self.nh = nh + 1
        self.no = no

        # 赋值
        self.ai = [1.0] * self.ni
        self.ah = [1.0] * self.nh
        self.ao = [1.0] * self.no

        # 建立隐藏层权重矩阵
        self.wi = np.zeros((self.ni, self.nh))
        # 建立输出层权重矩阵
        self.wo = np.zeros((self.nh, self.no))
        # 随机生成矩阵权重
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = random.uniform(-0.1, 0.1)
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = random.uniform(-1, 1)

    # 更新input
    def update(self, inputs):
        # 更新输入层
        for i in range(self.ni - 1):
            self.ai[i] = inputs[i]
        # 更新隐藏层
        ## 对于每一个隐藏层节点
        for j in range(self.nh):
            sum = 0.0
            # 输入方向加权求和
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[i][j]
            # 赋值
            self.ah[j] = sigmoid(sum)

        # 更新输出层
        ## 对于每一个输出层节点
        for k in range(self.no):
            sum = 0.0
            # 隐藏方向加权求和
            for j in range(self.nh):
                sum = sum + self.ah[j] * self.wo[j][k]
            # 赋值
            self.ao[k] = sigmoid(sum)
        # 返回输出层
        return self.ao[:]

    # 获取反向传播误差
    def backPropagate(self, targets, lr):
        # 更新输出层权重
        gj=np.zeros((1, self.no))
        # 对于每一个隐藏层节点
        for j in range(self.nh):
            # 连向输出层的每一条边
            for k in range(self.no):
                bh= self.ah[j] # 图5.7
                gj[0][k]= dsigmoid(self.ao[k])*(targets[k] - self.ao[k]) # 5.10
                whj_delta= lr* gj[0][k]* bh # 5.11
                self.wo[j][k] = self.wo[j][k] + whj_delta # 5.11
        # 更新输入层权重
        for i in range(self.ni):
            for j in range(self.nh):
                # 计算whjgj
                sum_wg= 0
                for k in range(self.no):
                    sum_wg+=self.wo[j][k]* gj[0][k] # 5.15
                # 计算vih
                eh= self.ah[j]*(1-self.ah[j])* sum_wg # 5.15
                xi= self.ai[i]
                vih_delta= lr* eh* xi # 5.13
                self.wi[i][j] = self.wi[i][j] + vih_delta # 5.13
        # 计算误差
        error = 0.0
        # 平方误差
        error += 0.5 * (targets[k] - self.ao[k]) ** 2
        return error

    # 测试
    def test(self, patterns):
        count = 0
        # 对于每一个测试样本
        for p in patterns:
            # 真实结果
            target = flowerLables[(p[1].index(1))]
            result = self.update(p[0])
            # 预测结果
            index = result.index(max(result))
            print("---")
            print(target, p[0])
            print('被预测为：', flowerLables[index])
            # 正确计数
            count += (target == flowerLables[index])
        # 准确率
        accuracy = float(count / len(patterns))
        print('accuracy: %-.9f' % accuracy)


    # 训练
    def train(self, patterns, iterations=1000, lr=0.1):
        # lr: 学习速率(learning rate)
        for i in range(iterations):
            error = 0.0
            # 对于每一个数据
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                # 更新神经网络
                self.update(inputs)
                # 网络建立完毕, 获取反向传播误差
                # 累计各个pattern的误差
                error = error + self.backPropagate(targets, lr)
            if i % 100 == 0:
                print('error: %-.9f' % error)

data= read_data()
# 随机排列数据
random.shuffle(data)
# 前100个数据为训练集
training = data[0:100]
# 100个后数据为测试集
test = data[101:]
# 初始化神经网络
myNN = myNeuralNetwork(4, 10, 3)
# 训练
myNN.train(training, iterations=10000)
# 测试
myNN.test(test)
