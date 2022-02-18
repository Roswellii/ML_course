# https://blog.csdn.net/qq_42570457/article/details/81454512
from __future__ import division
import math
import random
import pandas as pd

flowerLables = {0: 'Iris-setosa',
                1: 'Iris-versicolor',
                2: 'Iris-virginica'}

random.seed(0)


# 生成区间[a, b)内的随机数
def rand(a, b):
    return (b - a) * random.random() + a


# 生成大小 I*J 的矩阵，默认零矩阵
def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill] * J)
    return m


# 函数 sigmoid
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


# sigmoid 的导数
def dsigmoid(x): # x= f_sigmoid（x）
    return x * (1 - x) # 5.9


class NN:
    """ 三层反向传播神经网络 """

    # 神经网络初始化
    # 配置节点数
    # 初始化权重矩阵
    def __init__(self, ni, nh, no):
        # 输入层、隐藏层、输出层的节点（数）
        self.ni = ni + 1  # 增加一个偏差节点
        self.nh = nh + 1
        self.no = no

        # 激活神经网络的所有节点（向量）
        ## 相当于建立连接？
        self.ai = [1.0] * self.ni
        self.ah = [1.0] * self.nh
        self.ao = [1.0] * self.no

        # 建立隐藏层权重矩阵
        self.wi = makeMatrix(self.ni, self.nh)
        # 建立输出层权重矩阵
        self.wo = makeMatrix(self.nh, self.no)
        # 矩阵值设为随机
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = rand(-0.2, 0.2)
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = rand(-2, 2)

    # 更新input
    def update(self, inputs):
        # 确保输入节点数正确
        if len(inputs) != self.ni - 1:
            raise ValueError('与输入层节点数不符！')

        # 激活输入层
        for i in range(self.ni - 1):
            #
            self.ai[i] = inputs[i]

        # 激活隐藏层
        ## 对于每一个隐藏层节点
        for j in range(self.nh):
            sum = 0.0
            # 将连接的输入层节点加权求和
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[i][j]
            # 将结果signmoid
            self.ah[j] = sigmoid(sum)

        # 激活输出层
        # 对于每一个输出层节点
        for k in range(self.no):
            sum = 0.0
            # 将连接的隐藏层节点加权求和
            for j in range(self.nh):
                sum = sum + self.ah[j] * self.wo[j][k]
            # 将结果signmoid
            self.ao[k] = sigmoid(sum)
        # 返回隐藏层的结果
        return self.ao[:]

    # 网络建立完毕，获取反向传播误差
    def backPropagate(self, targets, lr):
        # 初始化输出层误差
        output_deltas = [0.0] * self.no
        # 对于每一个输出值
        for k in range(self.no):
            # 计算直接偏差
            error = targets[k] - self.ao[k] # 5.10
            # 输出层delta= sigmoid（输出层取值）* error
            output_deltas[k] = dsigmoid(self.ao[k]) * error # 5.10

        # 初始化隐藏层的误差
        hidden_deltas = [0.0] * self.nh
        # 对于每一个隐藏层节点
        for j in range(self.nh):
            error = 0.0
            # 对于每一个连向的输出层节点
            for k in range(self.no):
                # 累计计算误差
                error = error + output_deltas[k] * self.wo[j][k]  # 5.15 & 5.13
            # 隐藏层delta= sigmoid(隐藏层取值)* error
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error # 5.13

        # 更新输出层权重
        # 对于每一个隐藏层节点
        for j in range(self.nh):
            # 连向输出层的每一条边
            for k in range(self.no):
                # 改变量= delta* 上一次的值
                change = output_deltas[k] * self.ah[j] # 5.10 一个公式分成几步处理
                # 新的权重= 上一次的值+ 学习率*改变量
                self.wo[j][k] = self.wo[j][k] + lr * change # 5.11

        # 更新输入层权重
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j] * self.ai[i] # 5.13
                self.wi[i][j] = self.wi[i][j] + lr * change # 5.13
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
            print(p[0], ':', target, '->', flowerLables[index])
            # 正确计数
            count += (target == flowerLables[index])
        # 准确率
        accuracy = float(count / len(patterns))
        print('accuracy: %-.9f' % accuracy)

    # 输出权重
    def weights(self):
        print('输入层权重:')
        for i in range(self.ni):
            print(self.wi[i])
        print()
        print('输出层权重:')
        for j in range(self.nh):
            print(self.wo[j])

    # 训练
    def train(self, patterns, iterations=1000, lr=0.1):
        # lr: 学习速率(learning rate)
        for i in range(iterations):
            error = 0.0
            # patterns?
            # 对于每一个数据？
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


def iris():
    data = []
    # 读取数据
    raw = pd.read_csv('iris.csv')
    raw_data = raw.values
    # 提取样本属性
    raw_feature = raw_data[0:, 0:4]
    # 对于每一个属性
    for i in range(len(raw_feature)):
        # 生成属性列表
        ele = []
        ele.append(list(raw_feature[i]))
        # 分类结果
        if raw_data[i][4] == 'Iris-setosa':
            ele.append([1, 0, 0]) # 分类器的输出只有三类
        elif raw_data[i][4] == 'Iris-versicolor':
            ele.append([0, 1, 0])
        else:
            ele.append([0, 0, 1])
        data.append(ele)
    # 随机排列数据
    random.shuffle(data)
    # 取前100个数据为训练集
    training = data[0:100]
    # 100个后数据为测试集
    test = data[101:]
    # 初始化神经网络
    nn = NN(4, 7, 3)
    # 训练
    nn.train(training, iterations=10000)
    # 测试
    nn.test(test)


if __name__ == '__main__':
    iris()