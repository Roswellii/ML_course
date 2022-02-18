import numpy as np
import re

# 将数据集根据划分特征切分为两类
def split_dataset(data_x, data_y, fea_axis, fea_value):

    if isinstance(fea_value, float):
        # 如果特征值为浮点数(连续特征值)，那么进行连续值离散化
        equal_Idx = np.where(data_x[:, fea_axis] >= fea_value)
        nequal_Idx = np.where(data_x[:, fea_axis] < fea_value)
    else:
        equal_Idx = np.where(data_x[:, fea_axis] == fea_value)
        nequal_Idx = np.where(data_x[:, fea_axis] != fea_value)
    return data_x[equal_Idx], data_y[equal_Idx], data_x[nequal_Idx], data_y[nequal_Idx]


# 求解基尼指数
def cal_gini(data_y):
    m = len(data_y)  # 全部样本的标签数量
    labels = np.unique(data_y)  # 获得不同种类的标签（去重）
    gini = 1.0  # 最后返回的基尼指数
    for label in labels:
        ans = data_y[np.where(data_y[:] == label)].size / m  # 该标签的出现概率
        gini -= ans * ans  # 累减计算基尼指数（两两不同的总概率）
    return gini

# 分类方法实现最优特征的选取以及特征值划分
def classify_get_best_fea(data_x, data_y):
    m, n = np.shape(data_x)  # m，n分别为样本数以及特征属性数
    # 初始化
    best_fea = -1
    best_fea_val = -1
    min_fea_gini = np.inf

    for i in range(n):  # 遍历所有特征（列）
        feas = np.unique(data_x[:, i])  # 获得该特征下所有特征值
        # 分别以每个特征值为中心进行划分求基尼系数，找到使基尼系数最小的划分
        for j in feas:
            equal_data_x, equal_data_y, nequal_data_x, nequal_data_y = split_dataset(data_x, data_y, i, j)
            fea_gini = 0.0

            fea_gini = len(equal_data_y) / m * cal_gini(equal_data_y) + len(nequal_data_y) / m * cal_gini(nequal_data_y)
            # 如果该划分方式的基尼系数更小（纯度更高），那么直接进行更新
            if fea_gini < min_fea_gini:
                min_fea_gini = fea_gini
                best_fea = i
                best_fea_val = j

    return best_fea, best_fea_val

# 创建分类方法的决策树
def create_CART_classify(data_x, data_y, fea_label):
    labels = np.unique(data_y)
    # 只有一个标签的情况
    if len(labels) == 1:
        return data_y[0]
    # 特征集为0的情况，采用多数投票的方法
    if data_x.shape[1] == 0:
        best_fea, best_fea_num = 0, 0
        for label in labels:
            num = data_y[np.where(data_y == label)].size
            if num > best_fea_num:
                best_fea = label
                best_fea_num = num
        return best_fea

    best_fea, best_fea_val = classify_get_best_fea(data_x, data_y)
    best_fea_label = fea_label[best_fea]
    cartTree = {best_fea_label: {}}

    # 获得划分结果
    equal_data_x, equal_data_y, nequal_data_x, nequal_data_y = split_dataset(data_x, data_y, best_fea, best_fea_val)
    # 删除最优特征
    equal_data_x = np.delete(equal_data_x, best_fea, 1)
    nequal_data_x = np.delete(nequal_data_x, best_fea, 1)

    fea_label = np.delete(fea_label, best_fea, 0)
    # 递归生成CART分类树
    cartTree[best_fea_label]["{}_{}".format(1, best_fea_val)] = create_CART_classify(equal_data_x, equal_data_y,
                                                                                     fea_label)
    cartTree[best_fea_label]["{}_{}".format(0, best_fea_val)] = create_CART_classify(nequal_data_x, nequal_data_y,
                                                                                     fea_label)
    return cartTree


# 预测一条测试样本
def classify(inputTree, xlabel, testdata):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = xlabel.index(firstStr)  # 根据key值得到索引
    classLabel = '0'  # 定义变量classLabel，默认值为0

    ans = re.findall(r'\d+\.\d+', list(secondDict.keys())[0])
    if isinstance(testdata[featIndex], float):
        if float(testdata[featIndex]) >= float(ans[0]):
            if type(secondDict['1_' + ans[0]]).__name__ == 'dict':
                classLabel = classify(secondDict['1_' + ans[0]], xlabel, testdata)
            else:
                classLabel = secondDict['1_' + ans[0]]
        else:
            if type(secondDict['0_' + ans[0]]).__name__ == 'dict':
                classLabel = classify(secondDict['0_' + ans[0]], xlabel, testdata)
            else:
                classLabel = secondDict['0_' + ans[0]]
        return int(classLabel)
    else:
        if float(testdata[featIndex]) == float(ans[0]):
            if type(secondDict['1_' + ans[0]]).__name__ == 'dict':
                classLabel = classify(secondDict['1_' + ans[0]], xlabel, testdata)
            else:
                classLabel = secondDict['1_' + ans[0]]
        else:
            if type(secondDict['0_' + ans[0]]).__name__ == 'dict':
                classLabel = classify(secondDict['0_' + ans[0]], xlabel, testdata)
            else:
                classLabel = secondDict['0_' + ans[0]]
        return int(classLabel)

# 预测测试集
def classifytest(inputTree, xlabel, testDataSet):
    classLabelAll = []  # 创建空列表
    for testVec in testDataSet:  # 遍历每条数据
        classLabelAll.append(classify(inputTree, xlabel, testVec))  # 将每条数据得到的特征标签添加到列表
    return np.array(classLabelAll)

import pandas
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
mission_code= 1

if mission_code==1:
    #导入数据集iris
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    dataset = pandas.read_csv(url, names=names) #读取csv数据
    dataset.head()
    X = dataset.iloc[:,:-1].values
    y = dataset.iloc[:,-1].values
    #将标签分别改为0，1，2
    for i in range(len(y)):
        if y[i]=='Iris-setosa':
            y[i]=0
        elif y[i]=='Iris-versicolor':
            y[i]=1
        elif y[i]=='Iris-virginica':
            y[i]=2
    fea_label = names[:-1]
else:
    wine = load_wine()
    data = wine.data
    target = wine.target
    fea_label = list(wine.feature_names)
    X= data
    y= target
    names= fea_label
n=0.1
for i in range(5):
    #划分训练、测试集
    x_train,x_test,y_train,y_test = train_test_split(X,y,test_size= n,random_state = 666)
    #生成CART分类树
    cartTree = create_CART_classify(x_train, y_train, fea_label)
    print(cartTree)
    classlist=classifytest(cartTree,names,x_test)
    #for i in range(10):
       # print(y_test[i], "--->",classlist[i])
    print("accuracy：%.4f"%np.mean(classlist==y_test))
    n+=0.1
