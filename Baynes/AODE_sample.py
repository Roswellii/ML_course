import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import operator

# 特征字典，后面用到了好多次，干脆当全局变量了
featureDic = {
    '色泽': ['浅白', '青绿', '乌黑'],
    '根蒂': ['硬挺', '蜷缩', '稍蜷'],
    '敲声': ['沉闷', '浊响', '清脆'],
    '纹理': ['清晰', '模糊', '稍糊'],
    '脐部': ['凹陷', '平坦', '稍凹'],
    '触感': ['硬滑', '软粘']}


def getDataSet():
    """
    get watermelon data set 3.0.
    :return: 编码好的数据集以及特征的字典。
    """
    dataSet = [
        ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, 0.460, 1],
        ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.774, 0.376, 1],
        ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.634, 0.264, 1],
        ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.608, 0.318, 1],
        ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.556, 0.215, 1],
        ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.403, 0.237, 1],
        ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', 0.481, 0.149, 1],
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', 0.437, 0.211, 1],
        ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', 0.666, 0.091, 0],
        ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', 0.243, 0.267, 0],
        ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', 0.245, 0.057, 0],
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', 0.343, 0.099, 0],
        ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', 0.639, 0.161, 0],
        ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', 0.657, 0.198, 0],
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.360, 0.370, 0],
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', 0.593, 0.042, 0],
        ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', 0.719, 0.103, 0]
    ]

    features = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '密度', '含糖量']
    # features = ['color', 'root', 'knocks', 'texture', 'navel', 'touch', 'density', 'sugar']

    # #得到特征值字典，本来用这个生成的特征字典，还是直接当全局变量方便
    # featureDic = {}
    # for i in range(len(features)):
    #     featureList = [example[i] for example in dataSet]
    #     uniqueFeature = list(set(featureList))
    #     featureDic[features[i]] = uniqueFeature

    # 每种特征的属性个数
    numList = []  # [3, 3, 3, 3, 3, 2]
    for i in range(len(features) - 2):
        numList.append(len(featureDic[features[i]]))

    dataSet = np.array(dataSet)
    return dataSet, features


def AODE(dataSet, data, features):
    """
    AODE(Averaged One-Dependent Estimator)。意思为尝试将每个属性作为超父来构建SPODE。
    :param dataSet:
    :param data:
    :param features:
    :return:
    """
    print(data)
    print("---")
    m, n = dataSet.shape
    n = n - 3       # 特征不取连续值的属性，如密度和含糖量。
    pDir = {}       # 保存三个值。好瓜的可能性，坏瓜的可能性，和预测的值。
    for classLabel in ["好瓜", "坏瓜"]:
        P = 0.0
        if classLabel == "好瓜":
            sign = '1'
        else:
            sign = '0'
        extrDataSet = dataSet[dataSet[:, -1] == sign]    # 抽出类别为sign的数据
        for i in range(n):                               # 对于第i个特征
            xi = data[i] #?
            # 计算classLabel类，第i个属性上取值为xi的样本对总数据集的占比
            Dcxi = extrDataSet[extrDataSet[:, i] == xi]  # 第i个属性上取值为xi的样本数
            Ni = len(featureDic[features[i]])            # 第i个属性可能的取值数
            Pcxi = (len(Dcxi) + 1) / float(m + 2 * Ni) # 7.24
            # 计算类别为c且在第i和第j个属性上分别为xi和xj的样本，对于类别为c属性为xi的样本的占比
            mulPCond = 1
            for j in range(n): # 对于每一个特征
                xj = data[j] #
                Dcxij = Dcxi[Dcxi[:, j] == xj] # 类别为c,且在第i和第j个属性上为xi和xj的样本集合
                Nj = len(featureDic[features[j]])
                PCond = (len(Dcxij) + 1) / float(len(Dcxi) + Nj) # 7.25
                mulPCond *= PCond # 7.23
            P += Pcxi * mulPCond # 7.23
        pDir[classLabel] = P

    if pDir["好瓜"] > pDir["坏瓜"]:
        preClass = "好瓜"
    else:
        preClass = "坏瓜"

    return pDir["好瓜"], pDir["坏瓜"], preClass


def calcAccRate(dataSet, features):
    """
    计算准确率
    :param dataSet:
    :param features:
    :return:
    """
    cnt = 0
    for data in dataSet:
        _, _, pre = AODE(dataSet, data, features)
        if (pre == '好瓜' and data[-1] == '1') \
            or (pre == '坏瓜' and data[-1] == '0'):
            cnt += 1
    return cnt / float(len(dataSet))


def main():
    dataSet, features = getDataSet()
    pG, pB, pre = AODE(dataSet, dataSet[0], features) # 数据集，测试样本，特征字典
    print("pG = ", pG)
    print("pB = ", pB)
    print("pre = ", pre)
    print("real class = ", dataSet[0][-1])
    print(calcAccRate(dataSet, features))

if __name__ == '__main__':
    main()
