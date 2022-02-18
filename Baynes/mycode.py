# @Author  : 周逸群 20191583
import math
import numpy as np

def PDensity(m, v, x):
    """
    概率密度函数
    :param m:
    :param v:
    :param x:
    :return:
    """
    return math.exp(-(float(x)-m)**2/2*v)/(math.sqrt(2*math.pi*v))

def predict(data, features, BayesDic):
    """
    预测
    :param data:
    :param features:
    :param BayesDic:
    :return:
    """
    # 先验概率
    pGood= BayesDic['好瓜']['是']
    pBad= BayesDic['好瓜']['否']
    # 条件概率连乘
    for feature in features:
        # 获取属性对应的下标
        index= features.index(feature)
        if feature != '密度' and feature != '含糖量':
            # 属性条件概率连乘
            pGood *= BayesDic[feature][data[index]]['是']
            pBad *= BayesDic[feature][data[index]]['否']
        else:
            # 代入待预测样本的值
            pGood*= PDensity(BayesDic[feature]['是']['平均值'], BayesDic[feature]['是']['方差'], data[index])
            pBad*= PDensity(BayesDic[feature]['否']['平均值'], BayesDic[feature]['否']['方差'], data[index])
    # 预测结果
    predictClass=""
    if pBad>pGood:
        predictClass='坏瓜'
    else:
        predictClass= '好瓜'
    return pGood, pBad, predictClass

def getDataSet():
    """
    数据集读入
    :return:
    """
    # 属性
    features = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '密度', '含糖量']
    # 数据集
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
    dataSet = np.array(dataSet)
    # 属性取值
    featureDic = {
        '色泽': ['浅白', '青绿', '乌黑'],
        '根蒂': ['硬挺', '蜷缩', '稍蜷'],
        '敲声': ['沉闷', '浊响', '清脆'],
        '纹理': ['清晰', '模糊', '稍糊'],
        '脐部': ['凹陷', '平坦', '稍凹'],
        '触感': ['硬滑', '软粘']}
    return dataSet,features, featureDic

def multProbability(dataSet, index, value, classLabel, N):
    """
    计算先验概率、条件概率
    :param dataSet:
    :param index:
    :param value:
    :param classLabel:
    :param N:
    :return:
    """
    # 提取数据
    extrData= dataSet[dataSet[:,-1]==classLabel]
    # 计数
    cnt= 0
    for data in extrData:
        if data[index]== value:
            cnt+=1
    # 返回拉普拉斯修正值
    return (cnt+1)/(len(extrData)+ N)

def naiveBayesClassifier(dataSet, features, featureDic):
    dict={}
    # 每一个属性
    for feature in features:
        # 获取属性对应的下标
        index = features.index(feature)
        # 创建每一个属性的字典
        dict[feature]= {}
        # 离散值
        if feature !='密度' and feature != '含糖量':
            # 获得属性取值列表
            featIList= featureDic[feature]
            # 每一个属性取值的好坏概率
            for value in featIList:
                pthisGood= multProbability(dataSet, index, value, '1', len(featIList))
                pthisBad = multProbability(dataSet, index, value, '0', len(featIList))
                dict[feature][value] = {}
                dict[feature][value]["是"] = pthisGood
                dict[feature][value]["否"] = pthisBad
        # 连续值
        else:
            for label in ['1', '0']:
                dataExtra= dataSet[dataSet[:, -1]==label]
                extr= dataExtra[:, index].astype('float64')
                # 均值
                aver= extr.mean()
                # 方差
                var= extr.var()
                # 转换
                labelStr= ""
                if label== '1':
                    labelStr= '是'
                else:
                    labelStr= '否'
                # 保存
                dict[feature][labelStr]= {}
                dict[feature][labelStr]['平均值']= aver
                dict[feature][labelStr]['方差']= var
    # 计算类概率
    length= len(dataSet)
    classLabels= dataSet[:, -1].tolist()
    # 拉普拉斯修正后的类概率
    dict['好瓜']= {}
    dict['好瓜']['是']= (classLabels.count('1')+1)/(float(length)+2)
    dict['好瓜']['否']= (classLabels.count('0')+1)/(float(length)+2)
    return dict


# 主程序入口
# 获取数据集
dataSet, features, featureDic= getDataSet()
# 获取字典
dic= naiveBayesClassifier(dataSet, features, featureDic)
# 预测
pG, pB,pre= predict(dataSet[0], features, dic)
# 输出结果
print("好瓜概率={}\n坏瓜概率={}\n预测结果={}\n".format(pG,pB, pre))