# @Author  : 周逸群 20191583
import math
import numpy as np

# 密度函数
def PDensity(m, v, x):
    return math.exp(-(float(x)-m)**2/2*v)/(math.sqrt(2*math.pi*v))

# 预测
def predict(data, features, BayesDic):
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

# 读取数据集
def getDataSet():
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

# 拉普拉斯修正
def multProbability(dataSet, index, value, classLabel, N):
    # 提取数据
    extrData= dataSet[dataSet[:,-1]==classLabel]
    # 计数
    cnt= 0
    for data in extrData:
        if data[index]== value:
            cnt+=1
    # 返回拉普拉斯修正值
    return (cnt+1)/(len(extrData)+ N)

# 贝叶斯朴素
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

def AODE(dataSet, data, features, featureDic):
    # 读取数据维度
    m, n= dataSet.shape
    # 不考虑连续属性
    n= n- 3
    # 保存分类概率
    pResult={}
    # 对数据进行测试
    for melon_class in ["好瓜", "坏瓜"]:
        # 分到当前类的概率
        thisP= 0.0
        sign = '1' if melon_class== "好瓜" else '0'
        # 提取同类别西瓜
        melon_of_thisClass= dataSet[dataSet[:, -1]== sign]
        # 计算概率
        for i in range(n):
            # 测试样本的i属性取值
            xi= data[i]
            # 计算训练集上同类i属性取值为xi的样本占比
            ## 同类训练集i属性取值为xi的样本集
            Dcxi= melon_of_thisClass[melon_of_thisClass[:, i]== xi]
            ## i属性的可能取值个数
            Ni= len(featureDic[features[i]])
            ## 占比
            Pcxi= (len(Dcxi)+ 1)/float(m + 2* Ni) # 书本7.24公式

            # 计算类别为c在第i和第j个属性上为xi和xj的样本，占类别为c第i属性取值为xi样本的占比
            product= 1
            # 对每个特征连乘
            for j in range(n):
                # 测试样本的j属性取值
                xj= data[j]
                # 类别为c在第i和第j个属性上为xi和xj的样本
                Dcxixj= Dcxi[Dcxi[:, j]== xj]
                # j属性的可能取值个数
                Nj= len(featureDic[features[j]])
                # 占比
                Pcxixj= (len(Dcxixj)+1)/float(len(Dcxi)+ Nj) # 书本7.25公式
                # 连乘
                product*= Pcxixj # 书本7.23公式
            thisP+= Pcxi* product
        pResult[melon_class]= thisP
    # 输出结果
    result= "好瓜" if pResult["好瓜"] > pResult["坏瓜"] else "坏瓜"
    return pResult["好瓜"], pResult["坏瓜"],result


# 主程序入口
# 获取数据集
dataSet, features, featureDic= getDataSet()
# 预测样本
print("预测样本：")
test_sample= dataSet[0]
print(test_sample)
# 开始预测
pG, pB,pre= AODE(dataSet[1:len(dataSet)-1, :], test_sample, features, featureDic) # 训练数据，测试数据，特征目录
# 输出结果
print("好瓜概率={}\n坏瓜概率={}\n预测结果={}\n".format(pG,pB, pre))