# ML_EnsembleLearing
# 周逸群 20191583 2021.11
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.utils import resample

# 西瓜数据集alpha3.0
# 各作业通用
def getDataSet():
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
    return dataSet

# ？
def stumpClassify(X, dim, thresh_val, thresh_inequal):
    # 准备好矩阵
    # 行数= 数据维度数，列数= 1
    ret_array = np.ones((X.shape[0], 1))

    # 配置设定方法
    if thresh_inequal == 'lt':
        # 小于阈值的部分设为-1
        ret_array[X[:, dim] <= thresh_val] = -1
    else:
        # 大于阈值的部分设为-1
        ret_array[X[:, dim] > thresh_val] = -1
    # ret_array以阈值为界，一边为-1. 一边为1
    return ret_array

# 建立一个树桩
def buildStump(X, y):
    m, n = X.shape
    best_stump = {}
    min_error = 1

    for dim in range(n): # n是属性个数？

        # 第dim各属性的取值上下限
        x_min = np.min(X[:, dim])
        x_max = np.max(X[:, dim])
        # 平均分配得到分割点
        split_points = [(x_max - x_min) / 20 * i + x_min for i in range(20)]
        # 两种建树方法
        for inequal in ['lt', 'gt']:
            # 将每一个分割点作为阈值
            for thresh_val in split_points:
                # 用树桩决策
                ret_array = stumpClassify(X, dim, thresh_val, inequal)
                # 划分错误的比例
                cnt= 0
                for i in range(len(y)):
                    if ret_array[i]== y[i]:
                        cnt= cnt+ 1
                error= 1-cnt/float(len(y))

                # 如果有更好的划分，更新
                if error < min_error:
                    # 记录维度（维度之间也有竞争）
                    best_stump['dim'] = dim
                    # 记录阈值
                    best_stump['thresh'] = thresh_val
                    # 记录划分方法
                    best_stump['inequal'] = inequal
                    # 记录误差
                    best_stump['error'] = error
                    min_error = error
    # 返回最好的树
    return best_stump

# 利用树桩集进行预测
def stumpPredict(X, stumps):
    # 每一个样本对应一行，行列交叉即为预测结果
    # 相当于将之前建树桩的操作重新执行
    ret_arrays = np.ones((X.shape[0], len(stumps)))
    # 对于每一个树桩
    for i, stump in enumerate(stumps):
        # 输入样本以及每一个树桩对应的维度、阈值、划分方法, 进行预测
        ret_arrays[:, [i]] = stumpClassify(X, stump['dim'], stump['thresh'], stump['inequal'])
    # 求和投票
    return np.sign(np.sum(ret_arrays, axis=1))

# 作图
def visulize(X_, y_, stumps):
    good_melonx= []
    good_melony= []
    bad_melonx= []
    bad_melony= []
    for i in range(len(y_)):
        if y_[i]==1:
            good_melonx.append(X_[i, 0])
            good_melony.append(X_[i, 1])
        else:
            bad_melonx.append(X_[i, 0])
            bad_melony.append(X_[i, 1])


    # 含糖率
    x_tmp = np.linspace(0, 1, 600)
    # 密度
    y_tmp = np.linspace(-0.1, 0.7, 600)
    # 生成采样点
    X_tmp, Y_tmp = np.meshgrid(x_tmp, y_tmp)
    # 对每一个交点进行预测
    Z_ = stumpPredict(np.c_[X_tmp.ravel(), Y_tmp.ravel()], stumps).reshape(X_tmp.shape) # 改为一个采样点维度的向量
    # 标记预测结果，画等高线图（z为高度）
    plt.contour(X_tmp, Y_tmp, Z_, [0], colors='green', linewidths=5)

    plt.scatter(good_melonx, good_melony, label='Good', color='grey')
    plt.scatter(bad_melonx, bad_melony, label='Bad', color='black')
    plt.legend()
    plt.show()


if __name__ == "__main__":

    data= np.array(getDataSet())
    # 读取密度、含糖量
    X = data[:,6:8].astype('float64')
    # 读取标签
    y = data[:, -1:].astype('float64')
    # 为了便于集成，坏瓜设置为-1
    y[y == 0] = -1

    # 建立树桩
    stumps = []
    seed = 16
    for _ in range(40):
        # 自助采样
        x_selc, y_selc = resample(X, y, random_state=seed)
        # 相同的seed会有同样的采样结果，所以要变化
        seed += 1
        # 建桩并插入树桩集
        stumps.append(buildStump(x_selc, y_selc))
    # 可视化输出，对图上每个点进行预测
    visulize(X, y, stumps)