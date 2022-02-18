# https://github.com/han1057578619/MachineLearning_Zhouzhihua_ProblemSets/tree/master/ch8--%E9%9B%86%E6%88%90%E5%AD%A6%E4%B9%A0
# https://zhuanlan.zhihu.com/p/51206123?from_voters_page=true
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.utils import resample

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

        # 这里第一次尝试使用排序后的点作为分割点，效果很差，因为那样会错过一些更好的分割点；
        # 所以后来切割点改成将最大值和最小值之间分割成20等份。

        # 平均分配得到分割点
        split_points = [(x_max - x_min) / 20 * i + x_min for i in range(20)]

        # ？配置两种建树方法
        for inequal in ['lt', 'gt']:
            # 将每一个分割点作为阈值
            for thresh_val in split_points:
                # 用树桩决策
                ret_array = stumpClassify(X, dim, thresh_val, inequal)
                # 划分错误的比例
                error = np.mean(ret_array != y)

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

# 建立树桩集
def stumpBagging(X, y, nums=20):
    stumps = []
    seed = 16
    for _ in range(nums):
        X_, y_ = resample(X, y, random_state=seed)  # sklearn 中自带的实现自助采样的方法。
        seed += 1 # 相同的seed会有同样的采样结果
        stumps.append(buildStump(X_, y_)) # 建桩并插入树桩集
    return stumps

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
def pltStumpBaggingDecisionBound(X_, y_, stumps):
    pos = y_ == 1
    neg = y_ == -1
    # 含糖率
    x_tmp = np.linspace(0, 1, 600)
    # 密度
    y_tmp = np.linspace(-0.1, 0.7, 600)
    # 生成采样点
    X_tmp, Y_tmp = np.meshgrid(x_tmp, y_tmp)
    # 对每一个交点进行预测
    Z_ = stumpPredict(np.c_[X_tmp.ravel(), Y_tmp.ravel()], stumps).reshape(X_tmp.shape) # 改为一个采样点维度的向量
    # 标记预测结果，画等高线图（z为高度）
    plt.contour(X_tmp, Y_tmp, Z_, [0], colors='orange', linewidths=1)

    plt.scatter(X_[pos, 0], X_[pos, 1], label='1', color='c')
    plt.scatter(X_[neg, 0], X_[neg, 1], label='0', color='lightcoral')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    data_path = r'..\data\watermelon3_0a_Ch.txt'

    data = pd.read_table(data_path, delimiter=' ')

    X = data.iloc[:, :2].values # 前两列
    y = data.iloc[:, 2].values # 第二列

    y[y == 0] = -1 # 分类标签设置为-1

    stumps = stumpBagging(X, y, 21) # 建立树桩

    print(np.mean(stumpPredict(X, stumps) == y)) # 重新分析，输出成立的占比
    pltStumpBaggingDecisionBound(X, y, stumps)