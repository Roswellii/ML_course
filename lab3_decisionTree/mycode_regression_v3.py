import numpy as np
from sklearn.datasets import load_boston
from sklearn.datasets import load_diabetes
boston = load_boston()
diabetes = load_diabetes()

# 计算数据集的总方差
def reg_err(data_y):
    return np.var(data_y) * len(data_y)

def classify_get_best_fea(data_x, data_y, ops=(1, 4)):
    m, n = np.shape(data_x)
    final_s = ops[0]  # 停止的精度
    final_n = ops[1]  # 停止的样本最小划分数
    # 样本均为同类， 生成当前类的叶节点
    if len(np.unique(data_y)) == 1:
        return None, reg_leaf(data_y)

    # 获取最优特征和特征值
    total_err = reg_err(data_y)  # 总的误差
    best_err = np.inf
    best_fea_idx = 0
    best_fea_val = 0

    for i in range(n):
        feas = np.unique(data_x[:, i])
        for fea_val in feas:
            data_D1_x, data_D1_y, data_D2_x, data_D2_y = split_dataset(data_x, data_y, i, fea_val)
            # 不满足最小划分集合，不进行计算
            if data_D1_x.shape[0] < final_n or data_D2_x.shape[0] < final_n:
                continue
            con_err = reg_err(data_D1_y) + reg_err(data_D2_y)
            if con_err < best_err:
                best_err = con_err
                best_fea_idx = i
                best_fea_val = fea_val

    # 预剪枝，求解的误差小于最小误差停止继续划分
    if total_err - best_err < final_s:
        return None, reg_leaf(data_y)

    # 一直无法进行划分，在这里进行处理
    data_D1_x, data_D1_y, data_D2_x, data_D2_y = split_dataset(data_x, data_y, best_fea_idx, best_fea_val)
    if data_D1_x.shape[0] < final_n or data_D2_x.shape[0] < final_n:
        return None, reg_leaf(data_y)

    return best_fea_idx, best_fea_val

# 在节点上切分数据
def split_dataset(data_x, data_y, fea_axis, fea_value):
    if isinstance(fea_value, int) or isinstance(fea_value, float): # 判断连续与离散
        equal_Idx = np.where(data_x[:, fea_axis] <= fea_value)
        nequal_Idx = np.where(data_x[:, fea_axis] > fea_value)
    else:
        equal_Idx = np.where(data_x[:, fea_axis] == fea_value)
        nequal_Idx = np.where(data_x[:, fea_axis] != fea_value)
    return data_x[equal_Idx], data_y[equal_Idx], data_x[nequal_Idx], data_y[nequal_Idx]


# 叶子结点均值计算
def reg_leaf(data_y):
    return np.mean(data_y)


def create_CART_regression(data_x, data_y, ops=(1, 4)):
    #
    fea_idx, fea_val = classify_get_best_fea(data_x, data_y, ops)
    if fea_idx == None:
        return fea_val
    # 递归建立CART回归决策树
    CART_tree = {}
    CART_tree['fea_idx'] = fea_idx
    CART_tree['fea_val'] = fea_val
    # 当前节点切分数据
    data_D1_x, data_D1_y, data_D2_x, data_D2_y = split_dataset(data_x, data_y, fea_idx, fea_val)
    # 建立左右子树
    CART_tree['left'] = create_CART_regression(data_D1_x, data_D1_y, ops)
    CART_tree['right'] = create_CART_regression(data_D2_x, data_D2_y, ops)
    return CART_tree

# 预测一个样本
def classify(inputTree, testdata):
    first_fea_idx = inputTree[list(inputTree.keys())[0]]  # 对应的特征下标
    fea_val = inputTree[list(inputTree.keys())[1]]  # 对应特征的分割值

    classLabel = 0.0  # 定义变量classLabel，默认值为0

    if testdata[first_fea_idx] >= fea_val:  # 进入右子树
        if type(inputTree['right']).__name__ == 'dict':
            classLabel = classify(inputTree['right'], testdata)
        else:
            classLabel = inputTree['right']
    else:
        if type(inputTree['left']).__name__ == 'dict':
            classLabel = classify(inputTree['left'], testdata)
        else:
            classLabel = inputTree['left']

    return round(classLabel, 2)


# 预测测试集
def classifytest(inputTree, testDataSet): # 输入(树, 测试集）
    classLabelAll = []  # 创建空列表
    for testVec in testDataSet:  # 对于每一条测试样本
        classLabelAll.append(classify(inputTree, testVec))  # 保存预测结果
    return np.array(classLabelAll)

# 程序入口
misson_code= 1
if(misson_code==1):
    data = boston.data
    target = boston.target
else:
    data = diabetes.data
    target = diabetes.target

data= data[:500, :]
target= target[:500]

x_train = data[:350,:] # 前300个数据
y_train = target[:350]
x_test = data[351:,:] # 后300个数据
y_test = target[351:]
# 生成树
cartTree = create_CART_regression(x_train, y_train)
# 打印树
print(cartTree)
# 预测
classlist=classifytest(cartTree,x_test)
# 输出10个数据
# for i in range(10):
#     print(y_test[i], "--->", classlist[i])
# 平均误差
print("avg_error：",abs(np.sum(classlist)-np.sum(y_test))/len(y_test))
