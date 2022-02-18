# 周逸群 20191583 CQU
# 2021-11
import math
import numpy as np
# 计算信息增益
def gain(attribute, labels, is_value=False):
    # 当前结点的信息熵
    info_gain = ent(labels)
    n = len(labels)
    split_value = None  # 如果是连续值的话，也需要返回分隔界限的值

    if is_value: # 取值连续
        # 取值从小到大排序
        sorted_attribute = attribute.copy()
        sorted_attribute.sort()
        # 计算候选分割点
        split = []
        for i in range(0, n - 1):
            temp = (sorted_attribute[i] + sorted_attribute[i + 1]) / 2
            split.append(temp)
        # 计算各点的信息增益
        info_gain_list = []
        for temp_split in split:
            low_labels = []
            high_labels = []
            # 按照大小分为两类
            for i in range(0, n):
                if attribute[i] <= temp_split:
                    low_labels.append(labels[i])
                else:
                    high_labels.append(labels[i])
            # 计算该点的信息增益
            temp_gain = info_gain - len(low_labels) / n * ent(low_labels) - len(high_labels) / n * ent(high_labels)
            info_gain_list.append(temp_gain)

        # print('info_gain_list', info_gain_list)
        info_gain = max(info_gain_list)
        max_index = info_gain_list.index(info_gain)
        split_value = split[max_index]
    else: # 取值离散
        attribute_dict = {}
        label_dict = {}
        index = 0
        for item in attribute:
            if attribute_dict.__contains__(item):
                attribute_dict[item] = attribute_dict[item] + 1
                label_dict[item].append(labels[index])
            else:
                attribute_dict[item] = 1
                label_dict[item] = [labels[index]]
            index += 1

        for key, value in attribute_dict.items():
            info_gain = info_gain - value / n * ent(label_dict[key])

    return info_gain, split_value


def ent(labels):
    # 统计不同分类下的样本数
    label_name = []
    label_count = []
    for item in labels:
        if not (item in label_name):
            label_name.append(item)
            label_count.append(1)
        else:
            index = label_name.index(item)
            label_count[index] = label_count[index] + 1
    # 由公式计算信息熵
    n = sum(label_count)
    entropy = 0.0
    for item in label_count:
        p = item / n
        entropy = entropy - p * math.log(p, 2)
    return entropy

def loss(attr_values, label):
    # 从小到大排列，分出多个点，尝试进行划分，分成两类计算平均值
    # label是训练样本的连续标签取值
    # 与label比较看loss，返回最小的loss以及对应的分割值
    sorted_values= attr_values.copy()
    sorted_values.sort()
    split_values=[]
    label_float=[]
    for i in range(len(label)):
             label_float.append(float(label[i]))
    for i in range(len(sorted_values)-1):
        split= (float(sorted_values[i])+float(sorted_values[i+1]))/2
        split_values.append(split)
    # 选出最小loss的分割点
    min_loss= float("inf")
    min_loss_split= -1
    loss= 0.0
    for split in split_values: # 对于每一个分隔值
        left_subtree=[]
        right_subtree=[]
        loss= 0.0
        # 左右分为两类
        for i in range(len(label_float)):
            if label_float[i]<= split:
                left_subtree.append(label_float[i])
            else:
                right_subtree.append(label_float[i])
        # 计算平均值
        left_avg= np.mean(left_subtree)
        right_avg= np.mean(right_subtree)
        # 计算loss
        for value in left_subtree:
            loss+= (value- left_avg)**2
        for value in right_subtree:
            loss+= (value- right_avg)**2
        # 比较
        if loss < min_loss:
            min_loss= loss
            min_loss_split= split
    return loss, min_loss_split

def process_node(current_node, data, label, misson_code,split_num,  max_split_num):
    split_num+= 1
    if(misson_code>=3 and (split_num>= max_split_num or len(current_node.data_index)<1)): # 叶子上只剩一个节点时提前终止
        return
    if (misson_code >= 3):
        sum = 0.0
        for index in current_node.data_index:
            sum += float(label[index])
        avg = sum / len(current_node.data_index)
        current_node.judge = avg
    n = len(label)
    # 叶子结点1-- 数据同类，设为叶子
    if(misson_code<=2):
        judge, stdata= judgeAllSame(current_node, data, label)
        if(judge):
            current_node.judge = stdata
            return
    # 叶子结点2-- 属性为空，设置为叶，分类从多
    rest_title = current_node.rest_attribute  # 候选属性
    if len(rest_title) == 0:  # 如果候选属性为空，则是个叶子结点。需要选择最多的那个类作为该结点的类
        if(misson_code<=2):
            label_count = {}
            temp_data = current_node.data_index
            for index in temp_data:
                if label_count.__contains__(label[index]):
                    label_count[label[index]] += 1
                else:
                    label_count[label[index]] = 1
            final_label = max(label_count)
            current_node.judge = final_label
        return
    # 叶子结点3-- 多个属性，信息增益，遍历讨论
    title_gain = {}  # 记录每个属性的信息增益
    title_split_value = {}  # 记录每个属性的分隔值，如果是连续属性则为分隔值，如果是离散属性则为None
    for title in rest_title: # 遍历属性
        attr_values = [] # 所有样本在某一个属性上的取值
        current_label = [] # 所有样本的标签
        for index in current_node.data_index:
            this_data = data[index] # 取样本, 包含了属性、标签信息
            attr_values.append(this_data[title]) # 属性
            current_label.append(label[index]) # 样本标签
        temp_data = data[0]
        if misson_code<=2: # 分类任务
            this_gain, this_split_value = gain(attr_values, current_label, is_number(temp_data[title], misson_code))  # 计算属性增益
        else: # 回归任务
            this_gain, this_split_value= loss(attr_values,current_label) # current_label在回归任务中是连续的取值
            this_gain= -this_gain # 取反
        title_gain[title] = this_gain # 保存该属性的增益
        title_split_value[title] = this_split_value # 保存该属性分隔值
    best_attr = max(title_gain, key=title_gain.get)  # 求最大信息增益
    current_node.attribute_to_split = best_attr # 设置当前结点的划分属性
    current_node.split = title_split_value[best_attr] # 设置当前结点的分割点
    rest_title.remove(best_attr) # 删除最优属性
    # 非叶子结点， 建树
    a_data = data[0]
        # 最优属性连续取值
    if is_number(a_data[best_attr], misson_code):  # 如果是该属性的值为连续数值
        split_value = title_split_value[best_attr] # 读取分割值
        small_data = []
        large_data = []
        for index in current_node.data_index:
            this_data = data[index]
            if float(this_data[best_attr]) <= split_value:
                small_data.append(index) # 存入小于分隔值的样本下标
            else:
                large_data.append(index) # 存入大于分隔值的样本下标
        small_str = '<=' + str(split_value)
        large_str = '>' + str(split_value)
        # 生成左子树
        if(misson_code<=2 or (len(small_data)!= 0 and len(large_data)!=0 )):
            small_child = TreeNode(parent=current_node, data_index=small_data, attr_value=small_str,
                                   rest_attribute=rest_title.copy())
            # 生成右子树
            large_child = TreeNode(parent=current_node, data_index=large_data, attr_value=large_str,
                                   rest_attribute=rest_title.copy())
        else:
           return
        # 设置左右子树
        current_node.children = [small_child, large_child]
        # 最优属性离散取值
    else:
        best_titlevalue_dict = {}  # key是属性值的取值，value是个list记录所包含的样本序号
        for index in current_node.data_index:
            this_data = data[index] # 取一个瓜
            if best_titlevalue_dict.__contains__(this_data[best_attr]): # 瓜的属性取值已保存
                temp_list = best_titlevalue_dict[this_data[best_attr]] # 找到属性取值对应的表
                temp_list.append(index) # 存瓜
            else: # 瓜的属性取值未保存
                temp_list = [index] # 新建表存瓜
                best_titlevalue_dict[this_data[best_attr]] = temp_list # 将表补充到结点上

        children_list = []
        for key, index_list in best_titlevalue_dict.items(): # 按照属性分类建树
            a_child = TreeNode(parent=current_node, data_index=index_list, attr_value=key,
                               rest_attribute=rest_title.copy())
            children_list.append(a_child) # 插入子树清单
        current_node.children = children_list # 更新子树清单

    # print(current_node.to_string())
    for child in current_node.children:  # 递归
        process_node(child, data, label, misson_code,split_num,max_split_num)

# 决策树结点
class TreeNode:
    current_index = 0
    # 初始化结点
    def __init__(self, parent=None, attr_name=None, children=None, judge=None, split=None, data_index=None,
                 attr_value=None, rest_attribute=None):

        self.parent = parent  # 父节点，根节点的父节点为 None
        self.attribute_to_split = attr_name  # 本节点上进行划分的属性名
        self.attribute_value = attr_value  # 本节点上划分属性的值，是与父节点的划分属性名相对应的
        self.children = children  # 孩子结点列表
        self.judge = judge  # 如果是叶子结点，需要给出判断
        self.split = split  # 如果是使用连续属性进行划分，需要给出分割点
        self.data_index = data_index  # 对应训练数据集的训练索引号
        self.index = TreeNode.current_index  # 当前结点的索引号
        self.rest_attribute = rest_attribute  # 尚未使用的属性列表
        TreeNode.current_index += 1

# 输出
    def to_string(self):
        node_desc = '当前结点 : ' + str(self.index) + ";\n"
        if not (self.parent is None):
            parent_node = self.parent
            node_desc = node_desc + '父结点 : ' + str(parent_node.index) + ";\n"
            node_desc = node_desc + str(parent_node.attribute_to_split) + " : " + str(self.attribute_value) + ";\n"
        node_desc = node_desc + "数据 : " + str(self.data_index) + ";\n"
        if not (self.children is None):
            node_desc = node_desc + '属性 : ' + str(self.attribute_to_split) + ";\n"
            child_list = []
            for child in self.children:
                child_list.append(child.index)
            node_desc = node_desc + '子结点 : ' + str(child_list)
        if not (self.judge is None):
            node_desc = node_desc + '标签 : ' + str(self.judge)
        return node_desc

# ?
def judgeAllSame(current_node, data, label):
    one_class = True
    this_data_index = current_node.data_index # 当前节点包含的样本参数集合
    for i in this_data_index:
        for j in this_data_index:
            if label[i] != label[j]:
                one_class = False
                break
        if not one_class:
            break
    print(this_data_index[0])
    return one_class, label[this_data_index[0]] # 返回是否全部为同一类，第一个样本的类别（全部同类时将其设为代表）

def is_number(s, misson_code):
    if misson_code==1:
        return False
    else:
        try:
            float(s)
            return True
        except ValueError:
            pass
        return False


def read_data(mission_code):
    dataset = []
    if(mission_code==1):
        filename= './classification_car/car.data'

        attribute =  ['buying','maint','doors','persons','lug_boot','safety']
    elif(mission_code==3):
        filename = './regression_boston/housing_small.data'
        attribute= ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PIRATIO','B','LSTAT']

    if(misson_code!=3):
        with open(filename) as file:
            for items in file:
                items = items.replace('\n', '')
                items = items.replace(',', ' ')
                item = items.split(' ')
                dataset.append(item)
    else:
        with open(filename) as file:
            for item in file:
                item = item.replace('\n', '')
                item = item.replace('   ', ' ')
                item = item.replace('  ', ' ')
                item = item.split(' ')
                item= item[1:]
                if(len(item)== 14): # 只使用不存在缺失值的数据
                    dataset.append(item)
    return dataset, attribute


def id3_tree(Data, title, label, misson_code):
    # 树初始化
    n = len(Data)
    rest_title = title.copy()
    root_data = []
    for i in range(0, n): # 依次0~n-1
        root_data.append(i)
    # 生成根结点
    root_node = TreeNode(data_index=root_data, rest_attribute=title.copy())
    max_split_num= 4
    split_num= 0
    process_node(root_node, Data, label, misson_code,split_num, max_split_num)

    return root_node


def print_tree(root=TreeNode()):
    node_list = [root]
    while (len(node_list) > 0):
        current_node = node_list[0]
        print('--------------------------------------------')
        print(current_node.to_string())
        print('--------------------------------------------')
        # 补充子结点
        children_list = current_node.children
        if not (children_list is None):
            for child in children_list:
                node_list.append(child)
        node_list.remove(current_node)


if __name__ == '__main__':
    misson_code= 3 # 1 for classification_car, 3 for regression_boston_housing,

    #从文件中读入样本与属性的名称
    items, attribute = read_data(misson_code)
    print(type(items), type(attribute))
    # 数据
    data = []
    # 标签
    label = []
    for item in items:
        item_dict = {}
        for i in range(0, len(item)-1):
            item_dict[attribute[i]] = item[i] # 将一条样本转化为字典的形式，可以用属性索引
        data.append(item_dict)  # 将一条样本插入数据集列表
        label.append(item[len(item)-1]) # 将样本的标签插入标签列表

    decision_tree = id3_tree(data, attribute, label, misson_code)
    print_tree(decision_tree)