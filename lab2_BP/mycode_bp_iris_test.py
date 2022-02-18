import pandas as pd
# 打开文件
data_raw = pd.read_table('iris.data', delimiter=',')
# 读取数据
data= data_raw.values
data= list(data)
# 转化标记
for i in range(len(data_raw)):
    if data[i][4]=='setosa':
        data[i][4] = [1, 0, 0]
    elif data[i][4]=='versicolor':
        data[i][4] = [0, 1, 0]
    else:
        data[i][4] = [0, 0, 1]
return data

