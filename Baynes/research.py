extrData= dataSet[dataSet[:,-1]==classLabel]





dataExtra = dataSet[dataSet[:, -1] == label]
extr = dataExtra[:, index].astype('float64')
# 均值
aver = extr.mean()
# 方差
var = extr.var()





classLabels = dataSet[:, -1].tolist()


print(f"p1 = {p1}")
print(f"p0 = {p0}")
print(f"pre = {pre}")