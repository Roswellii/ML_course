#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Jia ShiLin

import pandas as pd
import numpy as np

t1 = pd.DataFrame(np.arange(12).reshape(3, 4), index=list('abc'), columns=list('wxyz'))
print(t1)

# loc和iloc,[,]逗号左右代表行和列,可是是某个值,也可以是切片
# loc通过标签获取
# iloc通过下标获取,位置
# loc
az = t1.loc["a", 'z']
z = t1.loc[:, 'z']  # 可以切片
print(az)
print(z)
# 取第a行,和第c行[[],[]],取连续范围[,]
ac = t1.loc[['a', 'c'], :]
acc = t1.loc['b': 'c', :]
print(ac)
print(acc)
# iloc
iac = t1.iloc[[2, 1], :]
iacc = t1.iloc[:, 1:2]
print(iac)
print(iacc)