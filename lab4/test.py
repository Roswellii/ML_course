import numpy as np
a=[1,2,3,4]
b=[0,2,1,3]
a=np.asarray(a)
b=np.asarray(b)
print(a)
a=a[b]
print(a)