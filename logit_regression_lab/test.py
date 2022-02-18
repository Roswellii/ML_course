import numpy as np
p1=np.ones((1,3))
p1= p1+ [1,2,3]
print(p1)
p= np.diag(p1.reshape(-1))
print(p)