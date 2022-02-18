
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


rootpath='Yale'
X=[]
for person in range(15):
    for num in range(11):
        path=rootpath+'/'+str(person+1)+'/s'+str(num+1)+'.bmp'
        img=Image.open(path)
        X.append(np.array(img).reshape(-1))
X=np.array(X)

for i in range(3):
    for j in range(5):
        plt.subplot(3,5,i*5+j+1)
        plt.imshow(X[(i*5+j)*11,:].reshape(100,100),cmap='gray')
        plt.axis('off')
        plt.title('%d'%(i*5+j+1))
plt.show()

# PCA
pca=PCA(n_components=20)
Z=pca.fit_transform(X)
W=pca.components_

for i in range(5):
    for j in range(4):
        plt.subplot(4,5,i*4+j+1)
        plt.imshow(W[i*4+j,:].reshape(100,100),cmap='gray')
        plt.axis('off')
        plt.title('w%d'%(i*4+j+1))
plt.show()

X_re=pca.inverse_transform(Z)
for i in range(3):
    for j in range(5):
        plt.subplot(3,5,i*5+j+1)
        plt.imshow(X_re[(i*5+j)*11,:].reshape(100,100),cmap='gray')
        plt.axis('off')
        plt.title('%d'%(i*5+j+1))
plt.show()
