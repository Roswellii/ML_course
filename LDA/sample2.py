#-*- coding: UTF-8 -*-
from numpy import *
import numpy
def lda(c1,c2):
    #c1 第一类样本，每行是一个样本
    #c2 第二类样本，每行是一个样本

    #计算各类样本的均值和所有样本均值
    m1=mean(c1,axis=0)#第一类样本均值
    m2=mean(c2,axis=0)#第二类样本均值
    c=vstack((c1,c2))#所有样本
    m=mean(c,axis=0)#所有样本的均值

    #计算类内离散度矩阵Sw
    n1=c1.shape[0]#第一类样本数
    print(n1);
    n2=c2.shape[0]#第二类样本数
    #求第一类样本的散列矩阵s1
    s1=0
    for i in range(0,n1):
        s1=s1+(c1[i,:]-m1).T*(c1[i,:]-m1)
    #求第二类样本的散列矩阵s2
    s2=0
    for i in range(0,n2):
        s2=s2+(c2[i,:]-m2).T*(c2[i,:]-m2)
    Sw=(n1*s1+n2*s2)/(n1+n2)
    #计算类间离散度矩阵Sb
    Sb=(n1*(m-m1).T*(m-m1)+n2*(m-m2).T*(m-m2))/(n1+n2)
    #求最大特征值对应的特征向量
    eigvalue,eigvector=linalg.eig(mat(Sw).I*Sb)#特征值和特征向量
    indexVec=numpy.argsort(-eigvalue)#对eigvalue从大到小排序，返回索引
    nLargestIndex=indexVec[:1] #取出最大的特征值的索引
    W=eigvector[:,nLargestIndex] #取出最大的特征值对应的特征向量
    return W