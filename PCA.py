import numpy as np

#二维原始数据
li = [[1,1],[1,3],[2,3],[4,4],[2,4]]
#矩阵化
matrix=np.mat(li)
#计算平均值
mean_matrix=np.mean(matrix,axis=0)

#减去平均值
Dataadjust=matrix-mean_matrix

#协方差矩阵
covMatrix=np.cov(Dataadjust,rowvar=0)

#eigValues为特征值，eigVectors为特征向量
eigValues,eigVectors=np.linalg.eig(covMatrix)

#argsort函数返回的是数组值从小到大的索引值
eigValuesIndex=np.argsort(eigValues)

#保留前k个最大的特征值
#这条语句是什么意思呢？
eigValuesIndex=eigValuesIndex[:-10000000:-1]

#计算出对应的特征向量
trueEigVectors=eigVectors[:,eigValuesIndex]

#选择较大特征值对应的特征向量
maxvectors_eigval=trueEigVectors[:,0]

pca_result=maxvectors_eigval*Dataadjust.T

print(pca_result)