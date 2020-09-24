#机器学习实战 逻辑回归
import numpy as np
def loadData():
    dataMat=[]
    labelMat=[]
    fr=open("testSet.txt")
    for line in fr.readlines():
        line=line.strip().split()
        dataMat.append([1.0,float(line[0]),float(line[1])])
        labelMat.append(int(line[2]))
    return dataMat,labelMat

def sigmoid(ipt):
    return 1.0/(1.0+np.exp(-ipt))

def gradAscent(dataMat,labelMat):
    dataMatrix=np.mat(dataMat)
    labelMatrix=np.mat(labelMat).transpose()
    m,n=np.shape(dataMatrix)
    alpha=0.001
    maxCycles=500
    #n行1列
    weights=np.ones((n,1))
    for k in range(maxCycles):
        h=sigmoid(dataMatrix*weights)
        error=(labelMatrix-h)
        #这项是梯度上升，dataMatrix.transpose()*error 是梯度
        weights=weights+alpha*dataMatrix.transpose()*error
    return weights

# def main():
#     path="testSet.txt"
#     dataMat,labelMat=loadData(path)
#     gradAscent(dataMat,labelMat)
#     # print(sigmoid(3))

# main()