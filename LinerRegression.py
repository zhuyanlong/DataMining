#采用梯度下降法
import numpy as np
import pandas as pd
import random

def BatchGradientDescent(x,y,theta,alpha,m,maxIteration):
    xTrains=x.transpose()
    for i in range(0,maxIteration):
        #矩阵点乘就是矩阵的乘法
        hypothesis=np.dot(x,theta)
        loss=hypothesis-y
        #除以m实际上是为了得出梯度的平均值
        gradient=np.dot(xTrains,loss)/m
        theta=theta-alpha*gradient
    return theta

def StochasticGradientDescent(x,y,theta,alpha,m,maxIteration):
    #一共有十个数据，所以要随机选择，就生成一个数组，从里面挑就可以
    data=list(range(m))
    xTrains=x.transpose()
    for i in range(0,maxIteration):
        hypothesis=np.dot(x,theta)
        loss=hypothesis-y
        #从data中随机抽取1个数据
        index=random.choice(data)
        #用抽取的这个值计算梯度
        gradient=loss[index]*x[index]
        theta=theta-alpha*gradient
    return theta

def predict(x,theta):
    m,n=x.shape
    xTest=np.ones((m,n+1))
    #第一个:选中所有的行。第二个:-1选中除最后一列外的其他列
    xTest[:,:-1]=x
    res=np.dot(xTest,theta)
    return res
#最后一个数据，1是偏置
trainData = np.array([[1.1,1.5,1],[1.3,1.9,1],[1.5,2.3,1],[1.7,2.7,1],[1.9,3.1,1],[2.1,3.5,1],[2.3,3.9,1],[2.5,4.3,1],[2.7,4.7,1],[2.9,5.1,1]])
trainLabel = np.array([2.5,3.2,3.9,4.6,5.3,6,6.7,7.4,8.1,8.8])

#m是数据行数
m,n=np.shape(trainData)
#theta权重矩阵，相当于最后要学得的模型
theta=np.ones(n)
#学习率
alpha=0.1
#迭代次数
maxIteration=100

theta=BatchGradientDescent(trainData,trainLabel,theta,alpha,m,maxIteration)
# print("BGD",theta)
x = np.array([[3.1, 5.5], [3.3, 5.9], [3.5, 6.3], [3.7, 6.7], [3.9, 7.1]])
print(predict(x,theta))
theta=StochasticGradientDescent(trainData,trainLabel,theta,alpha,m,maxIteration)
x = np.array([[3.1, 5.5], [3.3, 5.9], [3.5, 6.3], [3.7, 6.7], [3.9, 7.1]])
print(predict(x,theta))
# print("SGD",theta)