import numpy as np
import math

#读入数据
def load_data():
    input_data=[[1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1],
                [1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1],
                [1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1],
                [0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0],
                [1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1],
                [1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0],
                [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
                [1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0]]
    labels=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    return input_data,labels
 
 #训练数据
def train(input_data,y,iteration,rate):
    #初始权值为0
    w=np.zeros(len(input_data[0]))
    #矩阵化
    w=np.mat(w)
    bias=0.0#偏置
    for i in range(iteration):
        samples= zip(input_data,y)
        for (input_i,label) in samples:#对每一组样本
            result=input_i*w.T+bias
            y_pred=sign(result)#计算输出值 y^
            w=w+rate*(label-y_pred)*np.array(input_i)#更新权重
            bias=bias+(label-y_pred)#更新bias
    return w,bias

#激活函数，阶跃函数
def sign(x):
    if x>=0:
        return 1
    else:
        return 0

#激活函数，sigmoid
def sigmoid(x):
    f=1.0/(1.0+math.exp(-x))
    if f>=0.5:
        return 1
    else:
        return 0


#预测类别
def predict(input_i,w,b):
    result=input_i*w.T+b
    result=sum(result)
    y_pred=sign(result)
    print(y_pred)
    
if __name__=='__main__':
    input_data,y=load_data()
    #迭代次数
    iteration=100
    #学习率
    rate=0.01
    #训练数据，获得w,b值
    w,b=train(input_data,y,iteration,rate)

    #测试数据
    predict([1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1],w,b)
    predict([1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1],w,b)