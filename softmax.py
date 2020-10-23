import numpy as np

def loadData(filepath):
    dataMat=[]
    labelMat=[]
    fr=open(filepath)
    for line in fr:
        lineArr=line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def one_hot(label_arr,n_samples,n_classes):
    y_one_hot=np.zeros((n_samples,n_classes))
    for i in range(len(label_arr)):
        y_one_hot[i][label_arr[i]]=1

    #不太能看得懂这条语句
#     y_one_hot[np.arange(n_samples), label_arr.T] = 1
    return y_one_hot

def train(data_arr,label_arr,iters=1000,alpha=0.1,lam=0.01):
    n_samples,n_features=data_arr.shape
    #计算类别数
    n_classes=len(set(label_arr))
    #随机产生系数
    weights=np.random.rand(n_classes,n_features)
    all_loss=list()
    y_one_hot=one_hot(label_arr,n_samples,n_classes)
    for i in range(iters):
        #计算后验概率,矩阵点乘就是行和列的相乘
        scores=np.dot(data_arr,weights.T)
        probs=softmax(scores)

        #计算损失函数值，对数损失
        loss=-(1.0/n_samples)*np.sum(y_one_hot*np.log(probs))
        all_loss.append(loss)

        #求解梯度值，最后一项为正则项
        dw=-(1.0/n_samples)*np.dot((y_one_hot-probs).T,data_arr)+lam*weights
        dw[:,0]=dw[:,0]-lam*weights[:,0]

        #更新权重
        weights=weights-alpha*dw
    return weights,all_loss
    
def softmax(scores):
    #按行求和，求每一行的和
    sum_exp=np.sum(np.exp(scores),axis=1,keepdims=True)
    softmax=np.exp(scores)/sum_exp
    return softmax
   
def predict(test_dataset,label_arr,weights):
    scores=np.dot(test_dataset,weights.T)
    probs=softmax(scores)
    return np.argmax(probs,axis=1).reshape((-1,1))
    
def main():
#     label_arr=np.array([0,1,2,3,0,1,2,3])
#     one=one_hot(label_arr,8,4)
#     print(one)
    dataMat,labelMat=loadData("train_dataset.txt")
    dataArr=np.array(dataMat)
    labelArr=np.array(labelMat)
    weights,all_loss=train(dataArr,labelArr)
    test_data_arr,test_label_arr=loadData("test_dataset.txt")
    test_data_arr=np.array(test_data_arr)
    test_label_arr=np.array(test_label_arr).reshape((-1,1))
    n_test_samples = test_data_arr.shape[0]
    y_predict = predict(test_data_arr, test_label_arr, weights)
    accuray = np.sum(y_predict == test_label_arr) / n_test_samples
    print(accuray)
    
main()