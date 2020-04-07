#当函数模块化以后，确实写起来会更方便些
#以zoo数据集为例
import random
import numpy as np
import pandas as pd

#It will read header as list
def loadData(path):
    data= open(path,'r')
    df=pd.read_csv(path)
    return df

def kmeans(dataSet,k):
    m,n=dataSet.shape
    dataSet=dataSet.values
    data=list(range(m))
    index=random.sample(data,k)
    centroids=[]
    oldCentroids=centroids.copy()
    iterations=0
    for i in range(k):
        centroids.append(dataSet[index[i]])
    while not shouldStop(oldCentroids,centroids,iterations):
        oldCentroids=centroids.copy()
        iterations+=1
        labels=getLabels(dataSet,centroids)
        centroids=getCentroids(labels,k,centroids)
    return centroids


def shouldStop(oldCentroids,centroids,iterations):
    maxiterations=10
    if iterations>maxiterations:
        return True
    return False
    # judge=1
    # for i in range(len(oldCentroids)):
    #     if not (oldCentroids[i]==centroids[i]).all():
    #         judge=0
    #         break
    # if judge==0:
    #     return False
    # else:
    #     return True

def getCentroids(labels,k,centroids):
    newcentroids=[]
    m=len(labels)
    for i in range(k):
        tmp=np.zeros(len(centroids[0]))
        count=0
        for j in range(m):
            #这里因为使用的是array，所以判断相等的方法也是不一样的
            if (labels[j][0]==centroids[i]).all():
                tmp+=labels[j][1]
                count+=1
        tmp/=count
        newcentroids.append(tmp)
    print(newcentroids)
    return newcentroids

#我觉得这个函数我只需要得到两个，一个是数据编号，另一个是簇
def getLabels(dataSet,centroids):
    labels=[]
    m=len(dataSet)
    for i in range(m):
        tmp=[]
        for centroid in centroids:
            tmp.append((centroid,dataSet[i],distance(dataSet[i],centroid)))
        result=sorted(tmp,key=lambda t:t[-1])
        labels.append(result[0][0:2])
    return labels

#距离函数是根据数据的类型特制的
def distance(data,centroid):
    dis=0
    nomi=0
    #if same, distance=0
    if data[12]==centroid[12]:
        nomi=0
    else:
        nomi=1
    #q:i:1 j:1 
    #r:i:1 j:0
    #s:i:0 j:1 
    #t:i:0 j:0
    q=0; r=0; s=0; t=0
    attrange1=list(range(0,12))
    attrange2=list(range(13,16))
    attrange=attrange1+attrange2
    for i in attrange:
        if data[i]==centroid[i]:
            if data[i]==1:
                q+=1
            else:
                t==1
        else:
            if data[i]==1:
                r+=1
            else:
                s+=1
    if q+r+s+t==0:
        booldis=0
    else:
        booldis=float(r+s)/float(q+r+s+t)
    #zoo has 16 attributes
    dis=float(nomi)/16.0+booldis*15.0/16.0
    return dis

def main():
    data=loadData('data\\zoo.csv')
    data.drop(columns=['type'])
    k=3
    print(kmeans(data,k))

main()
