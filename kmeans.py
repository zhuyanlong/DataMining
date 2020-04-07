#当函数模块化以后，确实写起来会更方便些
#以zoo数据集为例
import numpy as np 
import random

def kmeans(dataSet,k):
    m,n=dataSet.shape
    data=list(range(m))
    index=random.sample(data,k)
    centroids=[]
    oldCentroids=None
    iterations=0
    for i in range(k):
        centroids.append(dataSet[index[i]])
    while not shouldStop(oldCentroids,centroids,iterations):
        oldCentroids=centroids
        iterations+=1
        labels=getLabels(dataSet,centroids)
        centroids=getCentroids(dataSet,labels,k)
    return centroids


def shouldStop(oldCentroids,centroids,iterations):
    if iterations>maxiterations:
        return True
    return oldCentroids==centroids

def getCentroids(dataSet,labels,k):
    

#我觉得这个函数我只需要得到两个，一个是数据编号，另一个是簇
def getLabels(dataSet,centroids):
    labels=[]
    m,n=dataSet.shape
    for i in range(m):
        tmp=[]
        for centroid in centroids:
            tmp.append((centroid,distance(dataSet[i],centroid))):
        result=sorted(tmp,key=lambda t:t[1])
        labels.append(result[0])
    return labels

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
