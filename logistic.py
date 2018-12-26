import numpy as np
import math

def loadFile():
	dataList=[]
	labelList=[]
	fr=open("testSet.txt")
	for lines in fr:
		line=lines.split()
		dataList.append([1.0, float(line[0]), float(line[1])])
		labelList.append(int(line[2]))

	return dataList, labelList

def sigmoid(x):
	return 1.0/(1+np.exp(-x))

def gradAscent(dataList, labelList):
	dataMatrix=np.mat(dataList)
	labelMatrix=np.mat(labelList).transpose()#transpose()矩阵转置
	m,n=np.shape(dataMatrix)
	alpha=0.001
	maxCycles=500
	weights=np.ones((n,1))
	for k in range(maxCycles):
		
		h=sigmoid(dataMatrix*weights)
		print(type(h))


def main():
	dataList, labelList=loadFile();
	gradAscent(dataList, labelList)

main()