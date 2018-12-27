#现在发现限制自己的是python编程能力
import numpy as np

#训练集为zootrain.csv，测试集为zootest.csv，k取值为10，原始数据集见zoo文件夹

#欧几里得距离
def EuclidDis(v1,v2):
	distance=np.sqrt(np.sum(np.square(v1-v2)))
	return distance

#写到这我突然发现写不下去了，感觉用的数据结构不太对，这种简单的数据结构就很难实现一些功能
def kNN():
	trainMatrix, trainLabel=loadFile("zootrain.csv")#训练集
	testMatrix, testLabel=loadFile("zootest.csv")#测试集，testLabel为真实类别
	trainSize=np.shape(trainMatrix)[0]#只获取行数
	testSize=np.shape(testMatrix)[0]
	for j in range(testSize):
		dis={}#dis是一个字典
		for i in range(trainSize):
			dis[i]=EuclidDis(trainMatrix[i],testMatrix[j])#计算欧几里得距离，key值可以用来定位type，value值是距离
		temp=sorted(dis.items(),key=lambda item:item[1])#降序排序，得到的是一个tuple，第一个值是在trainMatrix中的序号，第二个是距离，傻了，反而而最远距离来算了，应该是最近
		tp=[]
		dic={}
		for m in range(10):#选取前10个tuple
		#找出对应位置上的type，然后对10各种的type的频数进行统计
			#每出现一次，就加一
			tp.append(trainLabel[temp[m][0]])

		#统计list中数字出现的次数，用字典统计
		#key为tp中的元素
		#dic.get(key, default=None)
		#default如果指定键不存在时，返回默认值，所以如果键不存在，则设为0
		for key in tp:
			dic[key]=dic.get(key,0)+1

		#返回value最大的键
		print(max(dic,key=dic.get))


	# print(testLabel)


#读取文件
def loadFile(filePath):
	dataList=[]
	labelList=[]
	fr=open(filePath)
	for lines in fr.readlines():
		line=lines.split(',')
		dataList.append([float(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4]), float(line[5]), float(line[6]), float(line[7]),float(line[8]), float(line[9]), float(line[10]), float(line[11]), float(line[12]), float(line[13]), float(line[14]), float(line[15])])
		labelList.append(int(line[16]))#默认距离1
	dataMatrix=np.mat(dataList)
	return dataMatrix, labelList

def main():
	kNN()

main()