import csv

#It will read header as list
def loadData(path):
    data= open(path,'r')
    reader=csv.reader(data)
    return reader

#instance is a data which need to be predicted
#type is list
class kNN:
    def __init__(self,path,k=9):
        self.data=loadData(path)
        self.k=k
        
    def predict(self,instance):
        dis=[]
        count=0
        for item in self.data:
            if count==0:
                count+=1
                continue
            tmp=self.distance(item,instance)
            dis.append((item[16],tmp))
        #sort by tuple[1]
        result=sorted(dis,key=lambda t:t[1])
        dic={}
        for i in range(self.k):
            if result[i][0] not in dic:
                dic[result[i][0]]=1
            else:
                dic[result[i][0]]+=1
        return max(dic,key=dic.get)
            
    #columns [0,11]:boolean, 12,16:nominal,[13,15]:boolean
    #type traindata:list, testdata:list
    #every attribute has the same weight
    def distance(self,traindata,testdata):
        dis=0
        nomi=0
        #if same, distance=0
        if traindata[12]==testdata[12]:
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
            if traindata[i]==testdata[i]:
                if traindata[i]==1:
                    q+=1
                else:
                    t==1
            else:
                if traindata[i]==1:
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
    s=kNN('data\\zoo.csv')
    label=s.predict(['1','0','0','1','0','0','1','1','1','1','0','0','4','0','0','1'])
    print(label)
main()