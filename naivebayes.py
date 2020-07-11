import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import operator

class NaiveBayes:
    def __init__(self,smooth=1):
        self.smooth=smooth
        self.priorprobability={}
        self.conditionprobability={}
        self.labeldict=Counter()
        self.nums_s=0
        
    def train(self,data_train,label_train):
        data_num=len(label_train)
        self.labeldict=Counter(label_train)
        K=len(self.labeldict)
        #compute prior probability
        #Laplace smoothing
        for key, val in self.labeldict.items():
            self.priorprobability[key]=(val+self.smooth)/(data_num+K*self.smooth)
        for d in range(data_train.shape[1]):
            nums_attribute = dict()
            attribute = data_train[:, d]
            self.nums_s = len(np.unique(attribute))
            for xd, y in zip(attribute, label_train):
                if (xd,y) not in nums_attribute.keys():
                    nums_attribute[(xd, y)]=0
                nums_attribute[(xd, y)] += 1
            for key, val in nums_attribute.items():
                self.conditionprobability[(d, key[0], key[1])] = (val + self.smooth) / (self.labeldict[key[1]] + self.nums_s * self.smooth)
    def evaluation(self,fold=10):
        sum=[]
        tmp=0
        for count in range(fold):
            tmp=0
            data_train,label_train=loadDataSet('outdir/train_'+str(count+1)+'.datasets')
            data_test,label_test=loadDataSet('outdir/test_'+str(count+1)+'.datasets')
            self.train(data_train,label_train)
            for i in range(len(label_test)):
                predictlabel=self.predict(data_test[i])
                if predictlabel[0]==label_test[i]:
                    tmp+=1
            sum.append(tmp/len(label_test))
            print("The",count,"iteration, the precision is:",tmp/len(label_test))
        tmp=0
        for i in range(fold):
            tmp+=sum[i]
        print("The average precision is:",tmp/fold)
    def predict(self,input_v):
        predict_value={}
        for y,p_y in self.priorprobability.items():
            p=p_y
            for d,v in enumerate(input_v):
                if (d,v,y) in self.conditionprobability.keys():
                    p*=self.conditionprobability[(d,v,y)]
                else:
                    p*=(self.smooth)/(self.nums_s+self.labeldict[y])
                predict_value[y]=p
        predict_value_sorted=sorted(predict_value.items(),key=operator.itemgetter(1),reverse=True)
        return predict_value_sorted[0]
    

        
#load Data
def loadDataSet(openfile):
    data= pd.read_csv(openfile, header=None)
    data=data.drop([0],axis=1)
    #specify class column
    classes=data[5].tolist()
    tmp=data.drop([5],axis=1) 
    attribute=tmp.values
    return attribute,classes
    
def main():
    nb=NaiveBayes()
    nb.evaluation()

main()