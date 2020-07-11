#10-fold-CrossValidation
import numpy as np
import os

#split data set
#split_size is the k
def kfoldSplit(fileName, split_size,outdir):
    #if not outdir,makrdir
    if not os.path.exists(outdir): 
        os.makedirs(outdir)
    #open fileName to read
    fr = open(fileName,'r')
    num_line = 0
    onefile = fr.readlines()
    num_line = len(onefile) 
    #get a seq and set len=numLine
    arr = np.arange(num_line) 
    #generate a random seq from arr
    np.random.shuffle(arr) 
    #to list
    list_index = arr.tolist()
    #size of each split sets
    each_size = int((num_line+1) / split_size)+1 
    result = []
    each_split = []
    #count_num
    count_num = 0    
    #count_split
    count_split = 0   
                                   
    for i in range(len(list_index)):
        each_split.append(onefile[int(list_index[i])].strip()) 
        count_num += 1
        if count_num == each_size:
            count_split += 1 
            tmp = np.array(each_split)
            np.savetxt(outdir + "/split_" + str(count_split) + '.data',tmp,fmt="%s", delimiter='\t')
            result.append(each_split)
            each_split = []
            count_num = 0
    #the last split
    tmp = np.array(each_split)
    np.savetxt(outdir + "/split_" + str(count_split+1) + '.data',tmp,fmt="%s", delimiter='\t')
    result.append(each_split)
    #compute the number of data in result
    sum=0
    for i in range(10):
        sum+=len(result[i])
    print(sum)
    return result

def kfoldgenerate(datadir,outdir):
    #if not outdir,makedir
    if not os.path.exists(outdir): 
        os.makedirs(outdir)
    listfile = os.listdir(datadir)
    train = []
    test = []
    cross = 0
    for eachfile1 in listfile:
        train_data = []
        test_data = []
        cross += 1
        for eachfile2 in listfile:
            if eachfile2 != eachfile1:
                datafile=datadir + '/' + eachfile2
                fr = open(datafile, 'r')
                tmp = fr.readlines()
                for i in range(len(tmp)):
                    train_data.append(tmp[i])
                    
        with open(outdir +"/test_"+str(cross)+".datasets",'w') as fw_test:
            with open(datadir + '/' + eachfile1, 'r') as fr_testsets:
                for each_testline in fr_testsets:                
                    test_data.append(each_testline) 
            for oneline_test in test_data:
                fw_test.write(oneline_test)
            test.append(test_data)
        with open(outdir+"/train_"+str(cross)+".datasets",'w') as fw_train:
            for oneline_train in train_data:   
                oneline_train = oneline_train
                fw_train.write(oneline_train)
            train.append(train_data)
    return train,test

#split data
kfoldSplit('breast-cancer.data',10,'data')
kfoldgenerate('data',"outdir")