import numpy as npy
from sklearn.svm import SVC
import scipy.io as scio

def EvalAccuracy(evalLabel,gtLabel,labelSet):
    numClass=len(labelSet)
    confMat=npy.zeros((numClass,numClass),dtype=npy.int32)
    vecOnes=npy.ones(len(gtLabel))
    for ii in labelSet:
        for jj in labelSet:
            confMat[ii,jj]=npy.sum(vecOnes[npy.logical_and(evalLabel==ii, gtLabel==jj)])
    
    oa=npy.float32(npy.sum(npy.diagonal(confMat)))/ npy.sum(confMat) 
    return (confMat,oa)  

def ReadCIFAR10Gist():     
    dataTemp=scio.loadmat("data\\cifar10\\cifar10_test_batch.mat") 
    testData=dataTemp["gistFeat"]
    testLabel=npy.ravel(dataTemp["labels"])
    dataTemp=scio.loadmat("data\\cifar10\\cifar10_train_batch1.mat") 
    trainData1=dataTemp["gistFeat"]
    trainLabel1=npy.ravel(dataTemp["labels"])
    dataTemp=scio.loadmat("data\\cifar10\\cifar10_train_batch2.mat") 
    trainData2=dataTemp["gistFeat"]
    trainLabel2=npy.ravel(dataTemp["labels"])
    dataTemp=scio.loadmat("data\\cifar10\\cifar10_train_batch3.mat") 
    trainData3=dataTemp["gistFeat"]
    trainLabel3=npy.ravel(dataTemp["labels"])
    trainData=npy.concatenate((trainData1,trainData2,trainData3),axis=0)
    trainLabel=npy.concatenate((trainLabel1,trainLabel2,trainLabel3))
    
    return (trainData, trainLabel,testData,testLabel)
        
if __name__=="__main__":
    trainData,trainLabel,testData,testLabel=ReadCIFAR10Gist()
    print("Training SVM...")
    svmClf=SVC(50,"linear")
    svmClf.fit(trainData,trainLabel)
    predLabel=svmClf.predict(testData)
    confMat,oa=EvalAccuracy(predLabel,testLabel,range(10))
    print("Overall Accuracy="+str(oa))
        
        