import numpy as npy
import struct
import os
from scipy import io as scio

def ReadFvecs(dataPath, dataFile, start=0, end=-1):
    filePath=os.path.join(dataPath,dataFile)
    with open(filePath,mode="rb") as fid:
        buf=fid.read(4)
        dimFeat=struct.unpack("i", buf[:4])
        numVecByte=(dimFeat[0]+1)*4
        
        fid.seek(0,2)
        numVec=fid.tell()/numVecByte
        if end<0:
            end=numVec
        if (start<0 or start>numVec or end>numVec or start>end):
            print("Start/End index is out of the data range")
        numDataEntry=(end-start)*(dimFeat[0]+1)
        numReadByte=numDataEntry*4
        fid.seek(start*numVecByte,0)
        buf=fid.read(numReadByte)
        data=npy.array(struct.unpack("f"*numDataEntry, buf[:numReadByte]))
    data=data.reshape((numVec, dimFeat[0]+1))    
    data=data[:,1:]
    return data
        
def ReadIvecs(dataPath, dataFile, start=0, end=-1):
    filePath=os.path.join(dataPath,dataFile)
    with open(filePath,mode="rb") as fid:
        buf=fid.read(4)
        dimFeat=struct.unpack("i", buf[:4])
        numVecByte=(dimFeat[0]+1)*4
        
        fid.seek(0,2)
        numVec=fid.tell()/numVecByte
        if end<0:
            end=numVec
        if (start<0 or start>numVec or end>numVec or start>end):
            print("Start/End index is out of the data range")
        numDataEntry=(end-start)*(dimFeat[0]+1)
        numReadByte=numDataEntry*4
        fid.seek(start*numVecByte,0)
        buf=fid.read(numReadByte)
        data=npy.array(struct.unpack("i"*numDataEntry, buf[:numReadByte]))
    data=data.reshape((numVec, dimFeat[0]+1))    
    data=data[:,1:]
    return data 

    
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
    dataTemp=scio.loadmat("data\\cifar10\\cifar10_train_batch4.mat") 
    trainData4=dataTemp["gistFeat"]
    trainLabel4=npy.ravel(dataTemp["labels"])
    dataTemp=scio.loadmat("data\\cifar10\\cifar10_train_batch5.mat") 
    trainData5=dataTemp["gistFeat"]
    trainLabel5=npy.ravel(dataTemp["labels"])
    
    trainData=npy.concatenate((trainData1,trainData2,trainData3,trainData4,trainData5),axis=0)
    trainLabel=npy.concatenate((trainLabel1,trainLabel2,trainLabel3,trainLabel4,trainLabel5))
    
    return (trainData, trainLabel,testData,testLabel)

if __name__=="__main__":
    dataPath="E:\\DevProj\\Datasets\\SIFT1M\\sift"
    dataFile="sift_groundtruth.ivecs"
    data=ReadIvecs(dataPath, dataFile)
