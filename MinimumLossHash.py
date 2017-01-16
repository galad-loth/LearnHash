import numpy as npy
from scipy import linalg
from LoadData import ReadFvecs
import Utils
import pdb

def GetTrainPairs(data, kq, kn, ks):
    ndata=data.shape[0]
    idx1=npy.random.permutation(ndata)
    dataQuery=data[idx1[:kq],:]
    idxNN=Utils.GetKnnIdx(dataQuery, data, kn*5)
    pairInfo=npy.zeros((kq*ks*2, 3))
    idx2=npy.random.permutation(kn)
    idx3=npy.random.permutation(kn*3)
    for i in npy.arange(kq):
        pairInfo[2*i*ks:(2*i+1)*ks,0]=1
        pairInfo[2*i*ks:(2*i+1)*ks,1]=idx1[i]
        pairInfo[2*i*ks:(2*i+1)*ks,2]=idxNN[i, 1+idx2[:ks]]
        pairInfo[(2*i+1)*ks:(2*i+2)*ks,0]=-1
        pairInfo[(2*i+1)*ks:(2*i+2)*ks,1]=idx1[i]
        pairInfo[(2*i+1)*ks:(2*i+2)*ks,2]=idxNN[i, kn*2+idx2[:ks]]
    npy.random.shuffle(pairInfo)
    return pairInfo

def TrainMLH(data, pairInfo):
    #pairInfo=GetTrainPairs(trainData, 1000, 100, 50)
    pass



if __name__=="__main__":
    dataPath="E:\\DevProj\\Datasets\\SIFT1M\\siftsmall"
    trainData=ReadFvecs(dataPath,"siftsmall_learn.fvecs")
    trainData=trainData.astype(npy.float32)
    queryData=ReadFvecs(dataPath,"siftsmall_query.fvecs")
    baseData=ReadFvecs(dataPath,"siftsmall_base.fvecs")
    queryData=queryData.astype(npy.float32)
    baseData=baseData.astype(npy.float32) 
    idxKnnGt=Utils.GetKnnIdx(queryData,baseData,100,0)    
    
    paraMLH={"nbit":64, "beta": 0.5, "eta": 0.1, "rau":5,
        "nepoch", 50, "batchsize":1000}
    TrainMLH(trainData, paraMLH)