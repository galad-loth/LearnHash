import numpy as npy
from sklearn.decomposition import PCA  
from scipy.linalg import svd
from LoadData import ReadFvecs
import cv2
from Utils import GetRetrivalMetric , GetKnnIdx, GetCompactCode

def ITQtrain(data,nbit, niter):
    data_mean=npy.mean(data,axis=0)
    data=data-data_mean
    objPCA=PCA(copy=True,n_components=nbit, whiten=False)
    dataTrans=objPCA.fit_transform(data)
    codeITQ=npy.ones(dataTrans.shape, dtype=npy.float32)
    codeITQ[dataTrans<0]=-1
    for tau in range(niter):
        dataTemp1=npy.dot(codeITQ.T,dataTrans)
        matL,sig, matR=svd(dataTemp1)
        matRot=npy.dot(matR.T,matL.T)
        dataTemp2=dataTrans.dot(matRot)
        codeITQ=npy.ones(dataTrans.shape, dtype=npy.float32)
        codeITQ[dataTemp2<0]=-1
    modelITQ={"mu":data_mean,"objPCA":objPCA, "matRot":matRot} 
    return modelITQ       


def ITQeval(data, modelITQ):
    data=data-modelITQ["mu"]
    objPCA=modelITQ["objPCA"]
    dataTrans=objPCA.transform(data)
    matRot=modelITQ["matRot"]
    dataTrans=dataTrans.dot(matRot)
    binCode=npy.zeros(dataTrans.shape, dtype=npy.int8)
    binCode[dataTrans>0]=1
    print binCode
    codeITQ=GetCompactCode(binCode)
    return codeITQ

if __name__=="__main__":
    dataPath="E:\\DevProj\\Datasets\\SIFT1M\\siftsmall"
    trainData=ReadFvecs(dataPath,"siftsmall_learn.fvecs")
    trainData=trainData.astype(npy.float32)
    queryData=ReadFvecs(dataPath,"siftsmall_query.fvecs")
    baseData=ReadFvecs(dataPath,"siftsmall_base.fvecs")
    queryData=queryData.astype(npy.float32)
    baseData=baseData.astype(npy.float32)
    idxKnnGt=GetKnnIdx(queryData,baseData,100, 0)

    modelITQ=ITQtrain(trainData,80,10)
    queryCode=ITQeval(queryData,modelITQ)
    baseCode=ITQeval(baseData,modelITQ)   
    
    numNN=30
    idxKnn=GetKnnIdx(queryCode,baseCode,numNN, 1)    
    retrivMetric=GetRetrivalMetric(idxKnnGt, idxKnn, numNN, baseData.shape[0]+1)
    print retrivMetric


