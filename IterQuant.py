import numpy as npy
from sklearn.decomposition import PCA  
from scipy.linalg import svd
from LoadData import ReadFvecs
import cv2
from Utils import GetRetrivalMetric , GetGtKnnIdx

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
    if 0==dataTrans.shape[1]%8:
        codeByteNum=dataTrans.shape[1]/8
    else:
        codeByteNum=1+dataTrans.shape[1]/8
    codeITQ=npy.zeros((dataTrans.shape[0],codeByteNum),dtype=npy.uint8)
    for kk in range(dataTrans.shape[1]):
        idxByte=kk/8
        idxBit=kk%8
        codeITQ[dataTrans[:,kk]>0,idxByte]+=(1<<idxBit)
    return codeITQ

if __name__=="__main__":
    dataPath="E:\\DevProj\\Datasets\\SIFT1M\\siftsmall"
    trainData=ReadFvecs(dataPath,"siftsmall_learn.fvecs")
    trainData=trainData.astype(npy.float32)
    queryData=ReadFvecs(dataPath,"siftsmall_query.fvecs")
    baseData=ReadFvecs(dataPath,"siftsmall_base.fvecs")
    queryData=queryData.astype(npy.float32)
    baseData=baseData.astype(npy.float32)
    idxKnnGt=GetGtKnnIdx(queryData,baseData,100)

    modelITQ=ITQtrain(trainData,80,10)
    queryCode=ITQeval(queryData,modelITQ)
    baseCode=ITQeval(baseData,modelITQ)
    
    numNN=30
    objMatcher=cv2.BFMatcher(cv2.NORM_HAMMING)
    matches=objMatcher.knnMatch(queryCode,baseCode,k=numNN)
    idxKnn=npy.zeros((queryData.shape[0],numNN), dtype=npy.int32)
    for kk in range(queryData.shape[0]):
        for ll in range(numNN):
            idxKnn[kk][ll]=matches[kk][ll].trainIdx
    
    retrivMetric=GetRetrivalMetric(idxKnnGt, idxKnn, numNN, baseData.shape[0]+1)
    print retrivMetric


