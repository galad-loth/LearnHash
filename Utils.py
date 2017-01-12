'''
Compute performance evaluation metrics for classification (confusion matrix,
user/producer/overall accuracy and kappa coeffcient) and retrival (precision, recall,
mAP)

Created by jlfeng, 2017-01-09

'''
import numpy as npy
import cv2

def GetGtKnnIdx(queryData,baseData,numNN):
    objMatcher=cv2.BFMatcher(cv2.NORM_L2)
    matches=objMatcher.knnMatch(queryData,baseData,k=numNN)
    idxKnn=npy.zeros((queryData.shape[0],numNN), dtype=npy.int32)
    for kk in range(queryData.shape[0]):
        for ll in range(numNN):
            idxKnn[kk][ll]=matches[kk][ll].trainIdx
    return idxKnn

def GetClassMetric(gtLabal, testLabel, numClass=-1, labelSet=npy.array([])):
    if numClass>0:
        labelSet=npy.arange(numClass)
    else:
        if labelSet.size()==0 or npy.min(labelSet)<0:
            return
        numClass=npy.max(labelSet)+1   

    confMat=npy.zeros((numClass,numClass),dtype=npy.float32)
    vecOnes=npy.ones(len(gtLabal))
    for ii in labelSet:
        for jj in labelSet:
            confMat[ii,jj]=npy.sum(vecOnes[npy.logical_and(testLabel==ii, gtLabal==jj)])
    
    ccn=npy.diagonal(confMat)
    oa=npy.sum(ccn)/npy.sum(confMat) 
    pa=ccn/npy.sum(confMat, axis=0) 
    ua=ccn/npy.sum(confMat, axis=1) 
    temp1=npy.sum(confMat)*npy.sum(ccn)-npy.sum(npy.sum(confMat,axis=1)*npy.sum(confMat,axis=0));
    temp2=npy.power(npy.sum(confMat),2)-npy.sum(npy.sum(confMat,axis=1)*npy.sum(confMat,axis=0));
    kappa=temp1/temp2
    confMat=confMat.astype(npy.int32)
    accMetric={"confMat":confMat, "oa":oa, "pa":pa, "ua":ua, "kappa": kappa}
    return accMetric     

def GetRetrivalMetric(gtIdx, testIdx, nnk, baseSize):
    testTimes=gtIdx.shape[0]
    nnr=gtIdx.shape[1]
    if nnk>testIdx.shape[1]:
        nnk=testIdx.shape[1]
    if gtIdx.shape[0]!=testIdx.shape[0] or gtIdx.shape[1]==0:
        return
    vecFlag=npy.zeros(baseSize, dtype=npy.float32)
    
    vecPrecision=npy.zeros(testTimes,dtype=npy.float32)
    vecRecall=npy.zeros(testTimes, dtype=npy.float32)
    vecAP=npy.zeros(testTimes, dtype=npy.float32)
    for i in npy.arange(testTimes):
        vecFlag.fill(0)
        vecFlag[gtIdx[i,:]]=1
        vecPrecision[i]=npy.sum(vecFlag[testIdx[i,:nnk]])/nnk
        vecRecall[i]=npy.sum(vecFlag[testIdx[i,:nnk]])/nnr
        if nnr==baseSize:
            temp=vecFlag[testIdx[i,:]]
            idx=temp==1
            temp=npy.cumsum(temp)/npy.arange(1, baseSize+1)
            vecAP[i]=npy.mean(temp[idx])
    precision=npy.mean(vecPrecision)
    recall=npy.mean(vecRecall)
    mAP=npy.mean(vecAP)
    retrivMetric={"precision":precision, "recall":recall, "mAP":mAP}
    return retrivMetric
    
    
if __name__=="__main__":
    gtIdx=npy.array([[0,1,2,3,4,5,6,7,8,9],[5,6,7,8,9, 0,1,2,3,4],[3,4,5,0,1,2, 6,7,8,9]])
    testIdx=gtIdx.copy()
    npy.transpose(npy.random.shuffle(testIdx.T))
    retrivMetric=GetRetrivalMetric(gtIdx[:,:8], testIdx, 4, 10)
    print gtIdx
    print testIdx
    print retrivMetric
    