import numpy as npy
from LoadData import ReadFvecs,ReadIvecs
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist,squareform 
from Utils import GetRetrivalMetric, GetKnnIdx
   
def PQTrain(data, lenSubVec,numSubCenter):
    (dataSize, dataDim)=data.shape
    if 0!=dataDim%lenSubVec:
        print "Cannot partition the feature space with the given segment number"
        return
    numSubVec=dataDim/lenSubVec
    centers=npy.zeros((numSubVec*numSubCenter,lenSubVec),dtype=npy.float32)
    distOfCenters=npy.zeros((numSubCenter,numSubCenter,numSubVec),dtype=npy.float32)
    objKmeans=KMeans(numSubCenter,'k-means++',3,100,0.001)
    for ii in range(numSubVec):
        print("PQ training. Processing "+str(ii)+"-th sub-vector")
        objKmeans.fit(data[:,ii*lenSubVec:(ii+1)*lenSubVec]) 
        centers[ii*numSubCenter:(ii+1)*numSubCenter,:]= objKmeans.cluster_centers_
        distOfCenters[:,:,ii]=squareform(pdist(objKmeans.cluster_centers_,metric="euclidean"))
    model={"centers":centers,"distOfCenters":distOfCenters}   
    return model

def PQEval(data,lenSubVec,numSubCenter,centersPQ):
    (dataSize, dataDim)=data.shape
    if 0!=dataDim%lenSubVec:
        print "Cannot partition the feature space with the given segment number"
        return
    numSubVec=dataDim/lenSubVec
    codePQ=-npy.ones((dataSize, numSubVec),dtype=npy.int32)
    objKmeans=KMeans(numSubCenter)
    if (centersPQ.shape[0]!=numSubVec*numSubCenter 
        or centersPQ.shape[1]!=lenSubVec):
        print "PQ model dimension is not compatible with input data"
        return
    for ii in range(numSubVec):
        objKmeans.cluster_centers_=centersPQ[ii*numSubCenter:(ii+1)*numSubCenter,:]
        codePQ[:,ii]=objKmeans.predict(data[:,ii*lenSubVec:(ii+1)*lenSubVec])
    return codePQ
    
def PQQuery(queryCode, baseCode, numSubCenter, modelPQ, k=5):
    if queryCode.shape[1]!=baseCode.shape[1]:
        print "Quary and Base codes are not with the same length"
        return
    nQuery=queryCode.shape[0]
    kRetr=npy.min((k,baseCode.shape[0]))
    distOfCenters=modelPQ["distOfCenters"]    
    knnIdx=-npy.ones((nQuery,kRetr),dtype=npy.int32) 
    distCodePair=npy.zeros(baseCode.shape, dtype=npy.float32)
    for ii in range(nQuery):
        distCodePair=distCodePair*0
        for jj in range(queryCode.shape[1]):
            distCodePair[:,jj]=distOfCenters[queryCode[ii,jj],baseCode[:,jj],jj]            
            idxSort=npy.argsort(npy.sum(npy.square(distCodePair),axis=1))
            knnIdx[ii,:]=idxSort[:kRetr]
    return knnIdx
            

if __name__=="__main__":
    dataPath="E:\\DevProj\\Datasets\\SIFT1M\\siftsmall"
    trainData=ReadFvecs(dataPath,"siftsmall_learn.fvecs")
    trainData=trainData.astype(npy.float32)
    lenSubVec=8
    numSubCenter=256
    
    modelPQ=PQTrain(trainData,lenSubVec,numSubCenter)
    
    queryData=ReadFvecs(dataPath,"siftsmall_query.fvecs")
    baseData=ReadFvecs(dataPath,"siftsmall_base.fvecs")
    idxGt=ReadIvecs(dataPath,"siftsmall_groundtruth.ivecs")    
    queryData=queryData.astype(npy.float32)
    baseData=baseData.astype(npy.float32)
    idxKnnGt=GetKnnIdx(queryData,baseData,100)
    
    queryCode=PQEval(queryData,lenSubVec,numSubCenter,modelPQ["centers"])
    baseCode=PQEval(baseData,lenSubVec,numSubCenter,modelPQ["centers"])
    
    idxKnnPred=PQQuery(queryCode, baseCode, numSubCenter, modelPQ, 100)
    retrivMetric=GetRetrivalMetric(idxKnnGt, idxKnnPred, 100, 1000010)
    print retrivMetric
    
    
    