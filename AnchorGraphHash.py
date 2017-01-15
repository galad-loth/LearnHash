import numpy as npy
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from LoadData import ReadFvecs
from Utils import KernelRBF, GetRetrivalMetric , GetKnnIdx, GetCompactCode
import pdb


def GetAnchorData(data, nAnchor, mode=0):
    '''
    Generate anchor data by random sampling or k-means
    '''
    if mode==0: #random sampling
        nAnchor=npy.min([data.shape[0], nAnchor])
        idxRand=npy.arange(data.shape[0])
        npy.random.shuffle(idxRand)
        anchorData=data[idxRand[:nAnchor],:]
    elif mode==1: #k-means
        nAnchor=npy.min([data.shape[0]/2, nAnchor])
        objKmeans=KMeans(nAnchor,'k-means++',3,500,0.001)
        objKmeans.fit(data)
        anchorData=objKmeans.cluster_centers_
    return anchorData.astype(npy.float32)
        
def TrainAGH(data, anchorData, kNNAnchor, sigmaRBF, nbit):
    ndata=data.shape[0]
    kNNAnchor=npy.min([kNNAnchor, anchorData.shape[0]])
    nbit=npy.min([nbit, anchorData.shape[0]])
    idxNNAnchor=GetKnnIdx(data, anchorData, kNNAnchor)
    distToAnchor=KernelRBF(data, anchorData, sigmaRBF)
    adjMat=npy.zeros(distToAnchor.shape, dtype=npy.float32)
    for k in range(ndata):
        adjMat[k, idxNNAnchor[k,:]]=distToAnchor[k, idxNNAnchor[k,:]]
    adjSum=npy.sum(adjMat, axis=1)
    adjMat=adjMat/adjSum[:,npy.newaxis]
    matLambda=npy.sum(adjMat, axis=0)
    matLambda=npy.diag(1/npy.sqrt(matLambda+1e-10))
    covAdjMat=npy.dot(adjMat.T, adjMat)
    covAdjMat=npy.dot(matLambda, npy.dot(covAdjMat, matLambda))
    eigVals, eigVecs=npy.linalg.eig(covAdjMat)
    idxSort=npy.argsort(eigVals)
    matW=eigVecs[:, idxSort[-1:-nbit-1:-1]]
    projData=npy.dot(adjMat, matW)
    code=projData>0  
    modelAGH={"anchorData":anchorData, "sigmaRBF":sigmaRBF,  \
    "kNNAnchor":kNNAnchor, "nbit":nbit, "matW":matW}
    return modelAGH, code
    
def EvalAGH(data, modelAGH):
    anchorData=modelAGH["anchorData"]
    sigmaRBF=modelAGH["sigmaRBF"]
    kNNAnchor=modelAGH["kNNAnchor"]
    nbit=modelAGH["nbit"]
    matW=modelAGH["matW"]
    idxNNAnchor=GetKnnIdx(data, anchorData, kNNAnchor)
    distToAnchor=KernelRBF(data, anchorData, sigmaRBF)
    adjMat=npy.zeros(distToAnchor.shape, dtype=npy.float32)
    for k in range(data.shape[0]):
        adjMat[k, idxNNAnchor[k,:]]=distToAnchor[k, idxNNAnchor[k,:]]
    adjSum=npy.sum(adjMat, axis=1)
    adjMat=adjMat/adjSum[:,npy.newaxis]
    projData=npy.dot(adjMat, matW)
    binCode=(projData>0).astype(npy.int8)
    compactCode=GetCompactCode(binCode)
    return compactCode
    
    
if __name__=="__main__":
    dataPath="E:\\DevProj\\Datasets\\SIFT1M\\siftsmall"
    trainData=ReadFvecs(dataPath,"siftsmall_learn.fvecs")
    trainData=trainData.astype(npy.float32)
    queryData=ReadFvecs(dataPath,"siftsmall_query.fvecs")
    baseData=ReadFvecs(dataPath,"siftsmall_base.fvecs")
    queryData=queryData.astype(npy.float32)
    baseData=baseData.astype(npy.float32)
    idxKnnGt=GetKnnIdx(queryData,baseData,100,0)
    
    nAnchors=500
    kNNAnchor=500
    sigmaRBF=300.0
    nbit=64
    anchorData=GetAnchorData(trainData, nAnchors, 1)
    modelAGH,code=TrainAGH(trainData, anchorData, kNNAnchor, sigmaRBF, nbit)
        
    baseCode=EvalAGH(baseData,modelAGH)
    queryCode=EvalAGH(queryData,modelAGH)
    numNN=20
    idxKnn=GetKnnIdx(queryCode,baseCode,numNN, 1)   
    retrivMetric=GetRetrivalMetric(idxKnnGt, idxKnn, numNN, baseData.shape[0]+1)
    print retrivMetric