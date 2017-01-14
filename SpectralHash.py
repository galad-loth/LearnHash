import numpy as npy
from sklearn.decomposition import PCA  
from LoadData import ReadFvecs
import cv2
from Utils import GetRetrivalMetric , GetKnnIdx, GetCompactCode

def TrainSH(data, nbit):
    dataSize, dataDim=data.shape
    npca=npy.min([dataDim, nbit])
    objPCA=PCA(copy=True,n_components=npca, whiten=False)
    dataTrans=objPCA.fit_transform(data)
    eigVals=npy.zeros(npca*nbit, dtype=npy.float32)
    limitVals=npy.zeros((npca,2), dtype=npy.float32)
    idxPc=npy.zeros(npca*nbit, dtype=npy.int32)
    idxEigs=npy.zeros(npca*nbit, dtype=npy.int32)
    nFract05=dataSize*5/100
    nFract95=dataSize-nFract05
    r=npy.arange(1, nbit+1)
    for k in range(npca):
        dataSort=npy.sort(dataTrans[:,k])
        limitVals[k,0]=dataSort[nFract05]
        limitVals[k,1]=dataSort[nFract95]
        eigVals[k*nbit:(k+1)*nbit]=npy.power(r/(limitVals[k,1]-limitVals[k,0]),2)
        idxPc[k*nbit:(k+1)*nbit]=k
        idxEigs[k*nbit:(k+1)*nbit]=r
    idxSort=npy.argsort(eigVals)
    eigVals=eigVals[idxSort[:nbit]]
    idxPc=idxPc[idxSort[:nbit]]
    idxEigs=idxEigs[idxSort[:nbit]]
    modelSH={"nbit":nbit, "objPCA": objPCA, "idxPc":idxPc, "idxEigs": idxEigs, "limitVals":limitVals}
    return modelSH
        
def EvalSH(data, modelSH):
    nbit=modelSH["nbit"]
    objPCA=modelSH["objPCA"]
    idxPc=modelSH["idxPc"]
    idxEigs=modelSH["idxEigs"]
    limitVals=modelSH["limitVals"]
    dataTrans=objPCA.transform(data)
    binCode=npy.zeros((dataTrans.shape[0], nbit), dtype=npy.int8)
    for k in range(nbit):
        ipc=idxPc[k]
        f=idxEigs[k]*npy.pi/(limitVals[ipc,1]-limitVals[ipc,0])
        phi=npy.sin(npy.pi/2+f*(dataTrans[:,ipc]-limitVals[ipc,0]))
        print str(idxPc[k])+"/"+str(idxEigs[k])+"/"+str(npy.sum(phi>0))
        binCode[:,k]=phi>0
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
    
    nbit=64
    modelSH=TrainSH(trainData, nbit)
    baseCode=EvalSH(baseData,modelSH)
    queryCode=EvalSH(queryData,modelSH)
    numNN=5
    idxKnn=GetKnnIdx(queryCode,baseCode,numNN, 1)   
    retrivMetric=GetRetrivalMetric(idxKnnGt, idxKnn, numNN, baseData.shape[0]+1)
    print retrivMetric
    
    