import numpy as npy
from scipy import linalg
from LoadData import ReadFvecs
import Utils
import pdb

def GetLabeledInfo(data, nDataL):
    ndata=data.shape[0]
    kn2=nDataL/3
    kn3=2*nDataL/3    
    idxLabelData=npy.arange(ndata)
    npy.random.shuffle(idxLabelData)
    idxLabelData=idxLabelData[:nDataL]
    dataL=data[idxLabelData,:]
    adjMat=npy.zeros((nDataL, nDataL), dtype=npy.float32)
    idxKnnGt=Utils.GetKnnIdx(dataL,dataL,nDataL,0)
    for k in range(nDataL):
        adjMat[k, idxKnnGt[k, 1:kn2+1]]=1
        adjMat[k, idxKnnGt[k, kn3:]]=-1
    adjMat=(adjMat+adjMat.T)/2
    return dataL, adjMat
    
def TrainSSHOPL(data, dataL, adjMat, nbit, eta):
    datamean=npy.mean(data, axis=0)
    data=data-datamean
    dataL=dataL-datamean
    covMatU=npy.dot(data.T, data)
    covMatL=npy.dot(dataL.T, npy.dot(adjMat, dataL))
    covMat=eta*covMatU+(1-eta)*covMatL
    eigVals, eigVecs=npy.linalg.eig(covMat)
    idxSort=npy.argsort(npy.abs(eigVals))
    nbit=npy.min([nbit,data.shape[1]])
    projMat=eigVecs[:,idxSort[-1:-nbit-1:-1]]
    modelSSH={"datamean":datamean, "projMat":projMat}
    return modelSSH

def TrainSSHNOPL(data, dataL, adjMat, nbit, eta, rau):
    datamean=npy.mean(data, axis=0)
    data=data-datamean
    dataL=dataL-datamean
    covMatU=npy.dot(data.T, data)
    covMatL=npy.dot(dataL.T, npy.dot(adjMat, dataL))
    covMat=eta*covMatU+(1-eta)*covMatL    
    eigVals, eigVecs=npy.linalg.eig(covMat)    
    idxSort=npy.argsort(npy.abs(eigVals))
    
    rau=npy.max([rau, npy.max([0, -npy.abs(eigVals[idxSort[0]])])])
    covMatReg=(covMat+rau*npy.eye(covMat.shape[0]))
    matL= linalg.cholesky(covMatReg, lower=False)
    nbit=npy.min([nbit,data.shape[1]])
    projMat=npy.dot(matL, eigVecs[:,idxSort[-1:-nbit-1:-1]])
    modelSSH={"datamean":datamean, "projMat":projMat}
    return modelSSH    
    
def TrainSSHSPL(data, dataL, adjMat, nbit, eta, alpha):
    datamean=npy.mean(data, axis=0)
    data=data-datamean
    dataL=dataL-datamean     
    adjMat0=adjMat.copy()
    projMat=npy.zeros((data.shape[1], nbit), dtype=npy.float32)
    for k in range(nbit):
        covMatU=npy.dot(data.T, data)   
        covMatL=npy.dot(dataL.T, npy.dot(adjMat0, dataL))
        covMat=eta*covMatU+(1-eta)*covMatL
        eigVals, eigVecs=npy.linalg.eig(covMat)
        idxSort=npy.argsort(npy.abs(eigVals))
        eigVec1=eigVecs[:,idxSort[-1]]
        projDataL=npy.dot(dataL, eigVec1)
        projData=npy.dot(data, eigVec1)
        data=data-npy.dot(projData[:,npy.newaxis], eigVec1[npy.newaxis,:])
        dataL=dataL-npy.dot(projDataL[:,npy.newaxis], eigVec1[npy.newaxis,:])
        adjMatDelt=npy.dot(projDataL[:, npy.newaxis], projDataL[npy.newaxis,:])
        adjMatDelt=adjMatDelt/npy.max(npy.abs(adjMatDelt))
        idxSameSign=((adjMatDelt*adjMat0)>=0)
        adjMatDelt[idxSameSign]=0
        adjMat0=adjMat0-alpha*adjMatDelt
        adjMat0=(adjMat0+adjMat0.T)/2
        projMat[:,k]=eigVec1   

    modelSSH={"datamean":datamean, "projMat":projMat}
    return modelSSH      
        
    
def EvalSSH(data, modelSSH):
    data=data-modelSSH["datamean"]
    projData=npy.dot(data, modelSSH["projMat"])
    binCode=(projData>npy.mean(projData,axis=0)).astype(npy.int8)
    compactCode=Utils.GetCompactCode(binCode)
    return compactCode      
    
    
if __name__=="__main__":
    dataPath="E:\\DevProj\\Datasets\\SIFT1M\\siftsmall"
    trainData=ReadFvecs(dataPath,"siftsmall_learn.fvecs")
    trainData=trainData.astype(npy.float32)
    queryData=ReadFvecs(dataPath,"siftsmall_query.fvecs")
    baseData=ReadFvecs(dataPath,"siftsmall_base.fvecs")
    queryData=queryData.astype(npy.float32)
    baseData=baseData.astype(npy.float32)
    idxKnnGt=Utils.GetKnnIdx(queryData,baseData,100,0)
    
    nDataL=2000
    dataL, adjMat=GetLabeledInfo(trainData, nDataL)
    eta=0.75
    nbit=64
    # modelSSH=TrainSSHOPL(trainData, dataL, adjMat, nbit, eta)
    alpha=0.5
    modelSSH=TrainSSHSPL(trainData, dataL, adjMat, nbit, eta, alpha)
    
    baseCode=EvalSSH(baseData,modelSSH)
    queryCode=EvalSSH(queryData,modelSSH)
    numNN=50
    idxKnn=Utils.GetKnnIdx(queryCode,baseCode,numNN, 1)   
    retrivMetric=Utils.GetRetrivalMetric(idxKnnGt, idxKnn, numNN, baseData.shape[0]+1)
    print retrivMetric