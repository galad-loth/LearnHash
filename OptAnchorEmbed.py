import numpy as npy
from sklearn.cluster import KMeans
from LoadData import ReadFvecs
import Utils
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

    
def OptAnchorEmbed(data, paras):
    data0=data.T
    r=paras["rOptEmbed"]
    tau=paras["tau"]
    nbit=paras["nbit"]
    matTmp=npy.random.randn(nbit,nbit)
    matTmp=npy.dot(matTmp,matTmp.T)
    eigVals, eigVecs=npy.linalg.eig(matTmp)
    matR=eigVecs
    matEye=npy.eye(nbit)
    for kk in range(50):
        datar=npy.dot(matR, data0)
        grad=-r*npy.sign(datar)*npy.power(npy.abs(datar)+1e-6, r-1)
        grad=npy.dot(grad, data0.T)
        gradAd=npy.dot(grad, matR.T)-npy.dot(matR, grad.T)
        mat1=npy.linalg.inv(matEye+gradAd*tau/2)
        mat2=matEye-gradAd*tau/2        
        matR=npy.dot(mat1, npy.dot(mat2,matR))
    datar=npy.dot(matR, data0)
    datar=datar.astype(npy.float32)
    return datar.T
    
    
def TrainSHODE(data, paras):
    '''
    Train model for Spectral Hash with Optimized Achor Enbeding
    '''
    nAnchor=paras["nAnchor"]
    pNN=npy.minimum(paras["pNN"], nAnchor)
    nbit=paras["nbit"]
    datamean=npy.mean(data, axis=0)
    data=data-datamean
    anchorData=GetAnchorData(data, nAnchor, 1)
    adjMat=Utils.KernelRBF(anchorData, anchorData, paras["sigmaRBF"])
    idxSort=npy.argsort(adjMat, axis=1)
    for kk in range(nAnchor):
        adjMat[kk, idxSort[kk,:-pNN]]=0
    adjMat=(adjMat+adjMat.T)/2
    matD=npy.diag(npy.sum(adjMat, axis=1))
    lapMat=matD-adjMat
    matDSR=npy.diag(1/npy.sqrt(npy.diag(matD)))
    lapMat=npy.dot(matDSR, npy.dot(lapMat, matDSR))
    eigVals, eigVecs=npy.linalg.eig(lapMat)
    idxSort1=npy.argsort(eigVals)  
    optEmbed=OptAnchorEmbed(eigVecs[:,idxSort1[1:nbit+1]],paras)  
    # optEmbed=eigVecs[:,idxSort1[1:nbit+1]]
    # optEmbed=optEmbed.astype(npy.float32)
    modelSHODE={"datamean":datamean, "anchorData":anchorData, 
        "embedAnchorData": optEmbed, "nbit":nbit, "pNN":pNN}
    return modelSHODE
  
def EvalSHODE(data, modelSHODE):
    datan=data-modelSHODE["datamean"]
    anchorData=modelSHODE["anchorData"]
    embedAnchorData=modelSHODE["embedAnchorData"]
    pNN=modelSHODE["pNN"]
    nbit=modelSHODE["nbit"]
    idxPNN=Utils.GetKnnIdx(datan, anchorData, pNN)
    ndata=data.shape[0]    
    embedData=npy.zeros((ndata, nbit), dtype=npy.float32)
    for kk in npy.arange(ndata):
        dataTmp=datan[kk,:]
        dataPNN=anchorData[idxPNN[kk,:],:]
        embedPNN=embedAnchorData[idxPNN[kk,:],:]
        coef=npy.random.randn(pNN,1).astype(npy.float32)
        coef=npy.abs(coef)
        temp1=npy.dot(dataPNN,dataTmp[:,npy.newaxis])
        temp2=npy.dot(dataPNN,dataPNN.T)
        temp1p=(npy.abs(temp1)+temp1)/2
        temp1n=(npy.abs(temp1)-temp1)/2
        temp2p=(npy.abs(temp2)+temp2)/2
        temp2n=(npy.abs(temp2)-temp2)/2
        for tt in range(10):            
            temp3=(temp1p+npy.dot(temp2n, coef))
            temp4=(temp1n+npy.dot(temp2p, coef)+1e-6)   
            temp=temp3/temp4            
            coef=coef*npy.sqrt(temp)
        embedData[kk,:]=npy.dot(coef.T, embedPNN)
    binCode=(embedData>0).astype(npy.int8)
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
    
    paras={"nbit":64, "nAnchor":500, "pNN":50, "sigmaRBF":300.0,
        "tau": 0.1, "rOptEmbed":0.5}
    print "Training model..."
    modelSHODE=TrainSHODE(trainData, paras)
    
    print "Generating Code model..."
    queryCode=EvalSHODE(queryData,modelSHODE)
    baseCode=EvalSHODE(baseData,modelSHODE)    
    numNN=20
    idxKnn=Utils.GetKnnIdx(queryCode,baseCode,numNN, 1)   
    retrivMetric=Utils.GetRetrivalMetric(idxKnnGt, idxKnn, numNN, baseData.shape[0]+1)
    print retrivMetric
    
    
    