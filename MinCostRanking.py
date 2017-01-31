import numpy as npy
from sklearn.cluster import KMeans
from sklearn.linear_model import OrthogonalMatchingPursuit as OMP
from LoadData import ReadFvecs
import Utils
import pdb

def GetNeighborDims(data, paras):
    ndata, ndim=data.shape
    kND=paras["kND"]
    objOMP=OMP(n_nonzero_coefs=kND)
    idxDict=npy.ones(ndim, dtype=npy.bool)
    w=npy.zeros((ndim-1, ndim), dtype=npy.float32)
    for kk in range(ndim):
        idxDict.fill(True)
        idxDict[kk]=False
        objOMP.fit(data[:,idxDict], data[:,kk])
        w[:,kk]=objOMP.coef_.astype(npy.float32)
    return w    
    

def TrainLinearClf(data, labels, paras):    
    ndata,ndim=data.shape
    vecOnes=npy.ones((ndata,1),dtype=npy.float32)
    data1=npy.concatenate((data, vecOnes), axis=1)
    ndim=ndim+1
    w0=npy.random.randn(ndim,1).astype(npy.float32)
    w1=w0
    bitLoss=npy.sum(w0*w0)/2
    gamma=paras["gamma"]
    delta=paras["delta"]
    lamda=paras["lamda"]
    psuLabel=npy.zeros(ndata, dtype=npy.float32)
    for iter in range(paras["maxIterTrainClf"]):
        bitLoss=npy.sum(w0*w0)/2
        clfScore=npy.dot(data1, w0)
        grad=w0
        for nn in range(ndata):
            psuLabel.fill(1)
            psuLabel[labels!=labels[nn]]=delta
            scoreProd=(clfScore[:,0]*clfScore[nn,0])*psuLabel
            idx=scoreProd<1            
            grad1=-npy.sum(clfScore[idx,0]*psuLabel[idx])*data1[nn,:].T
            grad=grad+lamda*grad1[:,npy.newaxis]
            idx[:nn]=False
            bitLoss=bitLoss+lamda*npy.sum(1-scoreProd[idx])
        grad=grad/ndata/ndata
        w1=w0-gamma*grad       
        # print "iter="+str(iter)+", loss="+str(bitLoss)        
        if npy.sum(npy.abs(w1-w0))<1e-5:
            break
        w0=w1
        if iter%10==0:
            gamma=gamma*0.9
     
    bitLoss=npy.sum(w1*w1)/2
    clfScore=npy.dot(data1, w1)
    for nn in range(ndata):
        psuLabel.fill(1)
        psuLabel[labels!=labels[nn]]=delta
        scoreProd=(clfScore[:,0]*clfScore[nn,0])*psuLabel
        idx=scoreProd<1
        idx[:nn]=False
        bitLoss=bitLoss+lamda*npy.sum(1-scoreProd[idx])   
    return w1, bitLoss   
        
    

def TrainMCR(data, paras):
    ndata, ndim=data.shape
    kCenter=paras["kCenter"]
    mu=npy.sqrt(npy.mean(npy.power(data,2)))
    datan=data/mu
    ndatas=paras["nTrainSample"]
    nbit=paras["nbit"]
    if ndatas>0:
        idxRand=npy.arange(ndata)
        npy.random.shuffle(idxRand)
        ndata=npy.minimum(ndata, ndatas)
        datan=datan[idxRand[:ndata],:]
    # centers=GetAnchorData(datan, kCenter, 0)
    objKmeans=KMeans(kCenter,'k-means++',3,500,0.001)
    objKmeans.fit(datan)
    centers=objKmeans.cluster_centers_
    labels=objKmeans.labels_
    recND=GetNeighborDims(centers, paras)        
    
    lossRec=npy.zeros(ndim, dtype=npy.float32)
    wRec=[]
    for dd in range(ndim):
        idxND,=recND[:,dd].nonzero()        
        wd,lossd=TrainLinearClf(datan[:, idxND], labels, paras)
        lossRec[dd]=lossd
        wRec.append(wd)
        print str(dd)+"-th dim, loss="+str(lossd)   
 
    idxSort=npy.argsort(lossRec)
    modelMCR={"nbit": nbit, "mu":mu, "w": wRec, "recND":recND, "dimOrder":idxSort}
    return modelMCR
    
    
def EvalMCR(data, model):
    ndata, ndim=data.shape
    datan=data/model["mu"]
    recND=model["recND"]
    w=model["w"]
    nbit=model["nbit"]    
    dimOrder=model["dimOrder"]
    clfScore=npy.zeros((ndata, nbit), dtype=npy.float32)
    vecOnes=npy.ones((ndata,1),dtype=npy.float32)
    for dd in range(nbit):
        idxND,=recND[:,dimOrder[dd]].nonzero()        
        dataTmp=npy.concatenate((datan[:, idxND], vecOnes), axis=1)
        clfScore[:,dd]=npy.dot(dataTmp, w[dimOrder[dd]]).flatten()
    binCode=(clfScore>0).astype(npy.int8)
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
    
    paras={"nbit":64, "kCenter":50, "kND":8, "lamda":0.5, 
        "gamma": 0.8, "maxIterTrainClf":50, "delta":-0.85,
        "nTrainSample":5000}
    print "Training model..."
    modelMCR=TrainMCR(trainData, paras)    
    
    print "Generating Code model..."
    queryCode=EvalMCR(queryData,modelMCR)
    baseCode=EvalMCR(baseData,modelMCR)    
    numNN=20
    idxKnn=Utils.GetKnnIdx(queryCode,baseCode,numNN, 1)   
    retrivMetric=Utils.GetRetrivalMetric(idxKnnGt, idxKnn, numNN, baseData.shape[0]+1)
    print retrivMetric