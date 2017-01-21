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
    pairInfo=npy.zeros((kq*ks*2, 3),dtype=npy.int32)
    idx2=npy.random.permutation(kn)
    idx3=npy.random.permutation(kn*3)
    for i in npy.arange(kq):
        pairInfo[2*i*ks:(2*i+1)*ks,0]=1
        pairInfo[2*i*ks:(2*i+1)*ks,1]=idx1[i]
        pairInfo[2*i*ks:(2*i+1)*ks,2]=idxNN[i, 1+idx2[:ks]]
        pairInfo[(2*i+1)*ks:(2*i+2)*ks,0]=0
        pairInfo[(2*i+1)*ks:(2*i+2)*ks,1]=idx1[i]
        pairInfo[(2*i+1)*ks:(2*i+2)*ks,2]=idxNN[i, kn*2+idx2[:ks]]
    return pairInfo

def PreComputLoss(paraMLH):
    nbit=paraMLH["nbit"]
    loss=npy.zeros((2, nbit+1), dtype=npy.float32)
    m=npy.arange(0, nbit+1)
    rau=paraMLH["rau"]
    loss[0,:]=paraMLH["beta"]*npy.maximum(0, rau-m+1)
    loss[1,:]=npy.maximum(0, m-rau+1)
    return loss
    
def InitializeWeights(dimdata, nbit):
    w=npy.random.randn(dimdata,nbit)
    w=w.astype(npy.float32)
    w=w/npy.sqrt(npy.sum(npy.power(w,2),axis=0))
    return w
 
def LossAdjustInfer(loss, datap1, datap2,dataSign):
    ndata, nbit=datap1.shape
    matZeros=npy.zeros(datap1.shape, dtype=npy.float32)
    delta=npy.maximum(datap1, datap2) \
        -npy.maximum(matZeros, datap1+datap2)
    idxSort=npy.argsort(delta, axis=1)
    maxVal=-1e10*npy.ones(ndata)
    b1res=npy.zeros(datap1.shape,dtype=npy.float32)
    b2res=npy.zeros(datap2.shape,dtype=npy.float32)
    for m in range(nbit+1):
        valTemp1=loss[dataSign, m]
        b1=npy.zeros(datap1.shape,dtype=npy.float32)
        b2=npy.zeros(datap2.shape,dtype=npy.float32)
        for n in range(ndata):
            idx1=npy.zeros(nbit, dtype=npy.bool)
            idx1[idxSort[n,-1:-m-1:-1]]=True
            b1[n,idx1]=datap1[n, idx1]>=datap2[n,idx1]
            b2[n,idx1]=datap1[n, idx1]<datap2[n,idx1]
            b1[n,~idx1]=(datap1[n, ~idx1]+datap2[n,~idx1])>0
            b2[n,~idx1]=b1[n,~idx1]
        valTemp2=npy.sum(datap1*b1+datap2*b2,axis=1)
        valTemp=valTemp1+valTemp2
        idx=valTemp>maxVal
        maxVal[idx]=valTemp[idx]
        b1res[idx,:]=b1[idx,:]
        b2res[idx,:]=b2[idx,:]
    return b1res,b2res  
    
def GetGrad(data, pairInfo,w, preLoss, paras):
    data1=data[pairInfo[:,1],:]
    data2=data[pairInfo[:,2],:]
    dataSign=pairInfo[:,0]
    datap1=npy.dot(data1,w)
    datap2=npy.dot(data2,w)
    datab1=npy.zeros(datap1.shape,dtype=npy.float32)
    datab2=npy.zeros(datap2.shape,dtype=npy.float32)
    datab1[datap1>0]=1
    datab2[datap2>0]=1
    datab1a, datab2a=LossAdjustInfer(preLoss, datap1, datap2,dataSign)
    dw=npy.dot(data1.T, datab1)+npy.dot(data2.T, datab2) \
        - npy.dot(data1.T, datab1a)-npy.dot(data2.T, datab2a)
    return dw/pairInfo.shape[0]
    
    
    
def TrainMLH(data, pairInfo, preLoss, paras):
    ndata, dimdata=data.shape
    datamean=npy.mean(data, axis=0)
    data=data-datamean
    w0=InitializeWeights(dimdata, paras["nbit"])
    w=w0.copy()
    batchSize=paras["batchSize"]
    miniBatchSize=paras["miniBatchSize"]
    nMiniBatch=batchSize/miniBatchSize
    for tau in range(paras["nepoch"]):  
        print "Training MLH. epoch "+str(tau)
        npy.random.shuffle(pairInfo)
        for k in range(nMiniBatch):
            dw=GetGrad(data, pairInfo[k*miniBatchSize:(k+1)*miniBatchSize,:], w, preLoss, paras)
            w=w+paras["eta"]*dw
            w=w/npy.sqrt(npy.sum(npy.power(w,2),axis=0))
    modelMLH={"datamean":datamean, "w":w, "w0":w0}
    return modelMLH
        
def EvalMLH(data, modelMLH):
    datac=data-modelMLH["datamean"]
    datap=npy.dot(datac, modelMLH["w"])
    binCode=npy.zeros(datap.shape, dtype=npy.int8)
    binCode[datap>0]=1
    codeMLH=Utils.GetCompactCode(binCode)
    return codeMLH

if __name__=="__main__":
    dataPath="E:\\DevProj\\Datasets\\SIFT1M\\siftsmall"
    trainData=ReadFvecs(dataPath,"siftsmall_learn.fvecs")
    trainData=trainData.astype(npy.float32)
    queryData=ReadFvecs(dataPath,"siftsmall_query.fvecs")
    baseData=ReadFvecs(dataPath,"siftsmall_base.fvecs")
    queryData=queryData.astype(npy.float32)
    baseData=baseData.astype(npy.float32) 
    idxKnnGt=Utils.GetKnnIdx(queryData,baseData,100,0)     

    pairInfo=GetTrainPairs(trainData, 5000, 100, 50)
    paraMLH={"nbit":64, "beta": 0.5, "eta": 0.1, "rau":10, "epsilon":150,
        "nepoch":50, "batchSize":5000, "miniBatchSize": 5}
    preLoss=PreComputLoss(paraMLH)
    
    modelMLH=TrainMLH(trainData, pairInfo, preLoss, paraMLH)
    
    queryCode=EvalMLH(queryData,modelMLH)
    baseCode=EvalMLH(baseData,modelMLH)   
    
    numNN=20
    idxKnn=Utils.GetKnnIdx(queryCode,baseCode,numNN, 1)    
    retrivMetric=Utils.GetRetrivalMetric(idxKnnGt, idxKnn, numNN, baseData.shape[0]+1)
    print retrivMetric
    
    
    
    
    
    
    
    
    