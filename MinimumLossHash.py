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
    for i in npy.arange(kq):
        idx2=npy.random.permutation(kn)
        idx3=npy.random.permutation(kn*3)
        pairInfo[2*i*ks:(2*i+1)*ks,0]=1
        pairInfo[2*i*ks:(2*i+1)*ks,1]=idx1[i]
        pairInfo[2*i*ks:(2*i+1)*ks,2]=idxNN[i, 1+idx2[:ks]]
        pairInfo[(2*i+1)*ks:(2*i+2)*ks,0]=0
        pairInfo[(2*i+1)*ks:(2*i+2)*ks,1]=idx1[i]
        pairInfo[(2*i+1)*ks:(2*i+2)*ks,2]=idxNN[i, kn*2+idx3[:ks]]
    return pairInfo

def PreComputLoss(paraMLH):
    nbit=paraMLH["nbit"]
    loss=npy.zeros((2, nbit+1), dtype=npy.float32)
    m=npy.arange(0, nbit+1)
    rau=paraMLH["rau"]
    loss[0,:]=paraMLH["beta"]*npy.maximum(0, rau-m+1)
    loss[1,:]=npy.maximum(0, m-rau+1)
    loss=loss*paraMLH["epsilon"]
    return loss
    
def InitializeWeights(dimdata, nbit):
    w=npy.random.randn(dimdata,nbit)
    w=w.astype(npy.float32)
    w_norm=npy.sqrt(npy.sum(npy.power(w,2),axis=0))
    w=w/w_norm.reshape((1,w.shape[1]))
    return w
 
def LossAdjustInfer(loss, datap1, datap2,dataSign):
    ndata,nbit=datap1.shape
    data10=datap1-datap2
    data11= datap1+datap2
    delta=npy.maximum(data10, -data10) \
        -npy.maximum(data11,-data11)
    idxSort=npy.argsort(delta, axis=1)
    
    bs=npy.ones(datap1.shape,dtype=npy.float32)
    bd=npy.ones(datap1.shape,dtype=npy.float32)
    bs[data11<0]=-1
    bd[data10<0]=-1
    
    b1res=bs.copy()
    b2res=bs.copy()
    
    for n in range(ndata):
        maxLoss=loss[dataSign[n],0]
        mopt=0
        for m in range(1,nbit+1):
            tempLoss=npy.sum(delta[n,idxSort[n,-1:-m-1:-1]])  \
                     +loss[dataSign[n],m]
            if tempLoss>maxLoss:
                maxLoss=tempLoss
                mopt=m
        if mopt>0:
            b1res[n,idxSort[n,-1:-mopt-1:-1]]=bd[n,idxSort[n,-1:-mopt-1:-1]]
            b2res[n,idxSort[n,-1:-mopt-1:-1]]=-bd[n,idxSort[n,-1:-mopt-1:-1]]      
    return  b1res, b2res
    
    
def GetGrad(data, pairInfo,w, preLoss, paras):
    batchSize=pairInfo.shape[0]
    data1=data[pairInfo[:,1],:]
    data2=data[pairInfo[:,2],:]
    dataSign=pairInfo[:,0]
    datap1=npy.dot(data1,w)
    datap2=npy.dot(data2,w)
    datab1=npy.ones(datap1.shape,dtype=npy.float32)
    datab2=npy.ones(datap2.shape,dtype=npy.float32)
    datab1[datap1<0]=-1
    datab2[datap2<0]=-1
    datab1a, datab2a=LossAdjustInfer(preLoss, datap1, datap2,dataSign)
    dw1=npy.dot(data1.T, datab1)+npy.dot(data2.T, datab2)
    dw2=npy.dot(data1.T, datab1a)+npy.dot(data2.T, datab2a)
    dw=(dw2-dw1)/batchSize
    return dw/pairInfo.shape[0]   

def GetLoss(data, pairInfo, w, paras):
    rau=paraMLH["rau"]
    ndata=pairInfo.shape[0]
    data1=data[pairInfo[:,1],:]
    data2=data[pairInfo[:,2],:]
    dataSign=pairInfo[:,0]
    datap1=npy.dot(data1,w)
    datap2=npy.dot(data2,w)
    datab1=npy.ones(datap1.shape,dtype=npy.int32)
    datab2=npy.ones(datap2.shape,dtype=npy.int32)
    datab1[datap1<0]=-1
    datab2[datap2<0]=-1
    dist=npy.sum(datab1!=datab2, axis=1)
    loss=npy.zeros(ndata, dtype=npy.float32)
    loss[dataSign==1]=npy.maximum(0,dist[dataSign==1]-rau+1)
    loss[dataSign==0]=npy.maximum(0,rau-dist[dataSign==0]+1)
    return npy.sum(loss)
    
    
    
def TrainMLH(data, pairInfoTrain, pairInfoVal, preLoss, paras):
    ndata, dimdata=data.shape
    datamean=npy.mean(data, axis=0)
    data=data-datamean
    w0=InitializeWeights(dimdata, paras["nbit"])
    w=w0.copy()
    batchSize=paras["batchSize"]
    lr=paras["lr"]
    lr_decay_step=paras["lr_decay_step"]
    lr_decay=paras["lr_decay"]
    numBatch=pairInfoTrain.shape[0]/batchSize
    numIter=0
    for tau in range(paras["nepoch"]):  
        print "Training MLH. epoch "+str(tau)
        npy.random.shuffle(pairInfoTrain)
        for k in range(numBatch):
            numIter+=1
            idxStart=k*batchSize
            dw=GetGrad(data, pairInfoTrain[idxStart:idxStart+batchSize,:],
                       w, preLoss, paras)
            l2Reg=paras["wd"]*w
            w=w-lr*dw-l2Reg
            w_norm=npy.sqrt(npy.sum(w*w,axis=0))
            w=w/w_norm.reshape((1,w.shape[1]))
            if numIter%lr_decay_step==0:
                lr=lr*lr_decay
            if  numIter%10==0:
                loss=GetLoss(data,pairInfoVal, w, paras)
                print "Epoch:{}, iteration:{}, loss:{}".format(tau,numIter,loss)
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

    pairInfo=GetTrainPairs(trainData, 8000, 100, 40)
    npy.random.shuffle(pairInfo)
    pairInfoTrain=pairInfo[0:600000,:]
    pairInfoVal=pairInfo[600000:,:]

    paraMLH={"nbit":64, "beta": 0.8, "rau":15,
             "lr": 0.01, "lr_decay_step":1000, "lr_decay":0.9,
             "wd":0.01,"epsilon":5, 
             "nepoch":1, "batchSize":50}
    preLoss=PreComputLoss(paraMLH)
    
    modelMLH=TrainMLH(trainData, pairInfoTrain, pairInfoVal, preLoss, paraMLH)
    
    queryCode=EvalMLH(queryData,modelMLH)
    baseCode=EvalMLH(baseData,modelMLH)   
    
    numNN=20
    idxKnn=Utils.GetKnnIdx(queryCode,baseCode,numNN, 1)    
    retrivMetric=Utils.GetRetrivalMetric(idxKnnGt, idxKnn, numNN, baseData.shape[0]+1)
    print retrivMetric
    
    
    
    
    
    
    
    
    