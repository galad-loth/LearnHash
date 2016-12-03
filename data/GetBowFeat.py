import numpy as npy
import os
import cv2
from sklearn.cluster import KMeans
import cPickle

nr=32
nc=32
numPixel=1024

def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict
    
def GetPatchIdxCIFAR10(pr,pc,patchSize):
    pcPatch,prPatch=npy.meshgrid(npy.arange(pc,pc+patchSize),npy.arange(pr,pr+patchSize))
    idxPatchPixel=prPatch*32+pcPatch
    idxPatchPixel=idxPatchPixel.flatten()
    idxPixels=npy.hstack((idxPatchPixel,idxPatchPixel+numPixel,idxPatchPixel+numPixel*2))
    return idxPixels
    

def TrainCodingModel(dataset,datapath,param):
    print("Training dictionary for patch coding...")
    if dataset=="cifar10":
        patchSize=param
        numPatch=20000
        numAtom=200
        patchData=npy.zeros((numPatch,patchSize*patchSize*3),dtype=npy.float32)
        datafile=os.path.join(datapath,"data_batch_1")
        dataDict=unpickle(datafile)
        imgData=dataDict['data']
        numImg=imgData.shape[0]
        imgIdxArray=npy.random.randint(numImg,size=numPatch)
        prArray=npy.random.randint(0,nr-1-patchSize,size=numPatch)
        pcArray=npy.random.randint(0,nc-1-patchSize,size=numPatch)
        for kk in npy.arange(numPatch):
            idxPixels=GetPatchIdxCIFAR10(prArray[kk],pcArray[kk],patchSize)
            dataTemp=imgData[imgIdxArray[kk],idxPixels]
            mu=npy.mean(dataTemp)
            sigma=npy.std(dataTemp)
            patchData[kk,:]=(dataTemp-mu)/(sigma+1e-8)
        patchDataCov=npy.dot(patchData.T,patchData)/patchData.shape[0]
        eigVals,eigVecs=npy.linalg.eig(patchDataCov)
        diagEigVals = npy.diag(1./npy.sqrt(eigVals+1e-8))
        matWhitten= npy.dot(npy.dot(eigVecs, diagEigVals),eigVecs.T)
        patchDataWhitten=npy.dot(patchData,matWhitten)
        print("Patch sampling done, performing kmeans...")
        modelKmeans = KMeans(n_clusters=numAtom, random_state=0).fit(patchDataWhitten)
        modelCoding={"matWhitten":matWhitten,"codeDict":modelKmeans.cluster_centers_}
    return modelCoding
        
def GetImgFeat(img,patchSize,modelCoding): 
    codeDict=modelCoding["codeDict"]
    matWhitten=modelCoding["matWhitten"]
    sizeDict=codeDict.shape[0]
    nr1=(nr-patchSize)
    nc1=(nc-patchSize)
    numPatch=nr1*nc1
    patchCode=npy.zeros((numPatch,sizeDict),dtype=npy.float32)
    idxPatch=0
    dataTemp=npy.zeros(patchSize*patchSize*3,dtype=npy.float32)
    dataDiff=npy.zeros((sizeDict,patchSize*patchSize*3),dtype=npy.float32)
    for pr in range(nr1):
        for pc in range(nc1):
            idxPixels=GetPatchIdxCIFAR10(pr,pc,patchSize)
            dataTemp=img[idxPixels]
            mu=npy.mean(dataTemp)
            sigma=npy.std(dataTemp)
            dataTemp=(dataTemp-mu)/(sigma+1e-8)
            dataTemp=npy.dot(dataTemp,matWhitten)
            dataDiff=codeDict-dataTemp[npy.newaxis,:]
            distTemp=npy.sum(dataDiff*dataDiff,axis=1)/codeDict.shape[1]
            distTemp=npy.mean(distTemp)-distTemp
            patchCode[idxPatch,distTemp>0]=distTemp[distTemp>0]
            idxPatch+=1
    codeFeat=npy.zeros(4*sizeDict,dtype=npy.float32)
    idxPoolRegion=0
    stepPoolRow=(nr1+1)/2
    stepPoolCol=(nc1+1)/2
    for pr in [0,nr1-stepPoolRow-1]:
        for pc in [0,nc1-1-stepPoolCol]:
           pcPool,prPool=npy.meshgrid(npy.arange(pc,pc+stepPoolCol),
                                        npy.arange(pr,pr+stepPoolRow))
           idxPool=prPool*nc1+pcPool
           idxPool=idxPool.flatten() 
           codeFeat[idxPoolRegion*sizeDict:(idxPoolRegion+1)*sizeDict]= \
           npy.max(patchCode[idxPool,:],axis=0)
           idxPoolRegion+=1
    return codeFeat
    
    

def ProcessCIFAR10():
    datapath="E:\\DevProj\\Datasets\\CIFAR10\\cifar-10-batches-py"
    listDataProc=["data_batch_1","data_batch_2","data_batch_3",
                  "data_batch_4","data_batch_5","test_batch"]
    patchSize=7   
    
#    modelCoding={"matWhitten":npy.random.randn(147,147),"codeDict":npy.random.randn(200,147)}
    modelCoding=TrainCodingModel("cifar10",datapath,patchSize)    
    for ii in range(len(listDataProc)):
        datafile=os.path.join(datapath,listDataProc[ii])
        dataDict=unpickle(datafile)
        imgData=dataDict['data']
        numImg=imgData.shape[0]
        sizeDict=modelCoding["codeDict"].shape[0]
        batchFeature=npy.zeros((numImg,2*2*sizeDict),dtype=npy.float32)
        for jj in range(numImg):
            batchFeature[jj,:]=GetImgFeat(imgData[jj,:],patchSize,modelCoding)            
            if 0==jj%100:
                print ('Processing '+listDataProc[ii]+', '+str(jj)+' processed')
        batchFeature=batchFeature.astype(npy.float32)
        fo=open(listDataProc[ii]+"_bow","wb")
        cPickle.dump(batchFeature,fo,protocol=0) 
        
        fo.close()

if __name__=="__main__":
     ProcessCIFAR10()
    