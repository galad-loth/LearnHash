import numpy as npy
import cv2
from scipy.spatial.distance import cdist
from scipy import linalg
from LoadData import ReadCIFAR10Gist
from matplotlib import pyplot as plt


def TrainSDH(data,label,lenCode,numAnchors,paraLambda,paraEta,numIter):
    dataNum,dataDim=data.shape
    idxRand=npy.random.permutation(dataNum)
    anchorData=data[idxRand[:numAnchors],:]
    distToAnchors=cdist(anchorData, data,'euclidean')
    sigmaRBF=npy.std(npy.concatenate((distToAnchors,-distToAnchors)).flatten())
    phi=npy.exp(-npy.square(distToAnchors)/sigmaRBF/sigmaRBF/2)
    matAux=npy.dot(linalg.inv(npy.dot(phi, phi.T)),phi)
    code=npy.random.randint(0,2,(lenCode,dataNum))
    code=code.astype(npy.float32)*2-1
    #begin iteration
    for tau in range(numIter):
        print("Training SDH... round "+str(tau))
        matW=linalg.inv(npy.dot(code,code.T)+paraLambda*npy.eye(lenCode))
        matW=npy.dot(matW,npy.dot(code,label))
        matP=npy.dot(matAux,code.T)
        matQ=npy.dot(matW,label.T)+paraEta*npy.dot(matP.T,phi)
        for kk in range(lenCode):
            idxTemp=npy.ones(lenCode,dtype=npy.uint8)
            idxTemp[kk]=0
            vecZ=matQ[kk,:]-npy.dot(npy.dot(matW[kk,:],matW[idxTemp,:].T),code[idxTemp,:])
            code[kk,vecZ>=0]=1
            code[kk,vecZ<0]=-1 
    
    model={"anchorData":anchorData,"sigmaRBF":sigmaRBF,"matProj":matP}
    return code.T ,model 
    
def EvalSHD(data,model):
    dataNum,dataDim=data.shape
    distToAnchors=cdist(model["anchorData"], data,'euclidean')
    sigmaRBF=model["sigmaRBF"]
    phi=npy.exp(-npy.square(distToAnchors)/sigmaRBF/sigmaRBF/2)
    matP=model["matProj"]
    projData=npy.dot(matP.T,phi)
    code=npy.ones(projData.shape,dtype=npy.int32)
    code[projData<0]=-1
    if 0==code.shape[0]%8:
        codeByteNum=code.shape[0]/8
    else:
        codeByteNum=1+code.shape[0]/8
    compactCode=npy.zeros((code.shape[1],codeByteNum),dtype=npy.uint8)
    for kk in range(code.shape[0]):
        idxByte=kk/8
        idxBit=kk%8
        compactCode[code[kk,:]==1,idxByte]+=(1<<idxBit)
    return compactCode


if __name__=="__main__":
    trainData, trainLabel,testData,testLabel=ReadCIFAR10Gist()
    labelMat=npy.zeros((trainData.shape[0],10),dtype=npy.float32)
    for kk in range(10):
        labelMat[trainLabel==kk,kk]=1
    lenCode=64
    numAnchors=1024
    numIter=5
    paraLambda=1
    paraEta=1e-5
    codeTrain0, modelSDH=TrainSDH(trainData, labelMat,lenCode,numAnchors,paraLambda,paraEta,numIter)
    
    codeTrain=EvalSHD(trainData,modelSDH)
    codeTest=EvalSHD(testData,modelSDH)