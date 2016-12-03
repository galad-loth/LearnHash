import numpy as npy
from matplotlib import pyplot as plt
import cv2


def TestKptMatch():    
    img1=cv2.imread("E:\\DevProj\\Datasets\\VGGAffine\\bark\\img1.ppm",cv2.IMREAD_COLOR)
    img2=cv2.imread("E:\\DevProj\\Datasets\\VGGAffine\\bark\\img2.ppm",cv2.IMREAD_COLOR)
    
    gray1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    gray2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    gap_width=20
    black_gap=npy.zeros((img1.shape[0],gap_width),dtype=npy.uint8)
    
    objSIFT = cv2.SIFT(500)
    kpt1,desc1 = objSIFT.detectAndCompute(gray1,None) 
    kpt2,desc2 = objSIFT.detectAndCompute(gray2,None) 
    objMatcher=cv2.BFMatcher(cv2.NORM_L2)
    matches=objMatcher.knnMatch(desc1,desc2,k=2)
    
    goodMatches=[]
    for bm1,bm2 in matches:
        if bm1.distance < 0.7*bm2.distance:
            goodMatches.append(bm1)
    
    if len(goodMatches)>10:
        ptsFrom = npy.float32([kpt1[bm.queryIdx].pt for bm in goodMatches]).reshape(-1,1,2)
        ptsTo = npy.float32([kpt2[bm.trainIdx].pt for bm in goodMatches]).reshape(-1,1,2)
        matH, matchMask = cv2.findHomography(ptsFrom, ptsTo, cv2.RANSAC,5.0)
    
    imgcnb=npy.concatenate((gray1,black_gap,gray2),axis=1)
    
    plt.figure(1,figsize=(15,6))
    plt.imshow(imgcnb,cmap="gray")
    idx=0
    for bm in goodMatches:
        if 1==matchMask[idx]:
            kptFrom=kpt1[bm.queryIdx]
            kptTo=kpt2[bm.trainIdx]
            plt.plot(kptFrom.pt[0],kptFrom.pt[1],"rs",
                     markerfacecolor="none",markeredgecolor="r",markeredgewidth=2)
            plt.plot(kptTo.pt[0]+img1.shape[1]+gap_width,kptTo.pt[1],"bo",
                     markerfacecolor="none",markeredgecolor="b",markeredgewidth=2)
            plt.plot([kptFrom.pt[0],kptTo.pt[0]+img1.shape[1]+gap_width],
                     [kptFrom.pt[1],kptTo.pt[1]],"g-",linewidth=2)
        idx+=1
    plt.axis("off")


if __name__=="__main__":
    TestKptMatch()