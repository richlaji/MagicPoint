import cv2
import sys
import numpy as np
import os

#feature detector
#sift 
def siftDetector(img,pathName,imgName,kpsName):
    detector = cv2.xfeatures2d.SIFT_create()
    kps = detector.detect(img)
    img = cv2.drawKeypoints(img,kps,img,color=(255,255,255))
    writeKeyPoints(kps,kpsName+imgName.split('.')[0]+'.txt')
    cv2.imwrite(pathName+imgName,img)
    return img

#surf
def surfDetector(img,pathName,imgName,kpsName):
    detector = cv2.xfeatures2d.SURF_create()
    kps = detector.detect(img)
    img = cv2.drawKeypoints(img,kps,img,color=(255,255,255))
    writeKeyPoints(kps,kpsName+imgName.split('.')[0]+'.txt')
    cv2.imwrite(pathName+imgName,img)
    return img

#fast
def fastDetector(img,pathName,imgName,kpsName):
    detector = cv2.FastFeatureDetector_create()
    kps = detector.detect(img,None)
    img = cv2.drawKeypoints(img,kps,img,color=(255,255,255))
    writeKeyPoints(kps,kpsName+imgName.split('.')[0]+'.txt')
    cv2.imwrite(pathName+imgName,img)
    return img

#orb
def orbDetector(img,pathName,imgName,kpsName):
    detector = cv2.ORB_create(500)
    kps = detector.detect(img,None)
    img = cv2.drawKeypoints(img,kps,img,color=(255,255,255))
    writeKeyPoints(kps,kpsName+imgName.split('.')[0]+'.txt')
    cv2.imwrite(pathName+imgName,img)

#harris
def harrisDetector(img,pathName,imgName,kpsName):
    detector = cv2.cornerHarris(img, 2, 3, 0.04)
    #detector = cv2.dilate(detector,None)
    kps = findKPForHarris(detector)
    img = cv2.drawKeypoints(img,kps,img,color=(255,255,255))
    writeKeyPoints(kps,kpsName+imgName.split('.')[0]+'.txt')
    cv2.imwrite(pathName+imgName,img)

#Harris
def findKPForHarris(detector):
    max = detector.max()
    kps=[]
    for i in range(detector.shape[0]):
        for j in range(detector.shape[1]):
            if detector[i,j] > 0.01 * max:
                kps.append(cv2.KeyPoint(j,i,2))
    return kps

#store kp
def writeKeyPoints(kps,fileName):
    f = open(fileName,'w')
    for kp in kps:
        f.write(str(kp.pt[0]))
        f.write(',')
        f.write(str(kp.pt[1]))
        f.write('\n')
    f.close()

if __name__ == '__main__':
    #initial
    Width = 640
    Height = 480

    path = sys.argv[1]
    imgsName = os.listdir(path)
    print(imgsName)
    testImgs = []
    os.system("mkdir sift surf fast orb harris")
    os.system("mkdir siftkps surfkps fastkps orbkps harriskps")
    for iName in imgsName:
        tmpImg = cv2.imread(path+'/'+iName,0)
        tmpImg = cv2.resize(tmpImg,(Width,Height),interpolation=cv2.INTER_CUBIC)
        siftDetector(tmpImg,'sift/',iName,'siftkps/')
        surfDetector(tmpImg,'surf/',iName,'surfkps/')
        fastDetector(tmpImg,'fast/',iName,'fastkps/')
        orbDetector(tmpImg,'orb/',iName,'orbkps/')
        harrisDetector(tmpImg,'harris/',iName,'harriskps/')