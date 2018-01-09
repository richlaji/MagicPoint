import cv2
import numpy as np
import random
import sys
import os


def addGaussianNoise(src,degree=10):
    imGaussian = np.zeros([src.shape[0],src.shape[1]], dtype = np.int32)
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            imGaussian[i][j] = src[i][j] + random.gauss(0,degree)
            if imGaussian[i][j] < 0:
                imGaussian[i][j] = 0
            if imGaussian[i][j] > 255:
                imGaussian[i][j] = 255
    imGaussian = imGaussian.astype(np.uint8)
    return imGaussian

def generateBlur(img,foldName):
    for i in range(20)[1:]:
        imBlur = cv2.GaussianBlur(img,(11,11),i)
        cv2.imwrite(foldName+'/'+iName.split('.')[0]+'_blur'+str(i)+'.'+iName.split('.')[1],imBlur)

def generateNoise(img,foldName):
    for i in range(20)[1:]:
        imNoise = addGaussianNoise(img,i)
        cv2.imwrite(foldName+'/'+iName.split('.')[0]+'_noise'+str(i)+'.'+iName.split('.')[1],imNoise)

if __name__ == '__main__':
    imgsName = os.listdir(sys.argv[1])
    for iName in imgsName:
        img = cv2.imread(sys.argv[1]+'/'+iName,0)
        img = cv2.resize(img,(640,480),interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(sys.argv[2]+'/'+iName,img)
        generateNoise(img,sys.argv[2])
        generateBlur(img,sys.argv[2])
        
