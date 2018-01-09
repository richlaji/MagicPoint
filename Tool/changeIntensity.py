import cv2
import numpy as np
import random
import sys
import os

def changeIntensity(img,aimIntensity):
    width = img.shape[0]
    height = img.shape[1]
    averageIntensity = img.sum() / width / height
    img *= (aimIntensity/averageIntensity)
    return img

if __name__ == '__main__':
    imgsName = os.listdir(sys.argv[1])
    for iName in imgsName:
        img = cv2.imread(sys.argv[1]+'/'+iName,0)
        img = img.astype(float)
        img /= 255
        img = changeIntensity(img,float(sys.argv[3]))
        cv2.imwrite(sys.argv[2]+'/'+iName,img*255)