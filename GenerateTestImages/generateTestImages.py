import random
import sys
import os
import cv2
import math
import numpy as np

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

def generateBackgroundPic(height,width,light):
    img = np.zeros([height,width],dtype = np.int32)
    for i in range(height):
        for j in range(width):
            img[i,j] = int(light+random.gauss(0,50))
            if img[i,j] > 255:
                img[i,j] = 255
            if img[i,j] < 0:
                img[i,j] = 0
    img = img.astype(np.uint8)
    imgBlur = cv2.GaussianBlur(img,(11,11),11)
    return imgBlur

def generateQuadrangleWithCenter(img,centerX,centerY,Width,Height):
    return generateQuadrangle(img,centerX - Width / 2, centerY - Height / 2,Width,Height)

def generateQuadrangle(img,startX,startY,Width,Height):
    marginX = Width/10
    marginY = Height/10
    #generate four point from 0 to 0
    ltx = int(random.random()*(Width/2-2*marginX)+marginX)
    lty = int(random.random()*(Height/2-2*marginY)+marginY)
    rtx = int(random.random()*(Width/2-2*marginX)+marginX+Width/2)
    rty = int(random.random()*(Height/2-2*marginY)+marginY)
    lbx = int(random.random()*(Width/2-2*marginX)+marginX)
    lby = int(random.random()*(Height/2-2*marginY)+marginY+Height/2)
    rbx = int(random.random()*(Width/2-2*marginX)+marginX+Width/2)
    rby = int(random.random()*(Height/2-2*marginY)+marginY+Height/2)

    quadrangle = np.array([[[ltx+startX,lty+startY], [rtx+startX,rty+startY], [rbx+startX,rby+startY], [lbx+startX,lby+startY]]], dtype = np.int32)
    color = int(random.random()*200+55)
    cv2.fillConvexPoly(img, quadrangle, color)

    #record corner
    #corner coordinate start from 0
    xIndex = [ltx+startX,rtx+startX,rbx+startX,lbx+startX]
    yIndex = [lty+startY,rty+startY,rby+startY,lby+startY]

    return img,xIndex,yIndex,color

#rotation
def generateRotationPic(centerX,centerY,xIndex,yIndex,times,color,Width,Height):
    imgs = []
    newXIndexs = []
    newYIndexs = []
    for i in range(times)[1:]:
        img = np.zeros([Height,Width],dtype = np.uint8)
        theta = math.pi * 2 / times * i
        newXIndex = []
        newYIndex = []
        for j in range(len(xIndex)):
            newXIndex.append(centerX + (xIndex[j] - centerX) * math.cos(theta) - (yIndex[j] - centerY) * math.sin(theta))
            newYIndex.append(centerY + (xIndex[j] - centerX) * math.sin(theta) + (yIndex[j] - centerY) * math.cos(theta))
        quadrangle = np.array([[newXIndex[0],newYIndex[0]],[newXIndex[1],newYIndex[1]],[newXIndex[2],newYIndex[2]],[newXIndex[3],newYIndex[3]]],dtype = np.int32)
        cv2.fillConvexPoly(img,quadrangle,color)
        imgs.append(img)
        newXIndexs.append(newXIndex)  #not return now
        newYIndexs.append(newYIndex)  #not return now
    return imgs

#scale
def generateDifScalePic(centerX,centerY,xIndex,yIndex,times,color,Width,Height):
    imgs = []
    newXIndexs = []
    newYIndexs = []
    for i in range(times)[1:]:
        img = np.zeros([Height,Width],dtype = np.uint8)
        scale = 1.0 / times * i
        newXIndex = []
        newYIndex = []
        for j in range(len(xIndex)):
            newXIndex.append(centerX + (xIndex[j] - centerX) * scale)
            newYIndex.append(centerY + (yIndex[j] - centerY) * scale)
        quadrangle = np.array([[newXIndex[0],newYIndex[0]],[newXIndex[1],newYIndex[1]],[newXIndex[2],newYIndex[2]],[newXIndex[3],newYIndex[3]]],dtype = np.int32)
        cv2.fillConvexPoly(img,quadrangle,color)
        imgs.append(img)
        newXIndexs.append(newXIndex)  #not return now
        newYIndexs.append(newYIndex)  #not return now
    return imgs

#noise
def generateNoisePic():
    return 0

#blur
def generateBlurPic():
    return 0

#test images : ratation, scale, noise, blur
#
if __name__ == '__main__':
    Width = 640
    Height = 480
    Size = 400
    img = np.zeros([Height,Width],dtype = np.uint8)
    img,xIndex,yIndex,color = generateQuadrangleWithCenter(img,Width/2,Height/2,Size,Size)
    imgsRotation = generateRotationPic(Width/2,Height/2,xIndex,yIndex,36,color,Width,Height)
    imgsScale = generateDifScalePic(Width/2,Height/2,xIndex,yIndex,5,color,Width,Height)
    #to array
    imgs = [img]
    imgs += imgsRotation
    imgs += imgsScale

    #write
    for i in range(len(imgs)):
        cv2.imwrite(str(i+1)+'.png',imgs[i])

    print('finish!')


    