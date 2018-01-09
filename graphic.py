import cv2
import numpy as np
import random
import sys
import os

#for each pic
#orginal pic
#blur pic
#noise pic

imgCount = 3

#process picture
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
    cv2.fillConvexPoly(img, quadrangle, random.random()*255)

    #record corner
    #corner coordinate start from 0
    xIndex = [ltx+startX,rtx+startX,lbx+startX,rbx+startX]
    yIndex = [lty+startY,rty+startY,lby+startY,rby+startY]
    #for _ in range(imgCount):
    #    for j in range(len(xIndex)):
    #        f.write(str(xIndex[j])) 
    #        f.write(',')    
    #        f.write(str(yIndex[j])) 
    #        f.write(' ')            
    #    f.write('\n')
    return img,xIndex,yIndex

def generateTriangle(img,startX,startY,Width,Height):
    marginX = width/10
    marginY = height/10
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
    triangle = np.array(random.sample(quadrangle[0],3))
    cv2.fillConvexPoly(img, triangle, random.random()*200+55)

    triangle = list(triangle)
    xIndex = []
    yIndex = []

    #record corner

    for j in range(len(triangle)):
        xIndex.append(triangle[j][0])
        yIndex.append(triangle[j][1])

    #record
    return img,xIndex,yIndex

def generateLine(img,startX,startY,Width,Height,number):
    #draw n line
    xIndex = []
    yIndex = []
    miniDis = 50
    for _ in range(number):
        beginX = int(random.random()*Width)
        beginY = int(random.random()*Height)
        endX = int(random.random()*Width)
        endY = int(random.random()*Height)

        if abs(beginX-endX) + abs(beginY-endY) < miniDis:
            continue

        xIndex.append(beginX)
        xIndex.append(endX)
        yIndex.append(beginY)
        yIndex.append(endY)
        cv2.line(img,(startX+beginX,startY+beginY),(startX+endX,startY+endY),int(random.random()*200+55),int(random.random()*2+5))

    #record line corner

    return img,xIndex,yIndex

def generateEllipse(img,startX,startY,Width,Height):
    threshold = 20
    if Width < Height:
        Smaller = Width - threshold
    else:
        Smaller = Height - threshold
    cv2.ellipse(img,(startX + Width/2,startY + Height/2),(int(random.random()*Smaller/2)+threshold,int(random.random()*Smaller/2)+threshold),int(random.random()*360),0,360,int(random.random()*200+55),-1)

    xIndex = []
    yIndex = []
    #record corner (0 corner)

    return img,xIndex,yIndex

def drawCorner(originImg,corners):
    for c in corners:
        originImg = cv2.circle(originImg,c,10,(255,255,255))
    return originImg


if __name__ == '__main__':
    #python graphic 
    times = int(sys.argv[1])
    folder = sys.argv[2]+'/'
    os.system('mkdir ' + sys.argv[2])
    width = 640
    height = 480
    #y x--
    #|
    #|
    f = open('label.txt','w')
    for i in range(times):
        #init img
        xIndex = []
        yIndex = []
        img = generateBackgroundPic(480,640,int(random.random()*100+100))
        xIndex = np.zeros([10,4],dtype=np.int32)
        yIndex = np.zeros([10,4],dtype=np.int32)
        img,xIndex[0,:],yIndex[0,:] = generateQuadrangle(img,0,0,80,60)
        img,xIndex[1,:],yIndex[1,:] = generateQuadrangle(img,80,0,80,60)
        img,xIndex[2,:],yIndex[2,:] = generateQuadrangle(img,0,60,80,60)
        img,xIndex[3,:],yIndex[3,:] = generateQuadrangle(img,80,60,80,60)
        img,xIndex[4,:],yIndex[4,:] = generateQuadrangle(img,160,0,160,120)
        img,xIndex[5,:],yIndex[5,:] = generateQuadrangle(img,0,120,160,120)
        img,xIndex[6,:],yIndex[6,:] = generateQuadrangle(img,160,120,160,120)
        img,xIndex[7,:],yIndex[7,:] = generateQuadrangle(img,320,0,320,240)
        img,xIndex[8,:],yIndex[8,:] = generateQuadrangle(img,0,240,320,240)
        img,xIndex[9,:],yIndex[9,:] = generateQuadrangle(img,320,240,320,240)
        
        #img,xT,yT = generateTriangle(img,320,0,320,240)
        #img,xL,yL = generateLine(img,0,240,320,240,5)
        #img,xE,yE = generateEllipse(img,320,240,320,240)

        #corner 
        #xIndex = xQ + xT + xL + xE
        #yIndex = yQ + yT + yL + yE

        corners = []
        for _ in range(imgCount):
            for j in range(len(xIndex)):
                for k in range(len(xIndex[j])):
                    f.write(str(xIndex[j][k]))
                    f.write(',')
                    f.write(str(yIndex[j][k]))
                    f.write(' ')
                    corners.append((xIndex[j][k],yIndex[j][k]))
            f.write('\n')

        #imgWithCorners = drawCorner(img,corners)

        #cv2.imshow('corners',imgWithCorners)
        #cv2.waitKey(0)

        #blur 
        blurKernelSize = int(random.random()*10+1)*2+1
        imBlur = cv2.GaussianBlur(img,(blurKernelSize,blurKernelSize),random.random()*10)
        #noise
        imGauss = addGaussianNoise(img,random.random()*20)

        cv2.imwrite(folder+str(i*imgCount)+".png",img)
        cv2.imwrite(folder+str(i*imgCount+1)+".png",imBlur)
        cv2.imwrite(folder+str(i*imgCount+2)+".png",imGauss)
        
        if i % 10 == 0: 
            print(i)

    print("finish!") 
    f.close()
