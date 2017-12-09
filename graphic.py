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
def addGaussianNoise(src):
    imGaussian = np.zeros([src.shape[0],src.shape[1]], dtype = np.uint8)
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            imGaussian[i][j] = src[i][j] + random.gauss(0,0.5)
            if imGaussian[i][j] < 0:
                imGaussian[i][j] = 0
            if imGaussian[i][j] > 255:
                imGaussian[i][j] = 255
    return imGaussian

def generateQuadrangle(img,startX,startY,Width,Height):
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
    cv2.fillConvexPoly(img, quadrangle, random.random()*200+55)

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
        img = np.zeros([height, width], dtype = np.uint8)
        img,xQ,yQ = generateQuadrangle(img,0,0,320,240)
        img,xT,yT = generateTriangle(img,320,0,320,240)
        img,xL,yL = generateLine(img,0,240,320,240,5)
        img,xE,yE = generateEllipse(img,320,240,320,240)

        #corner 
        xIndex = xQ + xT + xL + xE
        yIndex = yQ + yT + yL + yE

        for _ in range(imgCount):
            for j in range(len(xIndex)):
                f.write(str(xIndex[j]))
                f.write(',')
                f.write(str(yIndex[j]))
                f.write(' ')
            f.write('\n')

        #blur 
        imBlur = cv2.GaussianBlur(img,(5,5),5)
        #noise
        imGauss = addGaussianNoise(img)

        cv2.imwrite(folder+str(i*imgCount)+".png",img)
        cv2.imwrite(folder+str(i*imgCount+1)+".png",imBlur)
        cv2.imwrite(folder+str(i*imgCount+2)+".png",imGauss)
        
        if i % 10 == 0: 
            print(i)

    print("finish!") 
    f.close()