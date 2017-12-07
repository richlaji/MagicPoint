import cv2
import numpy as np
import random
import sys
import os

#for each pic
#orginal pic
#blur pic
#noise pic

imgCount = 4

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
    print(xIndex)
    print(yIndex)
    for _ in range(imgCount):
        for j in range(len(xIndex)):
            f.write(str(xIndex[j])) 
            f.write(',')    
            f.write(str(yIndex[j])) 
            f.write(' ')            
        f.write('\n')
    return img

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
    for _ in range(imgCount):
        for j in range(len(triangle)):
            f.write(str(triangle[j][0])) 
            f.write(',')    
            f.write(str(triangle[j][1])) 
            f.write(' ')            
        f.write('\n')

    #record
    return img

def generateLine(img,startX,startY,Width,Height,number):
    #draw n line
    xIndex = []
    yIndex = []
    for _ in range(number):
        beginX = int(random.random()*Width)
        beginY = int(random.random()*Height)
        endX = int(random.random()*Width)
        endY = int(random.random()*Height)
        xIndex.append(beginX)
        xIndex.append(endX)
        yIndex.append(beginY)
        yIndex.append(endY)
        cv2.line(img,(startX+beginX,startY+beginY),(startX+endX,startY+endY),int(random.random()*200+55),int(random.random()*2+5))

    #record line corner
    for _ in range(imgCount):
        for j in range(len(xIndex)):
            f.write(str(xIndex[j])) 
            f.write(',')    
            f.write(str(yIndex[j])) 
            f.write(' ')            
        f.write('\n')

    return img

def generateEllipse(img,startX,startY,Width,Height):
    threshold = 20
    if Width < Height:
        Smaller = Width - threshold
    else:
        Smaller = Height - threshold
    cv2.ellipse(img,(startX + Width/2,startY + Height/2),(int(random.random()*Smaller/2)+threshold,int(random.random()*Smaller/2)+threshold),int(random.random()*360),0,360,int(random.random()*200+55),-1)

    #record corner (0 corner)
    for _ in range(imgCount):        
        f.write('\n')

    return img

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
        img = np.zeros([height, width], dtype = np.uint8)
        #img = generateQuadrangle(img,250,250,160,120)
        #img = generateTriangle(img,0,0,160,120)
        #img = generateLine(img,0,0,640,480,2)
        img = generateEllipse(img,100,100,200,200)
        #blur 
        imBlur = cv2.GaussianBlur(img,(3,3),5)
        imBlur2 = cv2.GaussianBlur(img,(11,11),5)
        #noise
        imGauss = addGaussianNoise(img)

        cv2.imwrite(folder+str(i*imgCount)+".png",img)
        cv2.imwrite(folder+str(i*imgCount+1)+".png",imBlur)
        cv2.imwrite(folder+str(i*imgCount+2)+".png",imBlur2)
        cv2.imwrite(folder+str(i*imgCount+3)+".png",imGauss)

        #c = cv2.waitKey(0)
        #if c == 27:
        #   exit()
        #write label file
        
        if i % 10 == 0: 
            print(i)
    print("finish!") 
    f.close()