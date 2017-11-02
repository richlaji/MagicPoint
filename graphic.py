import cv2
import numpy as np
import random
import sys
import os

#for each pic
#orginal pic
#blur pic
#noise pic

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

if __name__ == '__main__':
	#python graphic 
	times = int(sys.argv[1])
	folder = sys.argv[2]+'/'
	os.system('mkdir ' + sys.argv[2])
	margin = 15
	width = 160
	height = 120
	#y x--
	#|
	#|
	f = open('label.txt','w')
	for i in range(times) :
		ltx = int(random.random()*(width/2-2*margin)+margin)
		lty = int(random.random()*(height/2-2*margin)+margin)
		rtx = int(random.random()*(width/2-2*margin)+margin+width/2)
		rty = int(random.random()*(height/2-2*margin)+margin)
		lbx = int(random.random()*(width/2-2*margin)+margin)
		lby = int(random.random()*(height/2-2*margin)+margin+height/2)
		rbx = int(random.random()*(width/2-2*margin)+margin+width/2)
		rby = int(random.random()*(height/2-2*margin)+margin+height/2)
		#print ltx,lty
		#print rtx,rty
		#print lbx,lby
		#print rbx,rby
		#print "------------------"
		a = np.array([[[ltx,lty], [rtx,rty], [rbx,rby], [lbx,lby]]], dtype = np.int32)
		im = np.zeros([height, width], dtype = np.uint8)
		cv2.fillConvexPoly(im, a, random.random()*200+55)
		#blur 
		imBlur = cv2.GaussianBlur(im,(11,11),5)
		#noise
		imGauss = addGaussianNoise(im)

		cv2.imwrite(folder+str(i*3)+".png",im)
		cv2.imwrite(folder+str(i*3+1)+".png",imBlur)
		cv2.imwrite(folder+str(i*3+2)+".png",imGauss)

		cv2.circle(im,(ltx,lty),5,255)
		cv2.circle(im,(rtx,rty),5,255)
		cv2.circle(im,(lbx,lby),5,255)
		cv2.circle(im,(rbx,rby),5,255)
		#cv2.imshow("img", im)
		#cv2.imshow("imgBlur",imBlur)
		#cv2.imshow("imGauss",imGauss)

		xIndex = [ltx,rtx,lbx,rbx]
		yIndex = [lty,rty,lby,rby]

		#c = cv2.waitKey(0)
		#if c == 27:
		#	exit()
		#write label file
		#corner coordinate start from 0
		
		for _ in range(3):
			for j in range(len(xIndex)):
				f.write(str(xIndex[j]))	
				f.write(',')	
				f.write(str(yIndex[j]))	
				f.write(' ')			
			f.write('\n')
		if i % 10 == 0: 
			print(i)
	print("finish!") 
	f.close()