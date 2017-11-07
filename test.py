import numpy as np
import cv2
import sys

import tensorflow as tf

def readData(dataSetPath,begin,end):
    #read label & image (from begin to end)
    #label

    #label shape (number of image, height/8, width/8, 65)
    #coordinates shape (number of image, number of corners)

    f = open(dataSetPath+'/label.txt','r')
    coordinatesX = []
    coordinatesY = []
    for line in f.readlines():
        coordinates = line.split(' ')[:-1]
        coordinatesX.append([int(s.split(',')[0]) for s in coordinates])
        coordinatesY.append([int(s.split(',')[1]) for s in coordinates])
    f.close()

    #image
    images = []
    for i in range(end+1)[begin:]:
        img = cv2.imread(dataSetPath+'/image/'+str(i)+'.png',0)
        img = img.astype(float)
        img /= 255
        images.append(img)
    images = np.array(images)
    print(images.shape)

    #modified label
    label = np.zeros((len(coordinatesX),int(len(images[0])/8),int(len(images[0][0])/8),65))
    for i in range(len(coordinatesX)):
        for r in range(int(len(images[0])/8)):
            for c in range(int(len(images[0][0])/8)):
                label[i][r][c][64] = 1

    #for each block
    #0 1 2 3 4 5 6 7
    #8 9 10 11 12 13 14 15
    #....
    for i in range(len(coordinatesX)):
      for j in range(len(coordinatesX[i])):
          label[i][int(coordinatesY[i][j]/8)][int(coordinatesX[i][j]/8)][coordinatesY[i][j]%8*8+coordinatesX[i][j]%8] = 1
          label[i][int(coordinatesY[i][j]/8)][int(coordinatesX[i][j]/8)][64] = 0

    print(label.shape)

    return images,label

def fastDetect(img):
    fast = cv2.FastFeatureDetector_create()
    kp = fast.detect(img,None)
    cv2.drawKeypoints(img,kp,img,color = 255)
    return img

def siftDetect(img):
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(img)
    cv2.drawKeypoints(img,kp,img,color = 255)
    return img

def testFeatureDetector():
    img1 = cv2.imread('1.png')
    img2 = cv2.imread('2.png')
    img3 = cv2.imread('3.png')
    img1 = fastDetect(img1)
    img2 = fastDetect(img2)
    img3 = fastDetect(img3)
    cv2.imwrite('compare/1_fast.png',img1)
    cv2.imwrite('compare/2_fast.png',img2)
    cv2.imwrite('compare/3_fast.png',img3)

def testSIFTDetector():
    img1 = cv2.imread('1.png')
    img2 = cv2.imread('2.png')
    img3 = cv2.imread('3.png')
    img1 = siftDetect(img1)
    img2 = siftDetect(img2)
    img3 = siftDetect(img3)
    cv2.imwrite('compare/1_sift.png',img1)
    cv2.imwrite('compare/2_sift.png',img2)
    cv2.imwrite('compare/3_sift.png',img3)

def save():
    w = tf.Variable(tf.random_uniform([10,1]),name='w')
    x = tf.placeholder(tf.float32,[None,10])
    y = tf.matmul(x,w)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print("initial:")
    print(sess.run(w))
    saver = tf.train.Saver()
    saver.save(sess,"model/model.ckpt")

def getw():
    return tf.Variable(tf.random_uniform([10,1]),name='w')

def load():
    w = getw() 
    sess = tf.Session()
    saver = tf.train.Saver() 
    saver.restore(sess,"model/model.ckpt")
    print(sess.run(w))

if __name__ == '__main__':
    #save()
    #load()
    #images,labels = readData(sys.argv[1],0,8)
    #print images.shape
    #print labels.shape
    testFeatureDetector()
    testSIFTDetector()
