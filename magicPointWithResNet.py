from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
import cv2

import tensorflow as tf
from tensorflow.python.framework import ops  
from tensorflow.python.ops import control_flow_ops  
from tensorflow.python.training import moving_averages 

import numpy as np

import time
import os
import sys
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

height = 480
width = 640
itTimes = 60001
testTimes = 1000
saveTimes = 1000
batchSize = 10
totalData = 300000
learningRate = 2e-7
cornerThreshold = 0.05

#region size. The value in magic point is regionSize
regionSize = 16

testIndex = 3

#layer number
layerNumber = [3,3,3]

def readData(dataSetPath,begin,end):
    #read label & image (from begin to end)
    #label

    #label shape (number of image, height/regionSize, width/regionSize, regionSize*regionSize+1)
    #coordinates shape (number of image, number of corners)

    f = open(dataSetPath+'/label.txt','r')
    coordinatesX = []
    coordinatesY = []
    for line in f.readlines()[begin:end+1]:
        coordinates = line.split(' ')[:-1]
        coordinatesX.append([int(s.split(',')[0]) for s in coordinates])
        coordinatesY.append([int(s.split(',')[1]) for s in coordinates])
    f.close()

    #image
    images = []
    for i in range(end+1)[begin:]:
        img = cv2.imread(dataSetPath+'/'+str(i)+'.png',0)
        img = img.astype(float)
        img /= 255
        images.append(img)
    images = np.array(images)


    #modified label
    labels = np.zeros((len(coordinatesX),int(len(images[0])/regionSize),int(len(images[0][0])/regionSize),regionSize*regionSize+1))
    for i in range(len(coordinatesX)):
        for r in range(int(len(images[0])/regionSize)):
            for c in range(int(len(images[0][0])/regionSize)):
                labels[i][r][c][regionSize*regionSize] = 1

    #for each block
    #0 1 2 3 4 5 6 7
    #regionSize 9 10 11 12 13 14 15
    #....
    for i in range(len(coordinatesX)):
      for j in range(len(coordinatesX[i])):
          labels[i][int(coordinatesY[i][j]/regionSize)][int(coordinatesX[i][j]/regionSize)][coordinatesY[i][j]%regionSize*regionSize+coordinatesX[i][j]%regionSize] = 1
          labels[i][int(coordinatesY[i][j]/regionSize)][int(coordinatesX[i][j]/regionSize)][regionSize*regionSize] = 0


    return images,labels

#4 times conv2d_stride2 
def deepnn(x,is_train):
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, height, width, 1])

    # First convolutional layer
    with tf.name_scope('conv1'):    
        w1 = weight_variable([7, 7, 1, 64],'w1')
        b1 = bias_variable([64],'b1')   
        conv1 = conv2d(x_image, w1, 2)
        bn1 = batch_norm(conv1 + b1, is_train, 'bn1')
        h1 = tf.nn.relu(bn1)

    #layer store the result of last block
    layers = []
    layers.append(h1)
    
    #layerNumber is an array which indicates the number of blocks each layer
    #total 3 layer (i+1th layer's size is half of ith layer)
    # Second convolutional layer
    for i in range(layerNumber[0]):
        with tf.name_scope('conv2_%d' %i):
            resBlock = residual_block(layers[-1],128,is_train)
            layers.append(resBlock)

    # Third convolutional layer
    for i in range(layerNumber[1]):
        with tf.name_scope('conv3_%d' %i):
            resBlock = residual_block(layers[-1],256,is_train)
            layers.append(resBlock)

    # Fourth convolutional layer
    for i in range(layerNumber[2]):
        with tf.name_scope('conv4_%d' %i):
            resBlock = residual_block(layers[-1],512,is_train)
            layers.append(resBlock)
        
    # Fifth convolutional layer
    with tf.name_scope('conv5'):
        lastLayer = convolutionLayer(layers[-1],512,regionSize*regionSize+1,'c5',kernelSize=1)
        bn = batch_norm(lastLayer,is_train, 'bn5')
        h_conv5 = tf.nn.relu(bn)

    return h_conv5 


def residual_block(inputLayer,outputChannel,isTrain):
    #input channel
    inputChannel = inputLayer.get_shape().as_list()[-1]

    #whether change the dimension
    if inputChannel * 2 == outputChannel:
        stride = 2
    else:
        stride = 1

    #residual block
    conv1 = convolutionLayer(inputLayer,inputChannel,outputChannel,'conv1_',3,stride)
    bn1 = batch_norm(conv1, isTrain, 'bn1_')
    relu1 = tf.nn.relu(bn1)

    conv2 = convolutionLayer(relu1,outputChannel,outputChannel,'conv2_',3,1)
    bn2 = batch_norm(conv2, isTrain, 'bn2_')

    #add
    #if stride == 2 then need pool
    if stride == 2:
        poolInput = tf.nn.avg_pool(inputLayer,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
        paddedInput = tf.pad(poolInput,[[0,0],[0,0],[0,0],[inputChannel // 2,inputChannel // 2]])
    else:
        paddedInput = inputLayer

    output = bn2 + paddedInput
    return output

def convolutionLayer(x, inputChannel, outputChannel, name, kernelSize = 3, stride=1):
    w = weight_variable([kernelSize, kernelSize, inputChannel, outputChannel],name+'w1')
    b = bias_variable([outputChannel],name+'b1')
    return conv2d(x, w, stride) + b   


def conv2d(x, W, stride = 1):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')


def batch_norm(x, is_train,n):
    beta = tf.Variable(tf.constant(0.0, shape=[x.shape[-1]]), name=n+'beta', trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[x.shape[-1]]), name=n+'gamma', trainable=True)
    axises = list(range(len(x.shape) - 1))
    batch_mean, batch_var = tf.nn.moments(x, axises, name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.5)

    def mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    is_train = ops.convert_to_tensor(is_train)
    mean, var = tf.cond(is_train, mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
    normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape,n):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.001)
    return tf.Variable(initial,name=n)


def bias_variable(shape,n):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.001, shape=shape)
    return tf.Variable(initial,name=n)


def trainMagicPoint(dataSetPath,restore,modelName,modelTrainTimes):
    # Import data
    #mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    modelTrainTimes = int(modelTrainTimes)

    # Create the model
    x = tf.placeholder(tf.float32, [None, height, width])

    # Define loss and optimizer
    # pixel level corner detection
    y_ = tf.placeholder(tf.float32, [None, height/regionSize, width/regionSize, regionSize*regionSize+1])
    isTrain = tf.placeholder(tf.bool)

    # Build the graph for the deep net
    y_conv = deepnn(x,isTrain)
    softmax = tf.nn.softmax(y_conv)

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                            logits=y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(learningRate).minimize(cross_entropy)

    #with tf.name_scope('accuracy'):
    #  correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    #  correct_prediction = tf.cast(correct_prediction, tf.float32)
    #accuracy = tf.reduce_mean(correct_prediction)

    #graph_location = tempfile.mkdtemp() 
    #print('Saving graph to: %s' % graph_location)
    #train_writer = tf.summary.FileWriter(graph_location)
    #train_writer.add_graph(tf.get_default_graph())

    #for test
    images,labels = readData(dataSetPath,testIndex,testIndex+2)
    imgsName = [str(i+1)+'.png' for i in range(len(images))]
    #trainNum = 250
    testImgs = images
    testLbs = labels

    b = time.time()

    with tf.Session() as sess:
        #restore model
        if restore & (modelTrainTimes != 0):
            saver = tf.train.Saver()
            saver.restore(sess,modelName)
        else:
            sess.run(tf.global_variables_initializer())
            print("init variables")
        for it in range(modelTrainTimes+itTimes)[modelTrainTimes:]:
            #for train
            imgs,lbs = readData(dataSetPath,it*batchSize%totalData,it*batchSize%totalData+batchSize-1)
            train_step.run(feed_dict={x: imgs, y_: lbs, isTrain: True})
            #test code
            if it % testTimes == 0:
                print(str(it) +' times:')
                print(sess.run(cross_entropy, feed_dict={x: testImgs, y_: testLbs, isTrain: False}))
                #test calculate
                test = sess.run(softmax,feed_dict={x: testImgs, y_: testLbs, isTrain: False})
                print(test.shape)
                testImage(test,str(it),imgsName)
                e = time.time()
                print("cost time: " + str(e-b))
                b = e
            #save code
            if it % saveTimes == 0:
                print('save ' + str(it) + ' model')
                saver = tf.train.Saver()
                saver.save(sess,"model/model_"+str(it)+".ckpt")

#only test one or more image without training
def testMagicPoint(dataSetPath,modelName):
    # Import data
    #mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # Create the model
    x = tf.placeholder(tf.float32, [None, height, width])

    # Define loss and optimizer
    # pixel level corner detection
    y_ = tf.placeholder(tf.float32, [None, height/regionSize, width/regionSize, regionSize*regionSize+1])
    isTrain = tf.placeholder(tf.bool)

    # Build the graph for the deep net
    y_conv = deepnn(x,isTrain)
    softmax = tf.nn.softmax(y_conv)

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                            logits=y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)

    #for test
    images,labels = readData(dataSetPath,testIndex,testIndex+2)

    #trainNum = 250
    testImgs = images
    testLbs = labels

    with tf.Session() as sess:
        #sess.run(tf.global_variables_initializer())
        #restore model
        saver = tf.train.Saver()
        saver.restore(sess,modelName)
        
        #print(sess.run(cross_entropy, feed_dict={x: testImgs, y_: testLbs, isTrain: False}))
        #test calculate
        test = sess.run(softmax,feed_dict={x: testImgs, y_: testLbs, isTrain: False})
        print(test.shape)
        testImage(test,'test')
        for i in range(len(testImgs)):
            corners = findCorner2(test[i])
            imgWithCorner = drawCorner(testImgs[i],corners)
            cv2.imwrite(str(i+1)+'_withCorner.png',imgWithCorner*255)

#compare func
def comp(a,b):
    numA = int(a.split('.')[0])
    numB = int(b.split('.')[0])
    if numA > numB:
        return 1
    elif numB > numA:
        return -1
    else:
        return 0

#test a folder
def testMagicPointForAFolder(path,modelName,storeImages=False):
    # Import data
    #mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # Create the model
    x = tf.placeholder(tf.float32, [None, height, width])

    # Define loss and optimizer
    # pixel level corner detection
    y_ = tf.placeholder(tf.float32, [None, height/regionSize, width/regionSize, regionSize*regionSize+1])
    isTrain = tf.placeholder(tf.bool)

    # Build the graph for the deep net
    y_conv = deepnn(x,isTrain)
    softmax = tf.nn.softmax(y_conv)

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                            logits=y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)

    #test image loading
    imgsName = os.listdir(path)
    #imgsName = imgsName.sort(comp)
    print(imgsName)
    testImgs = []
    for iName in imgsName:
        tmpImg = cv2.imread(path+'/'+iName,0)
        tmpImg = cv2.resize(tmpImg,(width,height),interpolation=cv2.INTER_CUBIC)
        tmpImg = tmpImg.astype(float)
        tmpImg /= 255
        testImgs.append(tmpImg)

    with tf.Session() as sess:
        #sess.run(tf.global_variables_initializer())
        #restore model
        saver = tf.train.Saver()
        saver.restore(sess,modelName)
        
        #print(sess.run(cross_entropy, feed_dict={x: testImgs, y_: testLbs, isTrain: False}))
        #test calculate
        test = sess.run(softmax,feed_dict={x: testImgs, isTrain: False})
        print(test.shape)
        
        #new a folder to store result
        os.system('mkdir result')
        os.system('mkdir resultkps')
        testImage(test,'test',imgsName)
        print(storeImages)
        for i in range(len(testImgs)):
            print(i+1)
            corners = findCorner(test[i])  #cv2.KeyPoint
            #corners = localMaximumSuppresion(test[i],corners,3)
            testImgs[i] = (testImgs[i]*255).astype(np.uint8)
            #imgWithCorner = drawCorner(testImgs[i],corners)
            if storeImages:
                imgWithCorner = cv2.drawKeypoints(testImgs[i],corners,testImgs[i],color=(255,255,255))
                cv2.imwrite('result/'+imgsName[i].split('.')[0]+'_withCorner.png',imgWithCorner)
            writeKeyPoints(corners,'resultkps/'+imgsName[i].split('.')[0]+'.txt')

#store kp
def writeKeyPoints(kps,fileName):
    f = open(fileName,'w')
    for kp in kps:
        f.write(str(kp.pt[0]))
        f.write(',')
        f.write(str(kp.pt[1]))
        f.write('\n')
    f.close()

#only test one or more image without training
def testMagicPointForAImg(filename,modelName):
    # Import data
    #mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # Create the model
    x = tf.placeholder(tf.float32, [None, height, width])

    # Define loss and optimizer
    # pixel level corner detection
    y_ = tf.placeholder(tf.float32, [None, height/regionSize, width/regionSize, regionSize*regionSize+1])
    isTrain = tf.placeholder(tf.bool)

    # Build the graph for the deep net
    y_conv = deepnn(x,isTrain)
    softmax = tf.nn.softmax(y_conv)

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                            logits=y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)

    testImgs = cv2.imread(filename,0)
    testImgs = cv2.resize(testImgs,(width,height),interpolation=cv2.INTER_CUBIC)
    testImgs = testImgs.astype(float)
    testImgs /= 255
    testImgs = [testImgs]

    with tf.Session() as sess:
        #sess.run(tf.global_variables_initializer())
        #restore model
        saver = tf.train.Saver()
        saver.restore(sess,modelName)
        
        #print(sess.run(cross_entropy, feed_dict={x: testImgs, y_: testLbs, isTrain: False}))
        #test calculate
        test = sess.run(softmax,feed_dict={x: testImgs, isTrain: False})
        print(test.shape)

        #new a folder to store result
        os.system('mkdir result')
        testImage(test,'test')
        for i in range(len(testImgs)):
            corners = findCorner(test[i])
            imgWithCorner = drawCorner(testImgs[i],corners)
            cv2.imwrite('result/'+str(i+1)+'_withCorner.png',imgWithCorner*255)

#test image func
def testImage(test,name,imgsName):
    heatMap = []
    max = []

    for i in range(test.shape[0]):
        heatMap.append(np.zeros([height,width]))
        max.append(0)

    for i in range(int(height/regionSize)):
        for j in range(int(width/regionSize)):
            for k in range(regionSize*regionSize):
                for t in range(test.shape[0]):
                    heatMap[t][int(i*regionSize+k/regionSize)][int(j*regionSize+k%regionSize)] = int(test[t][i][j][k] * 255)
                    if test[t][i][j][k] > max[t]:
                        max[t] = test[t][i][j][k]      

    for i in range(test.shape[0]):
        print('max of pic ' + imgsName[i] + " is " + str(max[i]))

    for i in range(test.shape[0]): 
        cv2.imwrite('result/' + imgsName[i].split('.')[0] + '_' + name + '.png',heatMap[i])

#find corner through simple way
def findPoint(heatMap):
    max = 0
    for i in range(height):
        for j in range(width):
                if heatMap[i][j] > max:
                    max = heatMap[i][j]
    for i in range(height):
        for j in range(width):
            if heatMap[i][j] > 0.9 * max:
                print(j,i)

#find corner through compare
#if [64] is less than a value then may be a corner in this region
def findCorner(heatMap):
    corner = []
    for i in range(int(height/regionSize)):
        for j in range(int(width/regionSize)):
            #print(heatMap[i][j][64])
            maxIndex = 0
            for k in range(regionSize*regionSize):
                if heatMap[i][j][k] > heatMap[i][j][maxIndex]:
                    maxIndex = k
            if heatMap[i][j][maxIndex] > cornerThreshold:
                #print(heatMap[i][j][maxIndex])\
                corner.append(cv2.KeyPoint(int(j*regionSize+maxIndex%regionSize),int(i*regionSize+maxIndex/regionSize),2))
    corner = localMaximumSuppresion(heatMap,corner,3)
    print('corner:'+str(len(corner)))
    return corner

#input heatMap and cornerS
#output cornerS
def localMaximumSuppresion(heatMap,corners,kernelSize=3):
    newCorners = []
    for corner in corners:
        isMax = True
        x = int(corner.pt[0] - int(kernelSize/2))
        y = int(corner.pt[1] - int(kernelSize/2))
        centerX = corner.pt[0]
        centerY = corner.pt[1]
        for i in range(kernelSize):
            for j in range(kernelSize):
                if (x+i >= 0) & (x+i < width) & (y+j >=0) & (y+j < height):
                    #print(i*3+j+1,int((y+j)%regionSize*regionSize+(x+i)%regionSize+0.1),heatMap[int((y+j)/regionSize)][int((x+i)/regionSize)][int((y+j)%regionSize*regionSize+(x+i)%regionSize+0.1)])
                    if heatMap[int(centerY/regionSize)][int(centerX/regionSize)][int(centerY%regionSize*regionSize+centerX%regionSize)] < heatMap[int((y+j)/regionSize)][int((x+i)/regionSize)][int((y+j)%regionSize*regionSize+(x+i)%regionSize)]:
                        print(heatMap[int(centerY/regionSize)][int(centerX/regionSize)][int(centerY%regionSize*regionSize+centerX%regionSize)],heatMap[int((y+j)/regionSize)][int((x+i)/regionSize)][int((y+j)%regionSize*regionSize+(x+i)%regionSize)])
                        isMax = False
        if isMax:
            newCorners.append(corner)
    return newCorners


def drawCorner(originImg,corners):
    for c in corners:
        originImg = cv2.circle(originImg,c,10,(1,1,1))
    return originImg

def checkTheSumOfARegion(test):
    #test size: (number of img,width/regionSize,height/regionSize,regionSize*regionSize+1)
    for t in range(test.shape[0]):
        for i in range(int(height/regionSize)):
            for j in range(int(width/regionSize)):
                sumOfPossibility = 0
                for k in range(regionSize*regionSize+1):
                    sumOfPossibility += test[t][i][j][k]
                if sumOfPossibility <= 0.99:
                    print("Error at " + str(t) + " " + str(i) + " " + str(j))

def checkLabels(labels):
    #labels' size : (number of img,width/regionSize,height/regionSize,regionSize*regionSize+1)
    for t in range(labels.shape[0]):
        for i in range(int(height/regionSize)):
            for j in range(int(width/regionSize)):
                for k in range(regionSize*regionSize):
                    if labels[t][i][j][k] == 1:
                        print(int(j*regionSize+k%regionSize),int(i*regionSize+k/regionSize))

if __name__ == '__main__':
    if sys.argv[1] == 'train':
        trainMagicPoint(sys.argv[2],True,'model/model_'+sys.argv[3]+'.ckpt',sys.argv[3])
    #testMagicPoint(sys.argv[1],'model/model_'+sys.argv[2]+'.ckpt')
    #testMagicPointForAImg(sys.argv[1],'model/model_'+sys.argv[2]+'.ckpt')
    if sys.argv[1] == 'test':
        if sys.argv[4] == 'True':
            testMagicPointForAFolder(sys.argv[2],'model/model_'+sys.argv[3]+'.ckpt',True)
        else:
            testMagicPointForAFolder(sys.argv[2],'model/model_'+sys.argv[3]+'.ckpt',False)

