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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

height = 480
width = 640
itTimes = 48001
testTimes = 1000
saveTimes = 2000
batchSize = 50
totalData = 120000
learningRate = 5e-5

testIndex = 3

def readData(dataSetPath,begin,end):
    #read label & image (from begin to end)
    #label

    #label shape (number of image, height/8, width/8, 65)
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
    labels = np.zeros((len(coordinatesX),int(len(images[0])/8),int(len(images[0][0])/8),65))
    for i in range(len(coordinatesX)):
        for r in range(int(len(images[0])/8)):
            for c in range(int(len(images[0][0])/8)):
                labels[i][r][c][64] = 1

    #for each block
    #0 1 2 3 4 5 6 7
    #8 9 10 11 12 13 14 15
    #....
    for i in range(len(coordinatesX)):
      for j in range(len(coordinatesX[i])):
          labels[i][int(coordinatesY[i][j]/8)][int(coordinatesX[i][j]/8)][coordinatesY[i][j]%8*8+coordinatesX[i][j]%8] = 1
          labels[i][int(coordinatesY[i][j]/8)][int(coordinatesX[i][j]/8)][64] = 0


    return images,labels

def deepnn(x,is_train):
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, height, width, 1])

    # First convolutional layer - maps one grayscale image to 16 feature maps.
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([3, 3, 1, 64],'w1')
        b_conv1 = bias_variable([64],'b1')
        bn_conv1 = batch_norm(conv2d_stride2(x_image, W_conv1) + b_conv1,is_train,'bn1')
        h_conv1 = tf.nn.relu(bn_conv1)

    # Pooling layer - downsamples by 2X.
    #with tf.name_scope('pool1'):
    #    h_pool1 = max_pool_2x2(h_conv1)

    # Second convolutional layer -- maps 16 feature maps to 32.
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([3, 3, 64, 128],'w2')
        b_conv2 = bias_variable([128],'b2')
        bn_conv2 = batch_norm(conv2d_stride2(h_conv1, W_conv2) + b_conv2,is_train,'bn2')
        h_conv2 = tf.nn.relu(bn_conv2)

    # Second pooling layer.
    #with tf.name_scope('pool2'):
    #    h_pool2 = max_pool_2x2(h_conv2)

    # Third convolutional layer -- maps 32 feature maps to 65.
    with tf.name_scope('conv3'):
        W_conv3 = weight_variable([3, 3, 128, 256],'w3')
        b_conv3 = bias_variable([256],'b3')
        bn_conv3 = batch_norm(conv2d_stride2(h_conv2, W_conv3) + b_conv3,is_train,'bn3')
        h_conv3 = tf.nn.relu(bn_conv3)

    with tf.name_scope('conv4'):
        W_conv4 = weight_variable([1, 1, 256, 65],'w4')
        b_conv4 = bias_variable([65],'b4')
        bn_conv4 = batch_norm(conv2d(h_conv3, W_conv4) + b_conv4,is_train,'bn4')
        h_conv4 = tf.nn.relu(bn_conv4)

    # Third pooling layer.
    #with tf.name_scope('pool3'):
    #    h_pool3 = max_pool_2x2(h_conv3)

    #softmax
  
    return h_conv4


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def conv2d_stride2(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')

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
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial,name=n)


def bias_variable(shape,n):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial,name=n)


def trainMagicPoint(dataSetPath,restore,modelName,modelTrainTimes):
    # Import data
    #mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    modelTrainTimes = int(modelTrainTimes)

    # Create the model
    x = tf.placeholder(tf.float32, [None, height, width])

    # Define loss and optimizer
    # pixel level corner detection
    y_ = tf.placeholder(tf.float32, [None, height/8, width/8, 65])
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
                testImage(test,str(it))
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
    y_ = tf.placeholder(tf.float32, [None, height/8, width/8, 65])
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

#test a folder
def testMagicPointForAFolder(path,modelName):
    # Import data
    #mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # Create the model
    x = tf.placeholder(tf.float32, [None, height, width])

    # Define loss and optimizer
    # pixel level corner detection
    y_ = tf.placeholder(tf.float32, [None, height/8, width/8, 65])
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
    testImgs = []
    for iName in imgsName:
        tmpImg = cv2.imread(path+'/'+iName,0)
        print(tmpImg.shape())
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
        testImage(test,'test')
        for i in range(len(testImgs)):
            corners = findCorner(test[i])
            imgWithCorner = drawCorner(testImgs[i],corners)
            cv2.imwrite('result/'+str(i+1)+'_withCorner.png',imgWithCorner*255)

#only test one or more image without training
def testMagicPointForAImg(filename,modelName):
    # Import data
    #mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # Create the model
    x = tf.placeholder(tf.float32, [None, height, width])

    # Define loss and optimizer
    # pixel level corner detection
    y_ = tf.placeholder(tf.float32, [None, height/8, width/8, 65])
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
def testImage(test,name):
    heatMap = []
    max = []
    for t in range(test.shape[0]):
        heatMap.append(np.zeros([height,width]))
        max.append(0)

    for i in range(int(height/8)):
        for j in range(int(width/8)):
            for k in range(64):
                for t in range(test.shape[0]):
                    heatMap[t][int(i*8+k/8)][int(j*8+k%8)] = int(test[t][i][j][k] * 255)
                    if test[t][i][j][k] > max[t]:
                        max[t] = test[t][i][j][k]      
    for t in range(test.shape[0]): 
        cv2.imwrite('result/'+str(t+1)+'_'+name+'.png',heatMap[t])

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
    for i in range(int(height/8)):
        for j in range(int(width/8)):
            #print(heatMap[i][j][64])
            if heatMap[i][j][64] < 0.01:
                maxIndex = 0
                for k in range(64):
                    if heatMap[i][j][k] > heatMap[i][j][maxIndex]:
                        maxIndex = k
                corner.append((int(j*8+maxIndex%8),int(i*8+maxIndex/8)))
    print('corner:'+str(len(corner)))
    return corner

#find corner through compare
#for each region a corner should be the max and greater than a value
def findCorner2(heatMap):
    corner = []
    for i in range(int(height/8)):
        for j in range(int(width/8)):
            maxIndex = 64
            for k in range(64):
                if (heatMap[i][j][k] > heatMap[i][j][maxIndex]) & (heatMap[i][j][k] > 0.18):
                    maxIndex = k
            if maxIndex != 64:
                corner.append((int(j*8+maxIndex%8),int(i*8+maxIndex/8)))
    print('corner:'+str(len(corner)))
    return corner

def drawCorner(originImg,corners):
    for c in corners:
        originImg = cv2.circle(originImg,c,10,(1,1,1))
    return originImg

if __name__ == '__main__':
    #trainMagicPoint(sys.argv[1],True,'model/model_'+sys.argv[2]+'.ckpt',sys.argv[2])
    #testMagicPoint(sys.argv[1],'model/model_'+sys.argv[2]+'.ckpt')
    #testMagicPointForAImg(sys.argv[1],'model/model_'+sys.argv[2]+'.ckpt')
    testMagicPointForAFolder(sys.argv[1],'model/model_'+sys.argv[2]+'.ckpt')
    #img = cv2.imread('2_test.png',0)
    #findPoint(img)