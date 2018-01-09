import cv2
import numpy as np
import random
import sys
import os

#evaluate feature points
#1. stability of feature points quantity
#2. repeatability of feature points

threshold = 4

def fileToData(lines):
    data = []
    for line in lines:
        data.append([float(line.split(',')[0]),float(line[:-1].split(',')[1])])
    data = sorted(data)
    return data

def repeatbilityRate(dataRef,dataCur):
    pointRepeat = 0
    for pointRef in dataRef:
        for pointCur in dataCur:
            if (pointCur[0] < pointRef[0] - threshold) | (pointCur[0] > pointRef[0] + threshold):
                continue
            elif (pointCur[1] < pointRef[1] - threshold) | (pointCur[1] > pointRef[1] + threshold):
                continue
            elif ((pointCur[0] - pointRef[0]) ** 2 + (pointCur[1] - pointRef[1]) ** 2) < threshold ** 2:
                pointRepeat += 1
                break
    return pointRepeat / float(len(dataRef))


def estimateQuantityAndQuality(dataRef,dataCur):
    #1. stability of feature points quantity
    quantityStability = abs(len(dataRef)-len(dataCur))/float(len(dataRef))
    #2. repeatability of feature points
    repeatRate = repeatbilityRate(dataRef,dataCur)
    return quantityStability,repeatRate

def key(name):
    value = 0
    if name.split('_')[-1][0:4] == 'blur':
        value += 100
        return value + int(name.split('_')[-1].split('.')[0][4:])
    elif name.split('_')[-1][0:5] == 'noise':
        value += 200
        return value + int(name.split('_')[-1].split('.')[0][5:])


if __name__ == '__main__':
    imgNames = os.listdir(sys.argv[1])
    reference = open(sys.argv[1] + '/' + imgNames[0],'r')
    os.system('mkdir ' + sys.argv[2])
    dataRef = fileToData(reference.readlines())
    f = open(sys.argv[2] + '/result.txt','w')
    count = 0
    #sort imgNames
    imgNames = imgNames[1:]
    imgNames = sorted(imgNames,key=key)
    for imgName in imgNames:
        fr = open(sys.argv[1] + '/' + imgName)
        dataCur = fileToData(fr.readlines())
        quantityStability,repeatRate = estimateQuantityAndQuality(dataRef,dataCur)
        f.write(imgName + ' : ')
        f.write(str(quantityStability)+',')
        f.write(str(repeatRate))
        f.write('\n')
        fr.close()
        count += 1
        print(count)
    f.close()