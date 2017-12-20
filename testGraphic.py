#not four corner in each picture

import cv2
import numpy as np
import random

if __name__ == '__main__':
    a = []
    a.append([10,10])
    a.append([100,10])
    a.append([100,100])
    a.append([10,100])
    a = np.array(a)
    im = np.ones([120, 160], dtype = np.uint8)*255
    cv2.fillConvexPoly(im, a, random.random()*0)
    cv2.imwrite('test.png',im)