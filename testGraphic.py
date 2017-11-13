#not four corner in each picture

import cv2
import numpy as np
import random

if __name__ == '__main__':
    a = []
    for i in range(4):
        x = int(random.random()*160)
        y = int(random.random()*120)
        a.append([x,y])
    a = np.array(a)
    im = np.zeros([120, 160], dtype = np.uint8)
    cv2.fillConvexPoly(im, a, random.random()*200+55)
    cv2.imwrite('test.png',im)