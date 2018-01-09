import cv2
import numpy as np
import random
import sys
import os
import time

if __name__ == '__main__':
    os.system('python estimate.py fastkps fastRes')
    os.system('python estimate.py harriskps harrisRes')
    os.system('python estimate.py siftkps siftRes')
    os.system('python estimate.py surfkps surfRes')
    os.system('python estimate.py orbkps orbRes')
    