import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import os
import cv2
import pandas as pd

images = []
gt = []

for file in os.listdir( "7/"):
    file = str( file )
    images.append( file )

for file in os.listdir( "gt7/"):
    file = str( file )
    gt.append( file )

images.sort()
gt.sort()

num = len( gt )

for index in range( num ):

    gt_name = gt[index]
    len1 = len( gt_name)
    image_name = gt_name[0:len1-4] + '.jpg'

    origin_image = cv2.imread( "7/"+image_name )
    gt_image = cv2.imread("gt7/"+gt_name)

    cv2.imwrite("image/"+str(index)+'.bmp', origin_image )
    cv2.imwrite("gt/"+str(index)+'.png', gt_image )