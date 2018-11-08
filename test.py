import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import os
import cv2
import pandas as pd

index = 0
image = cv2.imread( "IsolationForest/msrc/image7/labels/"+str(index)+'.png')
image = cv2.cvtColor( image, cv2.COLOR_BGR2GRAY )
image_resize = cv2.resize( image, (28,28) )


num = np.array( [1,2,3,4,5] )
print( num[0:3])

num = np.zeros( 3, dtype=np.float32 )
num[0] += 1
print( num )

num = np.zeros( 5 )
num2 = np.ones( (2,2,5) )
num = num2[0,0,:]
print( num )

num = np.zeros( (3,3), dtype = np.uint8 )
num[0,:] = 1
num[1,:] = 50
num[2,:] = 101

print( num.shape  )
num = cv2.resize( num, (2,2) )
print( num )

label_image = cv2.imread( "IsolationForest/cosal/image2/labels/"+str(index)+'.png' ) #超像素的值,就是相当于我们把他保存了起来
label_image = cv2.cvtColor( label_image, cv2.COLOR_BGR2GRAY )
new_label_image = cv2.resize( label_image, (28,28) )
np.savetxt( "label.txt", label_image )
np.savetxt( "new_label.txt", new_label_image )