import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import os
import cv2
import pandas as pd
import inspect
import tensorflow as tf
import time
import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()

"""
这部分是我用ground_truth本身去统计特异点
边界点我们认为他不是显著性信息。
重要变量：
    num:numpy数组，记录了我们超像素分割后的信息值，其实就是一堆111，222，333变量值。
    co1，row:其实就是我们num数组的shape
    groundtruth:我们该标签下的groundtruth值
    num_diff: 统计所有特异点的总数（ 运用ground truth去计算 ）
    num_label: 统计所有标签的总数，其实就是我们的样本点

"""

class Vgg16:
    def __init__(self, vgg16_npy_path=None):
        if vgg16_npy_path is None:
            path = inspect.getfile(Vgg16)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "vgg16.npy")
            vgg16_npy_path = path
            print( path )

        # 加载网络权重参数
        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        print("npy file loaded")

    def build(self, bgr):
        """
        load variable from npy to build the VGG
        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """

        start_time = time.time()
        print("build model started")

        self.conv1_1 = self.conv_layer(bgr, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')

        print("build model finished: %ds" % (time.time() - start_time))

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):

            filt = self.get_conv_filter(name)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")

gt_image_path = "IsolationForest/gt"
label_path = "IsolationForest/labels"
feat_path = "IsolationForest/feature"

index = 0
gt_diff = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
           [], [], [], [], [], [], [], [], []]  # 统计每个标签下的特异点。
tot_label = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
             [], [], [], [], [], [], [], [], []]  # 统计每个标签下的边界的像素点

num_tot_diff = 0
num_tot_label = 0

x_train = np.zeros( (1, 8), dtype=np.float32 )
x_train_vgg = np.zeros( (1,259), dtype=np.float32 )

while index <= 19:

    label_name = os.path.join(label_path, str(index) + '.txt')
    gt_name = os.path.join(gt_image_path, str(index) + '.png')
    feat_name = os.path.join(feat_path, 'feat_' + str(index) + '.txt')

    diff = []  # 统计当前标签下的特异点信息

    #################统计当前标签下超像素分割的大小。该模块的作用就是我们的超像素分割保存的值都是在我们txt文件中，我们要把这部分信息转换成我们的numpy数组。

    row = 0
    col = 0
    with open(label_name, "r") as f:

        lines = f.readlines()
        for line in lines:
            col = 0
            row += 1
            length = len(line)
            for i in range(length):
                if line[i] == " " or line[i] == '\n':
                    col += 1

    num = np.zeros((row, col), dtype=np.int32)

    row = 0
    col = 0

    with open(label_name, "r") as f:

        lines = f.readlines()
        for line in lines:
            col = 0
            length = len(line)
            str1 = ""
            for i in range(length):
                if line[i] == " " or line[i] == '\n':
                    label = int(str1)
                    num[row, col] = label
                    col += 1
                    str1 = ""
                else:
                    str1 += line[i]
            row += 1

    cv2.imwrite(label_path + '/' + str(index) + '.png', num)  # 我们将我们的从txt文件保存的信息顺便就写入到我们的图片信息里面去
    ######################################################################

    ##################这部分我们开始统计我们的txt文件里的数据和我们的groundtruth的奇异点
    """
    这部分我们干了三件事：
        1：当前index下，我们的特异点有哪些，存在了数组diff里面
        2：统计所有index下diff信息，其实就是相当于将diff信息，加了一维，存在了gt_diff里面
        3：统计当前index下一共有多少标签数，存在了tot_label里面
    """

    groundtruth = cv2.imread(gt_name)
    groundtruth = cv2.cvtColor(groundtruth, cv2.COLOR_BGR2GRAY)
    # print(groundtruth.shape)

    ##### 统计第一行特异点信息

    now = num[0, 0]  # 现在我们统计的点的标签值

    black = 0
    white = 0
    col1 = 1

    if groundtruth[0, 0] == 0:  # 该点的像素值是黑色
        black += 1
    else:
        white += 1

    while col1 < col:

        pixel_value = groundtruth[0, col1]  # 第0行,col1列的像素值
        temp = num[0, col1]  # 第0行,col1列的标签值

        if temp == now:  # 依旧是当前标签值
            if pixel_value == 0:
                black += 1
            else:
                white += 1

        elif temp != now:  # 标签值发生了改变

            if black < white:
                diff.append(now)  # 这是特异值标签
                gt_diff[index].append(now)  # 记录当前标签下特异值的标签

            tot_label[index].append(now)  # 记录下改index下所有的标签值

            now = temp
            black = 0
            white = 0
            if pixel_value == 0:
                black += 1
            else:
                white += 1
        col1 += 1

    #######  最后一行

    now = num[row - 1, 0]

    black = 0
    white = 0
    col1 = 1

    if groundtruth[row - 1, 0] == 0:
        black += 1
    else:
        white += 1

    while col1 < col:

        pixel_value = groundtruth[row - 1, col1]
        temp = num[row - 1, col1]
        if temp == now:

            if pixel_value == 0:
                black += 1
            else:
                white += 1

        elif temp != now:

            if black < white:
                diff.append(now)
                gt_diff[index].append(now)  # 记录当前标签下特异值的标签

            tot_label[index].append(now)  # 记录下改index下所有的标签值
            now = temp
            black = 0
            white = 0

            if pixel_value == 0:
                black += 1
            else:
                white += 1
        col1 += 1

    ############# 第一列

    black = 0
    white = 0
    now = num[0, 0]
    row1 = 1

    if groundtruth[0, 0] == 0:
        black += 1
    else:
        white += 1

    while row1 < row:

        pixel_value = groundtruth[row1, 0]
        temp = num[row1, 0]

        if temp == now:

            if pixel_value == 0:
                black += 1
            else:
                white += 1

        elif temp != now:

            if black < white:
                diff.append(now)
                gt_diff[index].append(now)  # 记录当前标签下特异值的标签

            tot_label[index].append(now)  # 记录下改index下所有的标签值

            now = temp
            black = 0
            white = 0

            if pixel_value == 0:
                black += 1
            else:
                white += 1
        row1 += 1

    ##### 最后一列

    black = 0
    white = 0
    now = num[0, col - 1]
    row1 = 1

    if groundtruth[0, col - 1] == 0:
        black += 1
    else:
        white += 1

    while row1 < row:

        pixel_value = groundtruth[row1, col - 1]
        temp = num[row1, col - 1]

        if temp == now:
            if pixel_value == 0:
                black += 1
            else:
                white += 1

        elif temp != now:
            if black < white:
                diff.append(now)
                gt_diff[index].append(now)  # 记录当前标签下特异值的标签

            tot_label[index].append(now)  # 记录下改index下所有的标签值

            now = temp
            black = 0
            white = 0

            if pixel_value == 0:
                black += 1
            else:
                white += 1
        row1 += 1

    print("diff的总数", len(diff))
    print("%d index下的gt_diff" % index, len(gt_diff[index]))
    print("%d index下的tot_label" % index, len(tot_label[index]))

    diff = list(set(diff))
    gt_diff[index] = list(set(gt_diff[index]))
    tot_label[index] = list(set(tot_label[index]))

    print("去重后 diff的总数", len(diff))
    print("去重后 %d index下的gt_diff" % index, len(gt_diff[index]))
    print("去重后 %d index下的tot_label" % index, len(tot_label[index]))

    num_tot_diff += len(diff)
    num_tot_label += len(tot_label[index])
    print("num_tot_diff总数", num_tot_diff)
    print("num_tot_label总数", num_tot_label)

    diff.sort()
    gt_diff[index].sort()
    tot_label[index].sort()

    ############################################################################

    ################ 统计我们所有图片一周的特征值####################################

    """
    在我们之前的运算之中，我们每一个标签的特征值是8位，那么在我们的复杂背景中，可能会出现问题，所以我们这里提出使用vgg去提取我们的特征。
    每一个标签的特征值其实就是我们的原特征+vgg16的特征( 1*1*259 )
    这部分我们做的一件事就是使用我们的vgg16去提取我们的特征。
    
    """

    label_image = cv2.imread( "IsolationForest/cosal/image7/labels/"+str(index)+'.png' ) #超像素的值,就是相当于我们把他保存了起来
    origin_image = cv2.imread( "IsolationForest/cosal/image7/image/"+str(index)+'.bmp' ) #我们保存的图像

    origin_image1 = cv2.resize( origin_image,(224,224) )

    origin_image = origin_image.astype( np.float32 )

    new_label_image = cv2.resize( label_image, (28,28) ) #可以resize，信息没有丢失

    new_label_image = cv2.cvtColor( new_label_image,cv2.COLOR_BGR2GRAY ) #高层次后的信息,就是每一个位置的标签信息。

    inputs = origin_image1
    inputs = inputs.astype( np.float32 )
    inputs = np.reshape( inputs,newshape=(1,224,224,3) )

    vgg = Vgg16()
    vgg.build(inputs)

    feature_image = vgg.pool3

    feature_image1 = np.zeros( (28,28,256), dtype=np.float32 )

    row1 = 0
    col1 = 0

    #print( "赋值前feature_image的值", feature_image1[0,0,255] )
    #print("赋值前feature_image的值", feature_image1[0, 0, 200])
    for row1 in range(28):
        for col1 in range(28):
            feature_image1[row1,col1,:] = feature_image[0,row1,col1,:]

    #print( "赋值后feature_image的值", feature_image1[0,0,255] )
    #print("赋值后feature_image的值", feature_image1[0, 0, 200])

    len_tot_label = len( tot_label[index] ) #当前index下我们边界一周的所有的标签值
    #print( "index %d 边界一周的值" %index,len_tot_label )

    i = 0
    while i < len_tot_label: #我们开始遍历我们搜集到的当前标签下的一周的标签值

        pixel_value = tot_label[index][i] #我们当前的标签值

        temp = np.zeros( 3, dtype=np.float32 )

        num_of_this_pixel = 0

        #写这段代码的意思就是我们已经搜集到了当前一周的标签值了，我们遍历我们的数组，看看一共有多少当前的标签值
        row1 = 0
        col1 = 0

        for row1 in range(row): #原始的三层特征
            for col1 in range(col):
                if num[row1,col1] == pixel_value:
                    num_of_this_pixel += 1
                    temp[0] += origin_image[row1,col1,0 ]
                    temp[1] += origin_image[row1,col1,1 ]
                    temp[2] += origin_image[row1,col1,2 ]

        if num_of_this_pixel == 0:
            num_of_this_pixel = 1

        temp = temp / num_of_this_pixel

        temp1 = np.zeros( 256, dtype=np.float32 )
        num_of_this_pixel = 0

        row2 = 0
        col2 = 0
        for row2 in range( 28 ):
            for col2 in range( 28 ): #高层特征
                if new_label_image[row2,col2] == pixel_value:
                    temp1 += feature_image1[row2,col2,:]
                    num_of_this_pixel += 1

        #print("num_of_this_pixel", num_of_this_pixel)
        if num_of_this_pixel == 0:
            num_of_this_pixel = 1

        temp1 = temp1 / num_of_this_pixel

        temp = np.reshape( temp, newshape=(1,3) )
        temp1 = np.reshape( temp1, newshape=(1,256) )
        temp2 = np.concatenate( (temp,temp1), axis=1 )

        if i == 0 and index == 0:
            x_train_vgg = temp2
        else:
            x_train_vgg = np.concatenate( (x_train_vgg, temp2), axis=0 )
        #print( "temp.shape", temp.shape )
        #print( "temp1.shape", temp1.shape )
        #print( "temp2.shape", temp2.shape )

        i += 1


    ####################################################################################


    index += 1

print( x_train_vgg.shape )
X_train = x_train_vgg

rng = np.random.RandomState(42)
isofortrain = IsolationForest(n_estimators=1000,
                              max_samples='auto',
                              contamination=.20,
                              max_features=1,
                              random_state=rng,
                              n_jobs=-1)

isofortrain.fit(X_train)
anomalytrain = isofortrain.decision_function(X_train)
predicttrain = isofortrain.predict(X_train)

len_predictrain = len(predicttrain)
print("len_predictrain", len_predictrain)

num_iforest_diff = 0

for i in predicttrain:
    if i == -1:
        num_iforest_diff += 1

print("num_iforest_diff", num_iforest_diff)

same = 0
index = 0
k = 0

while index <= 19:

    len_tot_label = len(tot_label[index])  # 当前index下的tot_label数量
    i = 0
    while i < len_tot_label:  # 遍历当前index下所有的值

        now_pred_value = predicttrain[k]  # tot_label[index][i]这个点，是否是奇异值由我们的predicttrain[k]来决定
        k += 1
        if (now_pred_value == -1):  # 说明这个点是奇异值

            now = tot_label[index][i]
            for z in gt_diff[index]:
                if z == now:
                    same += 1
        i += 1
    index += 1

print("same值", same)

