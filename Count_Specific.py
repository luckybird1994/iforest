import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import os
import cv2
import pandas as pd

"""
这部分是我用ground_truth本身去统计特异点
边界点我们认为他不是显著性信息。
重要变量：
    num:numpy数组，记录了我们超像素分割后的信息值，其实就是一堆111，222，333变量值。
    co1，row:其实就是我们num数组的shape
    groundtruth:我们该标签下的groundtruth值
    num_diff: 统计所有特异点的总数（ 运用ground truth去计算 ）
    num_label: 统计所有标签的总数，其实就是我们的样本点

运用iforest的目的:
在我们做显著性检测的时候，我们在做边缘检测的时候，我们的边缘应该不算在显著性目标里面，就是他的背景颜色应该是黑色的。如果我们检测的时候，把
这部分检测出来了，就是检测成白色的了，我们就应该把他认为是特异点。
现在我们做完显著性检测后，检测边缘一共有多少特异点？我们不妨首先把周围一圈的所有值都统计出来，然后送到iforest里面去，去检测我们的特异点。
当然，对于我们这个工作，我们使用的超像素分割去先对原图进行特征处理，这样每一个像素的标签值都有一个对应点，每一个标签值又有他的特征值，我们就对
这些特征值进行建模然后使用iforest去检测特异点。
"""

gt_image_path = "IsolationForest/gt"
label_path = "IsolationForest/labels"
feat_path = "IsolationForest/feature"

index = 0
gt_diff = [ [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[] ] #统计每个标签下的特异点。
tot_label = [ [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[] ] #统计每个标签下的边界的像素点

num_tot_diff = 0
num_tot_label = 0

x_train = np.zeros( (1,8), dtype = np.float32 )

while index <= 19:

    label_name = os.path.join(label_path, str(index) + '.txt')
    gt_name = os.path.join(gt_image_path, str(index) + '.png')
    feat_name = os.path.join( feat_path, 'feat_'+str(index)+'.txt' )

    diff = [] #统计当前标签下的特异点信息

    #################统计当前标签下超像素分割的大小。该模块的作用就是我们的超像素分割保存的值都是在我们txt文件中，我们要把这部分信息转换成我们的numpy数组。

    row = 0
    col = 0
    with open( label_name, "r" ) as f:

        lines = f.readlines()
        for line in lines:
            col = 0
            row += 1
            length = len( line )
            for i in range( length ):
                if line[i] == " " or line[i] == '\n':
                    col += 1

    num = np.zeros((row, col), dtype=np.int32)

    row = 0
    col = 0

    with open( label_name, "r" ) as f:

        lines = f.readlines()
        for line in lines:
            col = 0
            length = len( line )
            str1 = ""
            for i in range( length ):
                if line[i] == " " or line[i] == '\n':
                    label = int( str1 )
                    num[row,col] = label
                    col += 1
                    str1 = ""
                else:
                    str1 += line[i]
            row += 1

    #print( num[row-1,col-1] )
    cv2.imwrite( label_path +'/'+str(index)+'.png', num ) #我们将我们的从txt文件保存的信息顺便就写入到我们的图片信息里面去
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
    #print(groundtruth.shape)

    ##### 统计第一行特异点信息

    now = num[0, 0] #现在我们统计的点的标签值

    black = 0
    white = 0
    col1 = 1

    if groundtruth[0,0] == 0: #该点的像素值是黑色
        black += 1
    else:
        white += 1

    while col1 < col:

        pixel_value = groundtruth[0,col1] #第0行,col1列的像素值
        temp = num[0,col1] #第0行,col1列的标签值

        if temp == now: #依旧是当前标签值
            if pixel_value == 0:
                black += 1
            else:
                white += 1

        elif temp != now: #标签值发生了改变

            if black < white:
                diff.append( now ) #这是特异值标签
                gt_diff[index].append( now ) #记录当前标签下特异值的标签

            tot_label[index].append( now ) #记录下改index下所有的标签值

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

    print( "diff的总数", len( diff) )
    print( "%d index下的gt_diff" %index, len( gt_diff[index]) )
    print( "%d index下的tot_label" %index, len( tot_label[index]) )

    diff = list( set( diff ) )
    gt_diff[index] = list( set(gt_diff[index] ) )
    tot_label[index] = list( set( tot_label[index] ) )

    print("去重后 diff的总数", len(diff))
    print("去重后 %d index下的gt_diff" % index, len(gt_diff[index]))
    print("去重后 %d index下的tot_label" % index, len(tot_label[index]))

    num_tot_diff += len( diff )
    num_tot_label += len(tot_label[index])
    print( "num_tot_diff总数", num_tot_diff )
    print( "num_tot_label总数", num_tot_label )

    diff.sort()
    gt_diff[index].sort()
    tot_label[index].sort()

    ############################################################################

    ################ 统计我们所有图片一周的特征值####################################

    """
    我们每一个index下四周的超像素分割的值都已经存储在我们的tot_label里面了，我们现在就根据他们各自的特征值信息把所有的特征值都存到我们x_train里面
    """

    len_tot_label = len( tot_label[index] )
    i = 0

    while i < len_tot_label: #搜集当前index下所有超像素的特征值

        if index == 0 and i == 0: #我们的x_train相当于我们的要送到iforest里面训练的值，我们要不断得去concat操作，所以一开始要特判断.

            label_value = tot_label[index][i] #当前index下的超像素的值
            #print( "%d index下的超像素标签值" %index, label_value )

            with open(feat_name, "r") as f:

                lines = f.readlines()

                line = lines[label_value - 1]
                #print( "%d index下的line像素值" %index, line ) #这些值是我们读的txt文件的值

                len_line = len(line)

                str1 = ""
                flag = 0
                k = 0

                for j in range(len_line):

                    if (line[j] == ' ' or line[j] == '\n'):
                        if flag == 1:

                            value = float(str1)
                            x_train[0][k] = value
                            k += 1
                            flag = 0
                            str1 = ""

                    else:
                        flag = 1
                        str1 += line[j]

                #print("%d index下的line像素值" % index, x_train)

        else:

            label_value = tot_label[index][i]  # 当前index下的超像素的值
            #print("%d index下的超像素标签值" % index, label_value)

            with open(feat_name, "r") as f:

                lines = f.readlines()
                line = lines[label_value - 1]
                #print("%d index下的line像素值" % index, line)

                len_line = len(line)
                temp1 = np.zeros((1, 8), dtype=np.float32)

                str1 = ""
                flag = 0
                k = 0

                for j in range(len_line):

                    if (line[j] == ' ' or line[j] == '\n'):
                        if flag == 1:

                            value = float(str1)
                            temp1[0][k] = value
                            k += 1
                            flag = 0
                            str1 = ""

                    else:
                        flag = 1
                        str1 += line[j]

                #print("%d index下的line像素值" % index, temp1)
                x_train = np.concatenate( (x_train,temp1), axis=0 )

        i += 1

    #########################################################################

    index += 1

print( x_train.shape )
X_train = x_train

rng = np.random.RandomState(42)
isofortrain = IsolationForest(n_estimators = 1000,
                             max_samples = 'auto',
                             contamination = .20,
                             max_features = 1,
                             random_state = rng,
                             n_jobs = -1)

isofortrain.fit(X_train)
anomalytrain = isofortrain.decision_function(X_train)
predicttrain = isofortrain.predict(X_train)

len_predictrain = len( predicttrain )
print( "len_predictrain", len_predictrain )

num_iforest_diff = 0

for i in predicttrain:
    if i == -1:
        num_iforest_diff += 1

print( "num_iforest_diff",num_iforest_diff )


same = 0
index = 0
k = 0

while index <= 19:

    len_tot_label = len( tot_label[index] ) #当前index下的tot_label数量
    i = 0
    while i < len_tot_label: #遍历当前index下所有的值

        now_pred_value = predicttrain[k] #tot_label[index][i]这个点，是否是奇异值由我们的predicttrain[k]来决定
        k += 1
        if( now_pred_value == -1 ): #说明这个点是奇异值

            now = tot_label[index][i]
            for z in gt_diff[index]:
                if z == now:
                    same += 1
        i += 1
    index += 1

print( "same值",same )



