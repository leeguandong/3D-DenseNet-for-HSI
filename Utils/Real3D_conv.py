# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from keras.models import Sequential
from keras import Input
from keras.layers import Convolution2D, MaxPooling2D, Convolution3D, MaxPooling3D,Conv3D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.utils.np_utils import to_categorical
from sklearn.decomposition import PCA
from keras.optimizers import Adam, SGD, Adadelta
import keras.callbacks as kcallbacks
from keras.regularizers import l2
import time
from Utils import zeroPadding
from Utils import normalization
from Utils import doPCA
import collections
from sklearn import metrics
from Utils import modelStatsRecord
from Utils import averageAccuracy
from Utils import sampleFixNum

def indexToAssignment(index_, Row, Col, pad_length):
    new_assign = {}
    for counter, value in enumerate(index_):
        assign_0 = value // Col + pad_length
        assign_1 = value % Col + pad_length
        new_assign[counter] = [assign_0, assign_1]
    return new_assign

def assignmentToIndex( assign_0, assign_1, Row, Col):
    new_index = assign_0 * Col + assign_1
    return new_index

def selectNeighboringPatch(matrix, pos_row, pos_col, ex_len):
    selected_rows = matrix[range(pos_row-ex_len,pos_row+ex_len+1), :, :]
    selected_patch = selected_rows[:, range(pos_col-ex_len, pos_col+ex_len+1), :]
    return selected_patch

def sampling(proptionVal, groundTruth):              #divide dataset into train and test datasets
    labels_loc = {}
    train = {}
    test = {}
    m = max(groundTruth)
    for i in range(m):
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1]
        np.random.shuffle(indices)
        labels_loc[i] = indices
        nb_val = int(proptionVal * len(indices))
        train[i] = indices[:-nb_val]
        test[i] = indices[-nb_val:]
#    whole_indices = []
    train_indices = []
    test_indices = []
    for i in range(m):
#        whole_indices += labels_loc[i]
        train_indices += train[i]
        test_indices += test[i]
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    return train_indices, test_indices

best_weights_path = 'D:/Tensorflow  Learning/SSRN-master/SSRN-master/Best_models/Indian_best_3DCONV.hdf5'

mat_data = sio.loadmat('D:/Tensorflow  Learning/SSRN-master/SSRN-master/datasets/IN/Indian_pines_corrected.mat')
data_IN = mat_data['indian_pines_corrected']
mat_gt = sio.loadmat('D:/Tensorflow  Learning/SSRN-master/SSRN-master/datasets/IN/Indian_pines_gt.mat')
gt_IN = mat_gt['indian_pines_gt']
print (data_IN.shape)

batch_size = 16
nb_classes = 16
nb_epoch = 400
INPUT_DIMENSION = 200
VALIDATION_SPLIT = 0.9
PATCH_LENGTH = 5                  #Patch_size (13*2+1)*(13*2+1)

data = data_IN.reshape(np.prod(data_IN.shape[:2]),np.prod(data_IN.shape[2:]))
gt = gt_IN.reshape(np.prod(gt_IN.shape[:2]),)

#data = normalization.Normalization(data)

#data_ = data.reshape(data_UP.shape[0], data_UP.shape[1],data_UP.shape[2])
# data_trans = data.transpose()
# whole_pca = doPCA.dimension_PCA(data_trans, data_UP, INPUT_DIMENSION)
#
# print (whole_pca.shape)

data_trans = data.transpose()
data_ = doPCA.dimension_PCA(data_trans, data_IN, INPUT_DIMENSION)

padded_data = zeroPadding.zeroPadding_3D(data_, PATCH_LENGTH)
print (padded_data.shape)


ITER = 1
CATEGORY = 16

KAPPA_CONV = []
OA_CONV = []
AA_CONV = []
TRAINING_TIME_CONV = []
TESTING_TIME_CONV = []
ELEMENT_ACC_CONV = np.zeros((ITER, CATEGORY))

for index_iter in range(ITER):
    print("# %d Iteration" % (index_iter + 1))

    millis = int(round(time.time()) * 1000) % 4294967295
    np.random.seed(millis)

    #train_indices, test_indices = sampling(VALIDATION_SPLIT, gt)
    train_indices, test_indices = sampleFixNum.samplingDiffFixedNum([30, 150, 150, 100, 150, 150, 20, 150, 15, 150, 150, 150, 150, 150, 50, 50], gt)

    y_train = gt[train_indices] - 1
    y_train = to_categorical(np.asarray(y_train))

    y_test = gt[test_indices] - 1
    y_test = to_categorical(np.asarray(y_test))

    #first principal component training data
    train_assign = indexToAssignment(train_indices, data_.shape[0], data_.shape[1], PATCH_LENGTH)
    train_data = np.zeros((len(train_assign), 2*PATCH_LENGTH + 1, 2*PATCH_LENGTH + 1, INPUT_DIMENSION))
    for i in range(len(train_assign)):
        train_data[i] = selectNeighboringPatch(padded_data,train_assign[i][0],train_assign[i][1],PATCH_LENGTH)

    #first principal component testing data
    test_assign = indexToAssignment(test_indices, data_.shape[0], data_.shape[1], PATCH_LENGTH)
    test_data = np.zeros((len(test_assign), 2*PATCH_LENGTH + 1, 2*PATCH_LENGTH + 1, INPUT_DIMENSION))
    for i in range(len(test_assign)):
        test_data[i] = selectNeighboringPatch(padded_data,test_assign[i][0],test_assign[i][1],PATCH_LENGTH)

    x_train = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2], INPUT_DIMENSION)
    x_test = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2], INPUT_DIMENSION)

    # x_val = x_test[-1765:]
    # y_val = y_test[-1765:]

    # ################################################################################################
    # 2D
    # model = Sequential()
    #
    # # model.add(Convolution2D(32, 1, 1, input_shape=(27, 27, INPUT_DIMENSION)))
    # # model.add(BatchNormalization())
    # # model.add(Activation('relu'))
    #
    # model.add(Convolution2D(32, 4, 4, input_shape=(27, 27, INPUT_DIMENSION)))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # model.add(Convolution2D(64, 5, 5))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # model.add(Convolution2D(128, 4, 4))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    #
    # model.add(Flatten())
    # model.add(Dense(16, W_regularizer=l2(0.01)))                                         #number of classes
    # model.add(Activation('sigmoid'))

    # #########################################################################
    # 3D
    model = Sequential()
    # 两层，第一层去整个图像，第二层每次只取一个7*7，对7*7进行处理
    model.add(Convolution3D(16,kernel_size = (3, 3, 20), strides = (1, 1, 10),input_shape = (11, 11, INPUT_DIMENSION,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 1)))

    model.add(Convolution3D(32, kernel_size=(2, 2, 3), strides=(1, 1, 1),))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(1, 1, 1)))

    model.add(Convolution3D(32, kernel_size=(1, 1, 3), strides=(1, 1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(16, W_regularizer=l2(0.01)))                                         #number of classes
    model.add(Activation('sigmoid'))

    adam = Adam(lr=0.003)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    # 选择最优模型
    earlyStopping = kcallbacks.EarlyStopping(monitor='val_loss', patience=100, verbose=1, mode='auto')
    saveBestModel = kcallbacks.ModelCheckpoint(best_weights_path, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

    # model = cnn_3d((1,27,27,200))
    # 训练
    tic2 = time.clock()
    history_conv = model.fit(x_train, y_train, validation_data=(x_test, y_test),  batch_size=batch_size, nb_epoch=nb_epoch, shuffle=True ,callbacks=[earlyStopping, saveBestModel])
    toc2 = time.clock()

    # 评估
    tic3 = time.clock()
    loss_and_metrics = model.evaluate(x_test, y_test, batch_size=256)
    toc3 = time.clock()

    print('3D CONV Training Time: ', toc2 - tic2)
    print('3D CONV Test time:', toc3 - tic3)

    print('3D CONV Test score:', loss_and_metrics[0])
    print('3D CONV Test accuracy:', loss_and_metrics[1])

    print(history_conv.history.keys())

    # 预测
    pred_test_conv = model.predict(x_test).argmax(axis=1)
    collections.Counter(pred_test_conv)

    gt_test = gt[test_indices] - 1
    overall_acc_conv = metrics.accuracy_score(pred_test_conv, gt_test)
    confusion_matrix_conv = metrics.confusion_matrix(pred_test_conv, gt_test)
    each_acc_conv, average_acc_conv = averageAccuracy.AA_andEachClassAccuracy(confusion_matrix_conv)
    kappa = metrics.cohen_kappa_score(pred_test_conv, gt_test)
    KAPPA_CONV.append(kappa)
    OA_CONV.append(overall_acc_conv)
    AA_CONV.append(average_acc_conv)
    TRAINING_TIME_CONV.append(toc2 - tic2)
    TESTING_TIME_CONV.append(toc3 - tic3)
    ELEMENT_ACC_CONV[index_iter, :] = each_acc_conv

    print ("Overall Accuracy:", overall_acc_conv)
    print ("Confusion matrix:", confusion_matrix_conv)
    print ("Average Accuracy:", average_acc_conv)

    print ("Each Class Accuracies are listed as follows:")
    for idx, acc in enumerate(each_acc_conv):
        print ("Class %d : %.3e" % (idx + 1, acc))

    print("3D CONV training finished.")

modelStatsRecord.outputStats(KAPPA_CONV, OA_CONV, AA_CONV, ELEMENT_ACC_CONV, TRAINING_TIME_CONV, TESTING_TIME_CONV,
                             history_conv, loss_and_metrics, CATEGORY, 'D:/Tensorflow  Learning/SSRN-master/SSRN-master/Real3D_conv/record_3D_CONV_IN.txt',
                             'D:/Tensorflow  Learning/SSRN-master/SSRN-master/Real3D_conv/element_acc_3D_CONV_IN.txt')