# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from keras.models import Sequential, Model
from keras.layers import Convolution2D, MaxPooling2D, Conv3D, MaxPooling3D, ZeroPadding3D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization, Input
from keras.utils.np_utils import to_categorical
from sklearn.decomposition import PCA
from keras.optimizers import Adam, SGD, Adadelta, RMSprop, Nadam
import keras.callbacks as kcallbacks
from keras.regularizers import l2
import time
from Utils import zeroPadding, normalization, doPCA, modelStatsRecord, averageAccuracy,  \
    densenet_UP,cnn_3D_UP

import collections
from sklearn import metrics, preprocessing
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def indexToAssignment(index_, Row, Col, pad_length):
    new_assign = {}
    for counter, value in enumerate(index_):
        assign_0 = value // Col + pad_length
        assign_1 = value % Col + pad_length
        new_assign[counter] = [assign_0, assign_1]
    return new_assign

def assignmentToIndex(assign_0, assign_1, Row, Col):
    new_index = assign_0 * Col + assign_1
    return new_index

def selectNeighboringPatch(matrix, pos_row, pos_col, ex_len):
    selected_rows = matrix[range(pos_row - ex_len, pos_row + ex_len + 1), :]
    selected_patch = selected_rows[:, range(pos_col - ex_len, pos_col + ex_len + 1)]
    return selected_patch

def sampling(proptionVal, groundTruth):  # divide dataset into train and test datasets
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
    # whole_indices = []
    train_indices = []
    test_indices = []
    for i in range(m):
        #        whole_indices += labels_loc[i]
        train_indices += train[i]
        test_indices += test[i]
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    return train_indices, test_indices

def model_DenseNet():
    model_dense = cnn_3D_UP.ResnetBuilder.build_resnet_8((1, img_rows, img_cols, img_channels), nb_classes)

    RMS = RMSprop(lr=0.0003)
    # Let's train the model using RMSprop
    model_dense.compile(loss='categorical_crossentropy', optimizer=RMS, metrics=['accuracy'])

    return model_dense

uPavia = sio.loadmat('D:/Tensorflow  Learning/3D-DenseNet-Hyperspectral/datasets/UP/PaviaU.mat')
gt_uPavia = sio.loadmat('D:/Tensorflow  Learning/3D-DenseNet-Hyperspectral/datasets/UP/PaviaU_gt.mat')
data_UP = uPavia['paviaU']
gt_UP = gt_uPavia['paviaU_gt']
print(data_UP.shape)

# new_gt_UP = set_zeros(gt_UP, [1,4,7,9,13,15,16])
new_gt_UP = gt_UP

batch_size = 16
nb_classes = 9
nb_epoch = 200  # 400
img_rows, img_cols = 11, 11  # 27, 27
patience = 200

INPUT_DIMENSION_CONV = 103
INPUT_DIMENSION = 103

# 10%:10%:80% data for training, validation and testing

TOTAL_SIZE = 42776
VAL_SIZE = 4281
TRAIN_SIZE = 4281
TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE

img_channels = 103
VALIDATION_SPLIT = 0.90  # 10% for training and %90 for validation and testing

img_channels = 103
PATCH_LENGTH = 5  # Patch_size (13*2+1)*(13*2+1)

data = data_UP.reshape(np.prod(data_UP.shape[:2]), np.prod(data_UP.shape[2:]))
gt = new_gt_UP.reshape(np.prod(new_gt_UP.shape[:2]), )

data = preprocessing.scale(data)

# scaler = preprocessing.MaxAbsScaler()
# data = scaler.fit_transform(data)

data_ = data.reshape(data_UP.shape[0], data_UP.shape[1], data_UP.shape[2])
whole_data = data_
padded_data = zeroPadding.zeroPadding_3D(whole_data, PATCH_LENGTH)

ITER = 3
CATEGORY = 9

train_data = np.zeros((TRAIN_SIZE, 2 * PATCH_LENGTH + 1, 2 * PATCH_LENGTH + 1, INPUT_DIMENSION_CONV))
test_data = np.zeros((TEST_SIZE, 2 * PATCH_LENGTH + 1, 2 * PATCH_LENGTH + 1, INPUT_DIMENSION_CONV))

KAPPA_3D_DenseNet = []
OA_3D_DenseNet = []
AA_3D_DenseNet = []
TRAINING_TIME_3D_DenseNet = []
TESTING_TIME_3D_DenseNet = []
ELEMENT_ACC_3D_DenseNet = np.zeros((ITER, CATEGORY))

# seeds = [1220, 1221, 1222, 1223, 1224, 1225, 1226, 1227, 1228, 1229]

seeds = [1220, 1221, 1222]

for index_iter in range(ITER):
    print("# %d Iteration" % (index_iter + 1))

    best_weights_DenseNet_path = 'D:/Tensorflow  Learning/3D-DenseNet-Hyperspectral/models-up-3dcnn/UP_best_3D_DenseNet_' + str(
        index_iter + 1) + '.hdf5'

    np.random.seed(seeds[index_iter])
    train_indices, test_indices = sampling(VALIDATION_SPLIT, gt)

    y_train = gt[train_indices] - 1
    y_train = to_categorical(np.asarray(y_train))

    y_test = gt[test_indices] - 1
    y_test = to_categorical(np.asarray(y_test))

    # print ("Validation data:")
    # collections.Counter(y_test_raw[-VAL_SIZE:])
    # print ("Testing data:")
    # collections.Counter(y_test_raw[:-VAL_SIZE])

    train_assign = indexToAssignment(train_indices, whole_data.shape[0], whole_data.shape[1], PATCH_LENGTH)
    for i in range(len(train_assign)):
        train_data[i] = selectNeighboringPatch(padded_data, train_assign[i][0], train_assign[i][1], PATCH_LENGTH)

    test_assign = indexToAssignment(test_indices, whole_data.shape[0], whole_data.shape[1], PATCH_LENGTH)
    for i in range(len(test_assign)):
        test_data[i] = selectNeighboringPatch(padded_data, test_assign[i][0], test_assign[i][1], PATCH_LENGTH)

    x_train = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2], INPUT_DIMENSION_CONV)
    x_test_all = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2], INPUT_DIMENSION_CONV)

    x_val = x_test_all[-VAL_SIZE:]
    y_val = y_test[-VAL_SIZE:]

    x_test = x_test_all[:-VAL_SIZE]
    y_test = y_test[:-VAL_SIZE]

    model_densenet = model_DenseNet()

    earlyStopping6 = kcallbacks.EarlyStopping(monitor='val_loss', patience=patience, verbose=1, mode='auto')
    saveBestModel6 = kcallbacks.ModelCheckpoint(best_weights_DenseNet_path, monitor='val_loss', verbose=1,
                                                save_best_only=True,
                                                mode='auto')

    tic6 = time.clock()
    print(x_train.shape, x_test.shape)
    history_3d_densenet = model_densenet.fit(
        x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], x_train.shape[3], 1), y_train,
        validation_data=(x_val.reshape(x_val.shape[0], x_val.shape[1], x_val.shape[2], x_val.shape[3], 1), y_val),
        batch_size=batch_size,
        nb_epoch=nb_epoch, shuffle=True, callbacks=[earlyStopping6, saveBestModel6])
    toc6 = time.clock()

    tic7 = time.clock()
    loss_and_metrics_3d_densenet = model_densenet.evaluate(
        x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], x_test.shape[3], 1), y_test,
        batch_size=batch_size)
    toc7 = time.clock()

    print('3D DenseNet  Time: ', toc6 - tic6)
    print('3D DenseNet  Test time:', toc7 - tic7)

    print('3D DenseNet  Test score:', loss_and_metrics_3d_densenet[0])
    print('3D DenseNet  Test accuracy:', loss_and_metrics_3d_densenet[1])

    print(history_3d_densenet.history.keys())

    pred_test = model_densenet.predict(
        x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], x_test.shape[3], 1)).argmax(axis=1)
    collections.Counter(pred_test)
    gt_test = gt[test_indices] - 1
    overall_acc = metrics.accuracy_score(pred_test, gt_test[:-VAL_SIZE])
    confusion_matrix = metrics.confusion_matrix(pred_test, gt_test[:-VAL_SIZE])
    each_acc, average_acc = averageAccuracy.AA_andEachClassAccuracy(confusion_matrix)
    kappa = metrics.cohen_kappa_score(pred_test, gt_test[:-VAL_SIZE])
    KAPPA_3D_DenseNet.append(kappa)
    OA_3D_DenseNet.append(overall_acc)
    AA_3D_DenseNet.append(average_acc)
    TRAINING_TIME_3D_DenseNet.append(toc6 - tic6)
    TESTING_TIME_3D_DenseNet.append(toc7 - tic7)
    ELEMENT_ACC_3D_DenseNet[index_iter, :] = each_acc

    print("3D DenseNet  finished.")
    print("# %d Iteration" % (index_iter + 1))

modelStatsRecord.outputStats(KAPPA_3D_DenseNet, OA_3D_DenseNet, AA_3D_DenseNet, ELEMENT_ACC_3D_DenseNet,
                             TRAINING_TIME_3D_DenseNet, TESTING_TIME_3D_DenseNet,
                             history_3d_densenet, loss_and_metrics_3d_densenet, CATEGORY,
                             'D:/Tensorflow  Learning/3D-DenseNet-Hyperspectral/records-up-3dcnn/UP_train_3D_10.txt',
                             'D:/Tensorflow  Learning/3D-DenseNet-Hyperspectral/records-up-3dcnn/UP_train_3D_element_10.txt')
