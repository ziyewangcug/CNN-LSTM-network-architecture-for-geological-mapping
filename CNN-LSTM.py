# python code for CNN-LSTM network architecture
# author: Tong Li,Ziye Wang
# contact: Ziye Wang (Email: ziyewang@cug.edu.cn)
# input positive and negative training data (.csv) is formatted in N-dimensional matrix, where N equals the number of features
# more information can be found in the README

from __future__ import print_function, division
from keras.layers import *
import os
from keras.optimizers import SGD, Adam
import numpy as np
from sklearn import metrics
import csv
from keras.models import Sequential
import pandas as pd
from pandas import read_csv
from keras import optimizers
from tensorflow import random
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

imgs = []
labels = []

img = read_csv('positive training_data.csv', header=None) #input positive training data
img = img.values
for i in range(img.shape[0]):
    imgs.append(img[i, :])
    labels.append(1)
img = read_csv('negative training_data.csv', header=None) #input negative training data
img = img.values
for i in range(img.shape[0]):
    imgs.append(img[i, :])
    labels.append(0)
imgs = np.asarray(imgs, np.float32)
data = np.reshape(imgs, [-1, 7, 1]) #dimension of input data
label = np.asarray(labels, np.int32)

num_folds = 10 #number of k fold
kfold = KFold(n_splits=num_folds, shuffle=True)

base = 32

pre = read_csv('testing_data.csv', header=None) #input testing data
pre = pre.values
label_all = read_csv('testing_labels.csv', header=None) #input testing labels:0 or 1
label_all=label_all.values

result = np.zeros((pre.shape[0]))
k_num = 1
for train, test in kfold.split(data, label):
    k_num = k_num+1
    model = Sequential()
    model.add(Conv1D(base, 3, strides=1, padding='same', input_shape=(7, 1))) #dimension of input data
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv1D(base, 2, strides=1, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(base*2, 2, strides=1, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2, padding='same'))
    model.add(Conv1D(base*4, 2, strides=1, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2, padding='same'))
    model.add(Conv1D(base*4, 2, strides=1, padding='same'))
    model.add(Activation('relu'))
    model.add(LSTM(base*8, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True))
    model.add(Dropout(0.5))
    model.add(Dense(base*8, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08), metrics=["accuracy"]) #parameters: leraning rate
    history = model.fit(data, label, batch_size=128, epochs=600, verbose=2) #parameters: batch_size, epochs
    acc = history.history['accuracy']
    loss = history.history['loss']
    epochs = range(1, len(acc) + 1)


    for i in range(pre.shape[0]):
        pres = np.asarray(pre[i, :], np.float32)
        predict = np.reshape(pres, [1, 7, 1]) #dimension of input data
        x = model.predict(predict, verbose=2)
        result[i]=result[i]+x

    loss = pd.DataFrame(loss)
    loss.to_csv('loss.csv',mode='a+',encoding='utf-8') #save the loss function of k-fold cross validation
    acc = pd.DataFrame(acc)
    acc.to_csv('acc.csv', mode='a+',encoding='utf-8') #save the accuracy of k-fold cross validation

result = pd.DataFrame(result/num_folds)
result.to_csv('result.csv',encoding='utf-8') #output the average predicted probability
