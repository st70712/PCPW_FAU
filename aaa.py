from __future__ import print_function
import numpy as np
import os
import sys
import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from libs.readfile_fau import readFile, separa_fau, combine_five
import h5py


filename_train = './data/FAU_train.arff'
filename_test =  './data/FAU_test.arff'
train_data, train_label, train_size = readFile(filename_train)
test_data, test_label, test_size = readFile(filename_test)
#teacher_label = np.load("teacher_label.npy")
#train_label = teacher_label  #use teacher_label

fau_a,fau_e,fau_n,fau_p,fau_r = separa_fau(train_data,train_label)
print("Anger:",fau_a.shape,"Emphatic:",fau_e.shape,"Neutral:",fau_n.shape,"Positive:",fau_p.shape,"Rest:",fau_r.shape)
print (train_data.shape)
