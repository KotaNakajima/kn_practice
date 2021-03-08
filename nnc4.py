import glob
import numpy as np
import pandas as pd
import os
import keras
import matplotlib.pyplot as plt
import glob
import tensorflow as tf

from collections import defaultdict

from keras.applications.resnet_v2 import ResNet152V2

from keras.activations import selu

from keras import models, callbacks
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D, DepthwiseConv2D, MaxPooling2D, AveragePooling2D
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.merge import add, multiply, concatenate, maximum, average
from keras.layers import Input, LeakyReLU, ELU
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from sklearn.model_selection import train_test_split
from keras import optimizers
from keras.preprocessing.image import load_img,img_to_array
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import layers
from keras.layers.normalization import BatchNormalization
from scipy import optimize
from sklearn.metrics import r2_score
from tensorflow.keras.utils import plot_model
from keras.optimizers import Adam, Nadam



class MODEL:
    def __init__(self):
        self.model = self.make_model()

    def make_model(self):
        inputs = Input(shape=(225,300,3))

        x = Conv2D(45,(3,3),padding='same')(inputs)
        x = AveragePooling2D((2,1))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(16,(3,3),padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Conv2D(16,(3,3),padding='same')(x)
        x = BatchNormalization()(x)
        x = AveragePooling2D((2,3))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2,2))(x)
        x = Conv2D(16,(1,1),padding='same')(x)
        x = BatchNormalization()(x)
        x = ELU(alpha=1)(x)
        x = Conv2D(16,(5,3),padding='same')(x)
        x = AveragePooling2D((2,2))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        x_1 = Conv2D(16,(3,3),padding='same')(x)
        x_1 = AveragePooling2D((2,2))(x_1)
        x_1 = BatchNormalization()(x_1)
        x_1 = Activation('relu')(x_1)  
        
        x_2 = Conv2D(16,(3,3),padding='same')(x)
        x_2 = AveragePooling2D((2,2))(x_2)
        x_2 = BatchNormalization()(x_2)
        x_2 = Activation('relu')(x_2)
        
        x_m = add([x_1,x_2])
        x_m = Flatten()(x_m)
        predictions = Dense(1,activation='relu')(x_m)
        
        currentmodel = Model(inputs=inputs, outputs=predictions)
        
        return currentmodel

def build():
    model = MODEL().model
    return model


#x = LeakyReLU(alpha=0.1)(x)