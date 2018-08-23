# -*- coding: utf-8 -*-
"""Functions to build neural network model"""
import numpy as np
import dogs_breed.features.build_features as bfeatures
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.models import Sequential

def build_model(network = 'Resnet50'):
    """
    Build network. Possible nets:
    Resnet50, Xception, InceptionV3, Resnet50, VGG19, VGG16
    """
    train_net, _, _ = bfeatures.build_features(network)

    net_model = Sequential()
    net_model.add(GlobalAveragePooling2D(input_shape=train_net.shape[1:]))
    net_model.add(Dense(133, activation='softmax'))
    
    print("__"+network+"__: ")
    net_model.summary()
    net_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    
    return net_model
