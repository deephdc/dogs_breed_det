# -*- coding: utf-8 -*-
import numpy as np
import dogs_breed.config as cfg
from os import path

def build_features(network = 'Resnet50'):
    """Load features from the file"""
    
    bottleneck_path = path.join(cfg.basedir,'models','bottleneck_features','Dog' + network + 'Data.npz')
    bottleneck_features = np.load(bottleneck_path)
    train_net = bottleneck_features['train']
    valid_net = bottleneck_features['valid']
    test_net = bottleneck_features['test']
    
    return train_net, valid_net, test_net

def extract_VGG16(tensor):
	from keras.applications.vgg16 import VGG16, preprocess_input
	return VGG16(weights='imagenet', include_top=False).predict(preprocess_input(tensor))

def extract_VGG19(tensor):
	from keras.applications.vgg19 import VGG19, preprocess_input
	return VGG19(weights='imagenet', include_top=False).predict(preprocess_input(tensor))

def extract_Resnet50(tensor):
	from keras.applications.resnet50 import ResNet50, preprocess_input
	return ResNet50(weights='imagenet', include_top=False).predict(preprocess_input(tensor))

def extract_Xception(tensor):
	from keras.applications.xception import Xception, preprocess_input
	return Xception(weights='imagenet', include_top=False).predict(preprocess_input(tensor))

def extract_InceptionV3(tensor):
	from keras.applications.inception_v3 import InceptionV3, preprocess_input
	return InceptionV3(weights='imagenet', include_top=False).predict(preprocess_input(tensor))