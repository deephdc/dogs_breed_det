# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 - 2019 Karlsruhe Institute of Technology - Steinbuch Centre for Computing
# This code is distributed under the MIT License
# Please, see the LICENSE file
#

import os
import numpy as np
import dogs_breed_det.config as cfg
import dogs_breed_det.dataset.data_utils as dutils


def set_features_file(dataset_type, network='Resnet50'):
    """ Function
        Returns according to the dataset_type and network 
        the full path to the file
    """

    # buld the file name
    file_name = 'Dogs_' + network + '_features_' + dataset_type + '.npz'
    # full path to the file
    file_path = os.path.join(cfg.DATA_DIR, 'bottleneck_features', file_name)

    return file_path

def build_features(data_type, network='Resnet50'):
    """Build bottleneck_features for set of files"""

    nets = {'VGG16': extract_VGG16,
            'VGG19': extract_VGG19,
            'Resnet50': extract_Resnet50,
            'InceptionV3': extract_InceptionV3,
            'Xception': extract_Xception,
    }

    data_dir = os.path.join(cfg.DATA_DIR, cfg.Dog_DataDir, data_type)
    img_files = dutils.load_data_files(data_dir)
    print("[DEBUG] build_features, img_files: ", img_files[:5])

    bottleneck_features = nets[network](dutils.paths_to_tensor(img_files))
    bottleneck_path = set_features_file(data_type, network,
                                        return_type='path')

    if data_type == 'train':
        np.savez(bottleneck_path, train=bottleneck_features)
    elif data_type == 'test':
        np.savez(bottleneck_path, test=bottleneck_features)
    elif data_type == 'valid':
        np.savez(bottleneck_path, valid=bottleneck_features)
    else:
        np.savez(bottleneck_path, features=bottleneck_features)

    print("[INFO] Bottleneck features size (build_features):",
          bottleneck_features.shape)

    return bottleneck_features


def load_features(data_type, network = 'Resnet50'):
    """Load features from the file
       Only one dataset, e.g. train, valid, test is loaded
    """

    bottleneck_path = set_features_file(data_type, network)
    print("[INFO] Using %s" % bottleneck_path)
    bottleneck_features = np.load(bottleneck_path)[data_type]

    return bottleneck_features

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
