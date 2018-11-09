# -*- coding: utf-8 -*-
import os
import numpy as np
import dogs_breed_det.config as cfg
import dogs_breed_det.dataset.data_utils as dutils


def set_features_file(dataset_type, network = 'Resnet50', return_type = 'path'):
    """ Function
        Returns according to the dataset_type and network either 
        directory with the file, filename, or full path to the file (default)
    """
    # directory where file is
    file_dir = os.path.join(cfg.BASE_DIR, 'models', 'bottleneck_features')
    # only file name
    file_name = 'Dogs_' + network + '_features_' + dataset_type  + '.npz'
    # full path to the file
    file_path = os.path.join(file_dir, file_name)
    
    if return_type == 'dir':
        return file_dir
    elif return_type == 'file':
        return file_name
    else:
        return file_path

def build_features_set(dataset_type, network = 'Resnet50'):
    """Build bottleneck_features for set of files"""

    nets = {'VGG16': extract_VGG16,
            'VGG19': extract_VGG19,
            'Resnet50': extract_Resnet50,
            'InceptionV3': extract_InceptionV3,
            'Xception': extract_Xception,
    }

    # check if directory with train, test, and valid images exist:
    dutils.maybe_download_and_unzip()
    
    data_dir = os.path.join(cfg.BASE_DIR,'data', cfg.Dog_DataDir, dataset_type)
    img_files, targets = dutils.load_dataset(data_dir)
      
    bottleneck_features = nets[network](dutils.paths_to_tensor(img_files))
    bottleneck_path = set_features_file(dataset_type, network,
                                        return_type='path')

    if dataset_type == 'train':
        np.savez(bottleneck_path, train=bottleneck_features)
    elif dataset_type == 'test':
        np.savez(bottleneck_path, test=bottleneck_features)
    elif dataset_type == 'valid':
        np.savez(bottleneck_path, valid=bottleneck_features)
    else:
        np.savez(bottleneck_path, features=bottleneck_features)
    
    print("[INFO] Bottleneck features size (build_features):", bottleneck_features.shape)    
    
    return bottleneck_features

def check_features_set(data_type, network = 'Resnet50'):
    """Check if features file exists locally
       Only one dataset, e.g. train, valid, test is checked
    """
    bottleneck_file =  set_features_file(data_type, network, return_type='file')
    bottleneck_exists, _ = dutils.maybe_download_data(
                                    data_dir='/models/bottleneck_features',
                                    data_file = bottleneck_file)        

    if not bottleneck_exists:
        print("[INFO] %s was neither found nor downloaded. Trying to build. It may take time .. " 
              % bottleneck_file)

        build_features_set(data_type, network)
        
        # Upload to nextcloud newly created file
        bottleneck_path = set_features_file(data_type, network)
        dest_dir = cfg.Dog_RemoteStorage.rstrip('/') + '/models/bottleneck_features'
        dutils.rclone_copy(bottleneck_path, dest_dir)

def load_features_set(data_type, network = 'Resnet50'):
    """Load features from the file
       Only one dataset, e.g. train, valid, test is loaded
    """
    check_features_set(data_type, network)
    
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