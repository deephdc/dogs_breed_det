# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import dogs_breed_det.config as cfg
import dogs_breed_det.dataset.data_utils as dutils
from six.moves import urllib

def set_bottleneck_file(dataset_type, network = 'Resnet50', return_type = 'path'):
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

def maybe_download_bottleneck(bottleneck_storage = cfg.Dog_Storage, bottleneck_file = 'Dogs_Resnet50_features_train.npz'):
    """
    Download bottleneck features if they do not exist locally.
    :param bottleneck_storage: base url from where to download file
    :param bottleneck_file: name of the file to download
    :return: true/false for download (success/not)
    """
    bottleneck_exist = False
    
    bottleneck_dir = set_bottleneck_file('train', return_type='dir')
    if not os.path.exists(bottleneck_dir):
        os.makedirs(bottleneck_dir)

    bottleneck_path = os.path.join(bottleneck_dir, bottleneck_file)
    bottleneck_url = bottleneck_storage.rstrip('/') + '/' + bottleneck_file

    # if bottleneck_features file does not exist, download it
    if not os.path.exists(bottleneck_path):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (bottleneck_file,
                float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        try:
            #check that URL exists
            response = urllib.request.urlopen(bottleneck_url, timeout=3)
        except (urllib.error.URLError, urllib.error.HTTPError) as error:
            print('[INFO] Unsuccessfully downloaded %s' % bottleneck_file)
            print('[INFO] Download Error: %s' % error.reason)            
            bottleneck_exist = False
        else:
            bottleneck_path, _ = urllib.request.urlretrieve(bottleneck_url, bottleneck_path, _progress)
            print()
            statinfo = os.stat(bottleneck_path)
            print('[INFO] Successfully downloaded', bottleneck_file, statinfo.st_size, 'bytes.')
            bottleneck_exist = True
    else:
        bottleneck_exist = True
        
    return bottleneck_exist
        
        
def build_features(img_files, dataset_type, network = 'Resnet50'):
    """Build bottleneck_features for set of files"""

    nets = {'VGG16': extract_VGG16,
            'VGG19': extract_VGG19,
            'Resnet50': extract_Resnet50,
            'InceptionV3': extract_InceptionV3,
            'Xception': extract_Xception,
    }
    
    bottleneck_features = nets[network](dutils.paths_to_tensor(img_files))
    bottleneck_path = set_bottleneck_file(dataset_type, network,
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


def load_features_set(dataset_type, network = 'Resnet50'):
    """Load features from the file
       Only one dataset, e.g. train, valid, test is loaded
    """

    bottleneck_file = set_bottleneck_file(dataset_type, 
                                          network, 
                                          return_type='file')
    bottleneck_exists = maybe_download_bottleneck(cfg.Dog_Storage, 
                                                  bottleneck_file)
    
    if (bottleneck_exists):
        print("[INFO] Using %s" % bottleneck_file)
        bottleneck_path = set_bottleneck_file(dataset_type, network)
        bottleneck_features = np.load(bottleneck_path)[dataset_type]       
    else:
        print("[INFO] %s was neither found nor downloaded. Trying to build. It may take time .. " 
              % bottleneck_file)
        Dog_ImagesDir = os.path.join(cfg.BASE_DIR,'data', cfg.Dog_DataDir)
        img_files, targets = dutils.load_dataset(os.path.join(Dog_ImagesDir,dataset_type))
        bottleneck_features = build_features(img_files, dataset_type, network)

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