# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import dogs_breed_det.config as cfg
import dogs_breed_det.dataset.data_utils as dutils
from six.moves import urllib
from tqdm import tqdm

def maybe_download_bottleneck(bottleneck_storage = cfg.Dog_Storage, bottleneck_file = 'Dog_Resnet50_features_train.npz'):
    """
    Download bottleneck features if they do not exist locally.
    :param bottleneck_file: name of the file to download
    """
    bottleneck_exist = False
    
    bottleneck_maindir = os.path.join(cfg.BASE_DIR,'models','bottleneck_features')
    if not os.path.exists(bottleneck_maindir):
        os.makedirs(bottleneck_maindir)

    bottleneck_path = os.path.join(bottleneck_maindir, bottleneck_file)
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
            print('[INFO] Download error: %s' % error.reason)            
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
        
        
def build_features(img_files, set_type, network = 'Resnet50'):
    """Build bottleneck_features for set of files"""

    nets = {'VGG16': extract_VGG16,
            'VGG19': extract_VGG19,
            'Resnet50': extract_Resnet50,
            'InceptionV3': extract_InceptionV3,
            'Xception': extract_Xception,
    }
    
    bottleneck_features = nets[network](dutils.paths_to_tensor(img_files))
    np.savez(os.path.join(cfg.BASE_DIR, 'models', 'bottleneck_features', 
             'Dog_' + network + 'features_' + set_type), set_type=bottleneck_features)
    
    print("[INFO] Bottleneck features size (build_features):", bottleneck_features.shape)    
    
    return bottleneck_features


def load_features_set(data_type, network = 'Resnet50'):
    """Load features from the file
       Only one dataset, e.g. train, valid, test is loaded
    """

    bottleneck_file = 'Dog_' + network + '_features_' + data_type + '.npz'
    bottleneck_exists = maybe_download_bottleneck(cfg.Dog_Storage, bottleneck_file)
    
    if (not bottleneck_exists):
        print("[INFO] %s was neither found nor downloaded. Trying to build. It may take time .. " 
              % bottleneck_file)
        Dog_ImagesDir = os.path.join(cfg.BASE_DIR,'data', cfg.Dog_DataDir)
        img_files, targets = dutils.load_dataset(os.path.join(Dog_ImagesDir,data_type))
        build_features(img_files, data_type, network)
    
    bottleneck_path = os.path.join(cfg.BASE_DIR,'models','bottleneck_features', bottleneck_file)
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