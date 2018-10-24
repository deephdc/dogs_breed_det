# -*- coding: utf-8 -*-
import os
import sys
import zipfile
import numpy as np
import dogs_breed_det.config as cfg
from sklearn.datasets import load_files       
from keras.utils import np_utils
from glob import glob
from six.moves import urllib

from keras.preprocessing import image                  
from tqdm import tqdm

### dirty trick for 'truncated images':
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
###

def maybe_download_and_extract(dataset=cfg.Dog_DataDir, datasetURL=cfg.Dog_DatasetURL):
  """Download and extract the zip archive.
     Based on tensorflow tutorials."""
  data_dir = os.path.join(cfg.BASE_DIR,'data')
  if not os.path.exists(data_dir):
      os.makedirs(data_dir)

  rawdata_dir = os.path.join(data_dir,'raw')
  if not os.path.exists(rawdata_dir):
      os.makedirs(rawdata_dir)
  

  if not os.path.exists(os.path.join(data_dir, dataset)):
      filename = datasetURL.split('/')[-1]
      filepath = os.path.join(rawdata_dir, filename)
      
      if not os.path.exists(filepath):
          def _progress(count, block_size, total_size):
              sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                  float(count * block_size) / float(total_size) * 100.0))
              sys.stdout.flush()
          filepath, _ = urllib.request.urlretrieve(datasetURL, filepath, _progress)
          print()
          statinfo = os.stat(filepath)
          print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

      dataset_zip = zipfile.ZipFile(filepath, 'r')    
      dataset_zip.extractall(data_dir)
      dataset_zip.close()


# define function to load train, test, and validation datasets
def load_dataset(path):
    """
    Function to load train / validation / test datasets
    :param path: path to dataset images
    :return: numpy array containing file paths to images, numpy array containing onehot-encoded classification labels
    """
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

def dog_names_create(dog_names_file):
    """
    Function to return dog names based on directories in 'train'.
    Also creates .txt file with the names
    :return:  list of string-valued dog breed names for translating labels
    """
    dataImagesTrain = os.path.join(cfg.BASE_DIR,'data', cfg.Dog_DataDir, 'train','*')
    dog_names = [os.path.basename(os.path.normpath(item))[4:] for item in sorted(glob(dataImagesTrain))]

    with open(dog_names_file, 'w') as listfile:
        for item in dog_names:
            listfile.write("%s\n" % item)
    return dog_names

def dog_names_read(dog_names_file):
    """
    Function to return dog names read from the file.
    Also creates .txt file with the names
    :return:  list of string-valued dog breed names for translating labels
    """
    
    if os.path.isfile(dog_names_file):
        with open(dog_names_file, 'r') as listfile:
            dog_names = [ line.rstrip('\n') for line in listfile ]
    else:
        print("Warning! File ", dog_names_file, " doesn't exist. Trying to create it..")
        dog_names = dog_names_create(dog_names_file)

    return dog_names


def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)
