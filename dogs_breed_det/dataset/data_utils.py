# -*- coding: utf-8 -*-
import os
import sys
import zipfile
import subprocess
import numpy as np
import dogs_breed_det.config as cfg
from sklearn.datasets import load_files       
from keras.utils import np_utils
from glob import glob

from keras.preprocessing import image                  
from tqdm import tqdm

### dirty trick for 'truncated images':
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
###


def rclone_call(src_path, dest_dir, cmd = 'copy', get_output=False):
    """ Function
        rclone calls
    """
    if cmd == 'copy':
        command = (['rclone', 'copy', '--progress', src_path, dest_dir]) #'--progress', 
    elif cmd == 'ls':
        command = (['rclone', 'ls', src_path])
    elif cmd == 'check':
        command = (['rclone', 'check', src_path, dest_dir])
    
    if get_output:
        result = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    else:
        result = subprocess.Popen(command, stderr=subprocess.PIPE) #stdout=subprocess.PIPE,
    output, error = result.communicate()
    return output, error

def rclone_copy(src_path, dest_dir, src_type='file'):
    """ Function for rclone call to copy data (sync?)
    :param src_path: full path to source (file or directory)
    :param dest_dir: full path to destination directory (not file!)
    :param src_type: if source is file (default) or directory
    :return: if destination was downloaded, and possible error 
    """

    error_out = None
    
    if src_type == 'file':
        src_dir = os.path.dirname(src_path)
        dest_file = src_path.split('/')[-1]
        dest_path = os.path.join(dest_dir, dest_file)
    else:
        src_dir = src_path
        dest_path =  dest_dir

    # check first if we find src_path
    output, error = rclone_call(src_path, dest_dir, cmd='ls')
    if error:
        print('[ERROR] %s (src):\n%s' % (src_path, error))
        error_out = error
        dest_exist = False
    else:
        # if src_path exists, copy it
        output, error = rclone_call(src_path, dest_dir, cmd='copy')
        if not error:
            # compare two directories, if copied file appears in output
            # as not found or not matching -> Error
            print('[INFO] File %s copied. Check if (src) and (dest) really match..' % (dest_file))
            output, error = rclone_call(src_dir, dest_dir, cmd='check')
            if 'ERROR : ' + dest_file in error:
                print('[ERROR] %s (src) and %s (dest) do not match!' % (src_path, dest_path))
                error_out = 'Copy failed: ' + src_path + ' (src) and ' + \
                             dest_path + ' (dest) do not match'
                dest_exist = False
            else:
                output, error = rclone_call(dest_path, dest_dir, 
                                            cmd='ls', get_output = True)
                file_size = [ elem for elem in output.split(' ') if elem.isdigit() ][0]
                print('[INFO] Checked: Successfully copied to %s %s bytes' % (dest_path, file_size))
                dest_exist = True
        else:
            print('[ERROR] %s (src):\n%s' % (dest_path, error))
            error_out = error
            dest_exist = False

    return dest_exist, error_out


def maybe_download_and_extract(dataset_storage=cfg.Dog_RemoteStorage, 
                               dataset=cfg.Dog_DataDir):
  """Download and extract the zip archive.
  """
  data_dir = os.path.join(cfg.BASE_DIR,'data')
  if not os.path.exists(data_dir):
      os.makedirs(data_dir)

  rawdata_dir = os.path.join(data_dir,'raw')
  if not os.path.exists(rawdata_dir):
      os.makedirs(rawdata_dir)
  
  dataset_URL = dataset_storage.rstrip('/') + \
               os.path.join('/data/raw','dogImages.zip')

  if not os.path.exists(os.path.join(data_dir, dataset)):
      file_name = dataset_URL.split('/')[-1]
      file_path = os.path.join(rawdata_dir, file_name)
      
      if not os.path.exists(file_path):
          status, _ = rclone_copy(dataset_URL, rawdata_dir)

      if os.path.exists(file_path):
          dataset_zip = zipfile.ZipFile(file_path, 'r')    
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
