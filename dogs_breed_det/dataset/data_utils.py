# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 - 2019 Karlsruhe Institute of Technology - Steinbuch Centre for Computing
# This code is distributed under the MIT License
# Please, see the LICENSE file
#
import os
import sys
import zipfile
import subprocess
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

def byte2str(str_in):
    '''
    Simple function to decode a byte string (str_in).
    In case of a normal string, return is unchanged
    '''
    try:
        str_in = str_in.decode()
    except (UnicodeDecodeError, AttributeError):
        pass
    
    return str_in    

def rclone_call(src_path, dest_dir, cmd='copy', get_output=False):
    """ Function
        rclone calls
    """
    
    if cmd == 'copy':
        command = (['rclone', 'copy', '--progress', src_path, dest_dir])
    elif cmd == 'ls':
        command = (['rclone', 'ls', '-L', src_path])
    elif cmd == 'check':
        command = (['rclone', 'check', src_path, dest_dir])
    
    try:
        if get_output:
            result = subprocess.Popen(command, 
                                      stdout=subprocess.PIPE, 
                                      stderr=subprocess.PIPE)
        else:
            result = subprocess.Popen(command, stderr=subprocess.PIPE)
        
        output, error = result.communicate()
    except OSError as e:
        output, error = None, e
    
    output = byte2str(output)
    error = byte2str(error)
    
    return output, error


def rclone_copy(src_path, dest_dir, src_type='file', verbose=False):
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
        print('[ERROR, rclone_copy()] %s (src):\n%s' % (src_path, error))
        error_out = error
        dest_exist = False
    else:
        # if src_path exists, copy it
        output, error = rclone_call(src_path, dest_dir, cmd='copy')
        if not error:       
            output, error = rclone_call(dest_path, dest_dir,
                                        cmd='ls', get_output=True)
            file_size = [ elem for elem in output.split(' ') if elem.isdigit() ][0]
            print('[INFO] Copied to %s %s bytes' % (dest_path, file_size))
            dest_exist = True
            if verbose:
                # compare two directories, if copied file appears in output
                # as not found or not matching -> Error
                print('[INFO] File %s copied. Check if (src) and (dest) really match..' % (dest_file))
                output, error = rclone_call(src_dir, dest_dir, cmd='check')
                if 'ERROR : ' + dest_file in error:
                    print('[ERROR, rclone_copy()] %s (src) and %s (dest) do not match!' 
                          % (src_path, dest_path))
                    error_out = 'Copy failed: ' + src_path + ' (src) and ' + \
                                 dest_path + ' (dest) do not match'
                    dest_exist = False     
        else:
            print('[ERROR, rclone_copy()] %s (src):\n%s' % (dest_path, error))
            error_out = error
            dest_exist = False

    return dest_exist, error_out

def url_download(url_path = cfg.Dog_RemoteShare,
                 data_dir = os.path.join(cfg.BASE_DIR, 'models'),
                 data_file = 'weights.best.ResNet50.hdf5'):
    """ Function to copy a file from URL
    :param url_path: remote URL to download
    :param data_dir: full path to where for storing the file
    :param data_file: file name into which for saving the remote file
    :return: if file is downloaded (=local version exists), possible error
    """
    
    file_path = os.path.join(data_dir, data_file)

    def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading %s %.1f' % (data_file,
                        float(count * block_size)))
        sys.stdout.flush()

    file_path, _ = urllib.request.urlretrieve(url_path, file_path, _progress)
    statinfo = os.stat(file_path)
    
    if os.path.exists(file_path):
        print('[INFO] Successfully downloaded %s, %d bytes' % 
               (data_file, statinfo.st_size))
        dest_exist = True
        error_out = None
    else:
        dest_exist = False
        error_out = '[ERROR, url_download()] Failed to download ' + data_file + \
                    ' from ' + url_path
        
    return dest_exist, error_out
    

def maybe_download_data(remote_storage=cfg.Dog_RemoteStorage, 
                        data_dir='/models/bottleneck_features',
                        data_file='Resnet50_features_train.npz'):
    """
    Download data if it does not exist locally.
    :param remote_storage: remote storage where to download from
    :param data_dir: remote _and_ local dir to put data
    :param data_file: name of the file to download
    """
    # status for data if exists or not
    status = False
    error_out = None

    #join doesn't work if data_dir starts with '/'!
    data_dir = data_dir.lstrip('/')
    data_dir = data_dir.rstrip('/')

    local_dir = os.path.join(cfg.BASE_DIR, data_dir)
    local_path = os.path.join(local_dir, data_file)
    # if data_file does not exist locally, download it
    if not os.path.exists(local_path):
        remote_url = remote_storage.rstrip('/') + '/' + \
                     os.path.join(data_dir, data_file)        
        print("[INFO] Url: %s" % (remote_url))
        print("[INFO] Local path: %s" % (local_path))        
        #check that every sub directory exists locally, if not -> create
        data_subdirs = data_dir.split('/')
        sub_dir = cfg.BASE_DIR
        for sdir in data_subdirs:
            sub_dir = os.path.join(sub_dir, sdir)
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)
        status, error_out = rclone_copy(remote_url, local_dir)
        #print("[DEBUG, maybe_download_data]: ", status, error_out)
    else:
        status = True
        error_out = None
        
    return status, error_out


def maybe_download_and_unzip(data_storage=cfg.Dog_RemoteStorage,
                             data_dir='/data/raw',
                             data_file='dogImages.zip'):
    """Download and extract the zip archive.
    """

    data_dir = data_dir.lstrip('/')
    # for now we assume that everything will be unzipped in ~/data directory
    unzip_dir = os.path.join(cfg.BASE_DIR, 'data')
  
    # remove last extension, should be .zip
    data_name = os.path.splitext(data_file)[0]

    # if 'data_name' is not present locally, 
    # try to download and de-archive corresponding .zip file
    unzip_dir_data = os.path.join(unzip_dir, data_name)
    if not os.path.exists(unzip_dir_data):
        print("[INFO] %s does not exist, trying dowload zipped file %s" % 
              (unzip_dir_data, data_file))
        # check if .zip file present in locally
        status, _ = maybe_download_data(remote_storage=data_storage, 
                                        data_dir=data_dir, 
                                        data_file=data_file)
        # if .zip is present locally, de-archive it
        file_path = os.path.join(cfg.BASE_DIR, data_dir, data_file)
        print(file_path)
        if os.path.exists(file_path):
            data_zip = zipfile.ZipFile(file_path, 'r')
            data_zip.extractall(unzip_dir)
            data_zip.close()


# define function to load train, test, and validation datasets
def build_targets(data_type):
    """
    Function to create train / validation / test one-hot encoded targets
    :param path: path to dataset images
    :return: numpy array containing onehot-encoded classification labels
    """
    data_dir = os.path.join(cfg.BASE_DIR, 'data', cfg.Dog_DataDir, data_type)
    data = load_files(data_dir)
    # get number of classes for one-hot encoding
    nclasses = len(dog_names_load()) # expect that cfg.Dog_LabelsFile exists
    dog_targets = np_utils.to_categorical(np.array(data['target']), nclasses)
    targets_file = 'Dogs_targets_' + data_type + '.npz'
    targets_path = os.path.join(cfg.BASE_DIR, 'data', targets_file) 
    
    if data_type == 'train':
        np.savez(targets_path, train=dog_targets)
    elif data_type == 'test':
        np.savez(targets_path, test=dog_targets)
    elif data_type == 'valid':
        np.savez(targets_path, valid=dog_targets)
    else:
        np.savez(targets_path, features=dog_targets)
    
    print("[INFO] Targets file shape (%s): %s" % 
           (data_type, dog_targets.shape) )
    
    return dog_targets

    
def load_data_files(path):
    """
    Function to load train / validation / test file names
    :param path: path to dataset images
    :return: numpy array containing file paths to images
    """    
    data = load_files(path)
    data_files = np.array(data['filenames'])
    return data_files


def load_targets(data_type):
    """Load features from the file
       Only one dataset, e.g. train, valid, test is loaded
    """
    
    targets_file = 'Dogs_targets_' + data_type + '.npz'
    targets_path = os.path.join(cfg.BASE_DIR, 'data', targets_file)
    print("[INFO] Using %s" % targets_path)
    targets = np.load(targets_path)[data_type]

    return targets


def dog_names_create(dataImagesTrain=os.path.join(cfg.BASE_DIR,
                                                  'data', cfg.Dog_DataDir, 
                                                  'train','*'),
                     dog_names_path=cfg.Dog_LabelsFile):
    """
    Function to create dog_names file based on sub-directories in 'train'.
    :return:  list of string-valued dog breed names for translating labels
    """
    
    #dog_names = [os.path.basename(os.path.normpath(item))[4:] for item in sorted(glob(dataImagesTrain))]
    
    # attempt to identify 'numbered' directories and 'not numbered'
    dot_count = 0
    dir_count = 0
    for item in sorted(glob(dataImagesTrain)):
        if '.' in item:
            dot_count += 1 
        dir_count += 1

    flag_dot = True if dot_count == dir_count else False
    
    if flag_dot:
        dog_names = [os.path.basename(os.path.normpath(item)).split('.', 1)[1] 
                     for item in sorted(glob(dataImagesTrain))]
    else:
        dog_names = [os.path.basename(os.path.normpath(item)) 
                     for item in sorted(glob(dataImagesTrain))]        

    print('[INFO] Creating %s file with %d classes (dogs breeds)' % 
          (cfg.Dog_LabelsFile, len(dog_names)))

    with open(dog_names_path, 'w') as listfile:
        for item in dog_names:
            listfile.write("%s\n" % item)
    
    dest_dir = cfg.Dog_RemoteStorage.rstrip('/') + '/data'
    print("[INFO] Upload %s to %s" % (cfg.Dog_LabelsFile, dest_dir))
    rclone_copy(cfg.Dog_LabelsFile, dest_dir)
            
    return dog_names


def dog_names_load(labels_path=cfg.Dog_LabelsFile):
    """
    Function to return dog names read from the file.
    :return:  list of string-valued dog breed names for translating labels
    """
    # we expect that file already exists
    with open(labels_path, 'r') as listfile:
        dog_names = [ line.rstrip('\n') for line in listfile ]

    return dog_names


def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3)
    # and return 4D tensor
    return np.expand_dims(x, axis=0)


def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)
