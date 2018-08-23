# -*- coding: utf-8 -*-
import numpy as np
import dogs_breed_det.config as cfg
from sklearn.datasets import load_files       
from keras.utils import np_utils
from os import path
from glob import glob

from keras.preprocessing import image                  
from tqdm import tqdm

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

def dog_names_create(dogNamesFile):
    """
    Function to return dog names based on directories in 'train'.
    Also creates .txt file with the names
    :return:  list of string-valued dog breed names for translating labels
    """
    dataImagesTrain = path.join(cfg.basedir,'data','dogImages','train','*')
    dogNames = [path.basename(path.normpath(item))[4:] for item in sorted(glob(dataImagesTrain))]

    with open(dogNamesFile, 'w') as listfile:
        for item in dogNames:
            listfile.write("%s\n" % item)
    return dogNames

def dog_names_read(dogNamesFile):
    """
    Function to return dog names read from the file.
    Also creates .txt file with the names
    :return:  list of string-valued dog breed names for translating labels
    """
    
    if path.isfile(dogNamesFile):
        with open(dogNamesFile, 'r') as listfile:
            dogNames = [ line.rstrip('\n') for line in listfile ]
    else:
        print("Warning! File ", dogNamesFile, " doesn't exist. Trying to create it..")
        dogNames = dog_names_create(dogNamesFile)

    return dogNames


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
