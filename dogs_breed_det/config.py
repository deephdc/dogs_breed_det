# -*- coding: utf-8 -*-
""" 
   Module to define CONSTANTS used across the project
"""

from os import path

# identify basedir for the package
BASE_DIR = path.dirname(path.normpath(path.dirname(__file__)))
Dog_RemoteStorage = 'deep-nextcloud:/Datasets/dogs_breed/'
Dog_DataDir = 'dogImages'
Dog_LabelsFile = path.join(BASE_DIR,'data','dog_names.txt')
Dog_LabelsNum = 133


