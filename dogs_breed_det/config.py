# -*- coding: utf-8 -*-
""" 
   Module to define CONSTANTS used across the project
"""

from os import path

# identify basedir for the package
BASE_DIR = path.dirname(path.normpath(path.dirname(__file__)))
Dog_Storage = 'https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/'
Dog_DatasetURL = Dog_Storage + 'dogImages.zip'
Dog_DataDir = 'dogImages'
Dog_LabelsFile = path.join(BASE_DIR,'data','dog_names.txt')
Dog_LabelsNum = 133


