# -*- coding: utf-8 -*-
from os import path

# identify basedir for the package
basedir = path.dirname(path.normpath(path.dirname(__file__)))
dogDatasetUrl = 'https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip'
dogNamesFile = path.join(basedir,'data','dog_names.txt')
