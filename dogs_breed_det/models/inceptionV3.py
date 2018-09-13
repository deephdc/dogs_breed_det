# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 21:29:57 2018

@author: valentin
"""

import dogs_breed_det.config as cfg
import dogs_breed_det.models.general_net as gennet


def get_metadata():
    """
    Simple call to get_metadata and set name to _InceptionV3:
    """ 
    meta = gennet.get_metadata()
    meta['Name'] = "Dogs_InceptionV3"

    return meta        

def build_model():
    """
    Simple call to InceptionV3:
    """  
    return gennet.build_model('InceptionV3', cfg.dogBreeds)
        

def predict_file(img_path):
    """
    Simple call to gennet.predict_file() using InceptionV3
    :param img_path: image to classify, full path  
    :return: most probable dogs breed
    """
    return gennet.predict_file(img_path, 'InceptionV3')


def predict_data(img):
    """
    Simple call to gennet.predict_data()  using InceptionV3
    """    
    return gennet.predict_data(img, 'InceptionV3')


def predict_url(*args):
    """
    Simple call to gennet.predict_url()
    """    
    return gennet.predict_url(*args)
        

def train(nepochs=10):
    """
    Simple call to gennet.train() using InceptionV3
    """ 
    return gennet.train(nepochs, 'InceptionV3')
