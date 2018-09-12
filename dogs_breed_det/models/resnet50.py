# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 21:29:57 2018

@author: valentin
"""

import dogs_breed_det.config as cfg
import dogs_breed_det.models.general_net as gennet


def get_metadata():
    name = "Dogs_breed_Resnet50"
    d = {
        "id": "0",
        "name": name,
        "description": ("This is a model for dogs breed detection"
                        "(loaded '%s')" % name),
        "author": 'Valentin Kozlov',
        "version": "0.3.0",
    }

    return d
        

def build_model():
    """
    Simple call to Resnet50:
    """  
    return gennet.build_model('Resnet50', cfg.dogBreeds)
        

def predict_file(img_path):
    """
    Simple call to gennet.predict_file() using Resnet50
    :param img_path: image to classify, full path  
    :return: most probable dogs breed
    """
    return gennet.predict_file(img_path, 'Resnet50')


def predict_data(img):
    """
    Simple call to gennet.predict_data() using Resnet50
    """    
    return gennet.predict_data(img, 'Resnet50')


def predict_url(*args):
    """
    Simple call to gennet.predict_url()
    """    

    return gennet.predict_url(*args)
        

def train(nepochs=10):
    """
    Simple call to gennet.train() using Resnet50
    """ 

    return gennet.train(nepochs, 'Resnet50')
