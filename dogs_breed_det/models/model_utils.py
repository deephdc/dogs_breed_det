# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 - 2019 Karlsruhe Institute of Technology - Steinbuch Centre for Computing
# This code is distributed under the MIT License
# Please, see the LICENSE file
#

import requests
import dogs_breed_det.config as cfg

def build_weights_filename(network):
    '''
    Builds filename for the weights file
    '''
    weights_filename = cfg.Dog_WeightsPattern.replace('NETWORK', network)
    return weights_filename

def format_prediction(labels, probabilities):
    d = {
        "status": "ok",
        "predictions": [],
    }

    for label, prob in zip(labels, probabilities):
        name = label

        pred = {
            "label": name,
            "probability": float(prob),
            "info": {
                "links": [{"link": 'Google images', "url": image_link(name)},
                          {"link": 'Wikipedia', "url": wikipedia_link(name)}],
            },
        }
        d["predictions"].append(pred)
    return d


def image_link(pred_lab):
    """
    Return link to Google images
    """
    base_url = 'https://www.google.es/search?'
    params = {'tbm': 'isch','q': pred_lab}
    link = base_url + requests.compat.urlencode(params)
    return link


def wikipedia_link(pred_lab):
    """
    Return link to wikipedia webpage
    """
    base_url = 'https://en.wikipedia.org/wiki/'
    link = base_url + pred_lab.replace(' ', '_')
    return link


def format_train(network, accuracy, nepochs, data_size, 
                 time_prepare, mn_train, std_train):


    train_info = {
        "network": network,
        "test accuracy": accuracy,
        "n epochs": nepochs,
        "train set (images)": data_size['train'],
        "validation set (images)": data_size['valid'],
        "test set (images)": data_size['test'],
        "time": {
                "time to prepare": time_prepare,
                "mean per epoch (s)": mn_train,
                "std (s)": std_train,
                },
    }

    return train_info