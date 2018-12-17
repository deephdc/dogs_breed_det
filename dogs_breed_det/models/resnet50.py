# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 21:29:57 2018

@author: valentin
"""
import time
import argparse
import dogs_breed_det.config as cfg
import dogs_breed_det.models.general_net as gennet


def get_metadata():
    """
    Simple call to get_metadata and set name to _Resnet50:
    """
    meta = gennet.get_metadata()
    meta['Name'] = "Dogs_Resnet50"

    return meta


def build_model():
    """
    Simple call to Resnet50:
    """  
    return gennet.build_model('Resnet50')


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


# during development it might be practical 
# to check your code from the command line
def main():
    """
       Runs above-described functions depending on input parameters
       (see below an example)
    """

    if args.method == 'get_metadata':
        get_metadata()       
    elif args.method == 'predict_file':
        predict_file(args.file)
    elif args.method == 'predict_data':
        predict_data(args.file)
    elif args.method == 'predict_url':
        predict_url(args.url)
    elif args.method == 'train':
        start = time.time() 
        train(args.n_epochs)
        print("Elapsed time:  ", time.time() - start)
    else:
        get_metadata()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model parameters')
    parser.add_argument('--method', type=str, default="get_metadata",
                        help='Method to use: get_metadata (default), \
                        predict_file, predict_data, predict_url, train')
    parser.add_argument('--file', type=str, help='File to do prediction on, full path')
    parser.add_argument('--url', type=str, help='URL with the image to do prediction on')      
    parser.add_argument('--n_epochs', type=int, default=15, 
                        help='Number of epochs to train on')                            
    args = parser.parse_args()    
    
    main()
