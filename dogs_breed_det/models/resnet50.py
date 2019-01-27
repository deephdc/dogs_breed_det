# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 21:29:57 2018

@author: valentin
"""
import time
import json
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



def train(user_args):
    """
    Simple call to gennet.train() using Resnet50
    """

    num_epochs = json.loads(user_args['n_epochs'])
    net = json.loads(user_args['network'])

    return gennet.train(nepochs=num_epochs, network=net)


def get_train_args():
    """
    Returns a dict of dicts with the following structure to feed the deepaas API parser:
    { 'arg1' : {'default': '1',     #value must be a string (use json.dumps to convert Python objects)
                'help': '',         #can be an empty string
                'required': False   #bool
                },
      'arg2' : {...
                },
    ...
    }
    """
  
    train_args = { 'n_epochs': {'default': json.dumps(10),
                               'help': 'Number of epochs to train on',
                               'required': False
                               },
                   'network': {'default': json.dumps("Resnet50"),
                               'help': 'Neural model to use: \"Resnet50\" (default),\
                                        \"InceptionV3\", \"VGG16\", \"VGG19\"',
                               'required': False
                               },
                 }
    
    return train_args

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
        train(vars(args))
        print("Elapsed time:  ", time.time() - start)
    else:
        get_metadata()


from absl import flags as absl_flags
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Model parameters')

    train_args = get_train_args()
    for k, v in train_args.items():
        parser.add_argument('--' + k, **v)
        print(k, v)

    parser.add_argument('--method', type=str, default="get_metadata",
                        help='Method to use: get_metadata (default), \
                        predict_file, predict_data, predict_url, train')
    parser.add_argument('--file', type=str, help='File to do prediction on, full path')
    parser.add_argument('--url', type=str, help='URL with the image to do prediction on')

    args = parser.parse_args()
    
    main()
