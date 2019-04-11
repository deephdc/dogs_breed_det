# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 - 2019 Karlsruhe Institute of Technology - Steinbuch Centre for Computing
# This code is distributed under the MIT License
# Please, see the LICENSE file
#
"""
Created on Mon Sep  3 21:29:57 2018

@author: valentin.kozlov
"""

import os
import time
import yaml
import keras
import tempfile
import argparse
import numpy as np
import pkg_resources
import dogs_breed_det.config as cfg
import dogs_breed_det.run_info as rinfo
import dogs_breed_det.dataset.data_utils as dutils
import dogs_breed_det.dataset.make_dataset as mdata
import dogs_breed_det.models.model_utils as mutils
import dogs_breed_det.features.build_features as bfeatures
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras import backend

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

## Authorization
from flaat import Flaat
flaat = Flaat()
    

class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.total_duration = 0.
        self.durations = []
        self.val_acc = 0.
        self.val_loss = 0.

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        duration_epoch = time.time() - self.epoch_time_start
        self.total_duration += duration_epoch
        self.durations.append(duration_epoch)
        self.val_acc = logs.get('val_acc')
        self.val_loss = logs.get('val_loss')

    
def get_metadata():
    """
    Function to read metadata
    """

    module = __name__.split('.', 1)

    pkg = pkg_resources.get_distribution(module[0])
    meta = {
        'Name': None,
        'Version': None,
        'Summary': None,
        'Home-page': None,
        'Author': None,
        'Author-email': None,
        'License': None,
        'Train_args': cfg.train_args,
    }

    for line in pkg.get_metadata_lines("PKG-INFO"):
        for par in meta:
            if line.startswith(par):
                _, value = line.split(": ", 1)
                meta[par] = value

    return meta


def prepare_data(network='Resnet50'):
    """ Function to prepare data
    """
    mdata.prepare_data(network)


def build_model(network='Resnet50'):
    """
    Build network. Possible nets:
    Resnet50, VGG19, VGG16, InceptionV3, Xception
    """

    # introduce bottleneck_features shapes manually 
    features_shape = {'VGG16': [7, 7, 512],
                      'VGG19': [7, 7, 512],
                      'Resnet50': [1, 1, 2048],
                      'InceptionV3': [5, 5, 2048],
                      'Xception': [7, 7, 2048],
    }

    #train_net = bfeatures.load_features('train', network)
    dog_names = dutils.dog_names_load()
    nclasses = len(dog_names)
    print('[INFO] Found %d classes (dogs breeds)' % nclasses)

    net_model = Sequential()
    # add a global average pooling layer    
    #net_model.add(GlobalAveragePooling2D(input_shape=train_net.shape[1:]))
    net_model.add(GlobalAveragePooling2D(input_shape=features_shape[network]))
    # add a fully-connected layer
    fc_layers = int(features_shape[network][2]/2.)
    net_model.add(Dense(fc_layers, activation='relu'))
    # add a classification layer    
    net_model.add(Dense(nclasses, activation='softmax'))

    print("__" + network+"__: ")
    net_model.summary()
    net_model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])
    
    return net_model


def predict_file(img_path, network='Resnet50'):
    """
    Function to make prediction which breed is closest
    :param img_path: image to classify, full path
    :param network: neural network to be used    
    :return: most probable dogs breed
    """

    nets = {
        'VGG16': bfeatures.extract_VGG16,
        'VGG19': bfeatures.extract_VGG19,
        'Resnet50': bfeatures.extract_Resnet50,
        'InceptionV3': bfeatures.extract_InceptionV3,
        'Xception': bfeatures.extract_Xception,
    }

    # clear possible pre-existing sessions. IMPORTANT!
    backend.clear_session()

    # check that all necessary data is there
    #prepare_data(network)

    weights_file = 'weights.best.' + network + '.hdf5'
    saved_weights_path = os.path.join(cfg.BASE_DIR, 'models', weights_file) 

    # check if the weights file exists locally. if not -> try to download
    status_weights, _ = dutils.maybe_download_data(data_dir='/models',
                                                   data_file=weights_file)
    
    dog_names_file = cfg.Dog_LabelsFile.split('/')[-1]
    # check if the labels file exists locally. if not -> try to download
    status_weights, _ = dutils.maybe_download_data(data_dir='/data',
                                                   data_file=dog_names_file)
                                                   

    # attempt to download default weights file for Resnet50                                                   
    if not status_weights and network=='Resnet50':
        print("[INFO] Trying to download weights and labels from the public link")
        url_weights = cfg.Dog_RemoteShare + weights_file
        status_weights, _ = dutils.url_download(
                                   url_path = url_weights, 
                                   data_dir='/models',
                                   data_file=weights_file)
        url_dog_names = cfg.Dog_RemoteShare + dog_names_file
        status_weights, _ = dutils.url_download(
                                   url_path = url_dog_names, 
                                   data_dir='/data',
                                   data_file=dog_names_file)                                   

    if status_weights:
        net_model = build_model(network)
        net_model.load_weights(saved_weights_path)

        # extract bottleneck features
        bottleneck_feature = nets[network](dutils.path_to_tensor(img_path))
        print("[INFO] Bottleneck feature size: %s" % str(bottleneck_feature.shape))
        dog_names  = dutils.dog_names_load()
        # obtain predicted vector
        predicted_vector = net_model.predict(bottleneck_feature)
        print("[INFO] Sum: %f" % np.sum(predicted_vector))
        # return dog breed that is predicted by the model
        idxs = np.argsort(predicted_vector[0])[::-1][:5] 
        # dog_names_best = [ dog_names[i] for i in idxs ]
        dog_names_best = []
        probs_best = []
        for i in idxs:
            dog_names_best.append(dog_names[i])
            probs_best.append(predicted_vector[0][i])
            print("%s : %f" % (dog_names[i], predicted_vector[0][i]))

        msg = mutils.format_prediction(dog_names_best, probs_best)
    else:
        msg = "[ERROR] No weights file found! Please first train the model " + \
              "with the " + network + " network!"
    return msg


def predict_data(img, network='Resnet50'):
    if not isinstance(img, list):
        img = [img]

    filenames = []

    for image in img:
        f = tempfile.NamedTemporaryFile(delete=False)
        f.write(image)
        f.close()
        filenames.append(f.name)
        print("tmp file: ", f.name)

    prediction = []
    try:
        for imgfile in filenames:
            prediction.append(predict_file(imgfile, network))
    except Exception as e:
        raise e
    finally:
        for imgfile in filenames:
            os.remove(imgfile)

    return prediction


def predict_url(*args):
    message = 'Not (yet) implemented in the model (predict_url())'
    return message


# Require only authorized people to do training
@flaat.login_required()
def train(train_args):
    """
    Train network (transfer learning)
    Parameters
    ----------
    train_args : dict
        Json dict (created with json.dumps) with the user's configuration parameters that will replace the defaults.
        Can be loaded with json.loads() or (better for strings) with yaml.safe_load()
    """
    run_results = { "status": "ok",
                    "run_info": [],
                    "training": [],
                  }
  
    print("train_args:", train_args)
    
    print(type(train_args.num_epochs), type(train_args.network))
    num_epochs = yaml.safe_load(train_args.num_epochs)

    #network = json.loads(args.network)
    network = yaml.safe_load(train_args.network)
    print("Network:", train_args.network, network)

    flag_run_info = yaml.safe_load(train_args.run_info)
    print(flag_run_info)
    if(flag_run_info):
        rinfo.get_run_info(cfg.machine_info)
        print("run_info should be True: ", flag_run_info)
        print(cfg.machine_info)
        run_results["run_info"].append(cfg.machine_info)
    else:
        print("run_info should be False: ", flag_run_info)

    # check that all necessary data is there
    time_start = time.time()
    prepare_data(network)
    time_prepare = time.time() - time_start

    train_targets = dutils.load_targets('train')
    valid_targets = dutils.load_targets('valid')
    test_targets = dutils.load_targets('test')

    train_net = bfeatures.load_features('train', network)
    valid_net = bfeatures.load_features('valid', network)
    test_net = bfeatures.load_features('test', network)

    print('[INFO] Sizes of bottleneck_features (train, valid, test):')
    print(train_net.shape, valid_net.shape, test_net.shape)
    data_size = {
        'train': len(train_targets),
        'valid': len(valid_targets),
        'test': len(test_targets)
        }

    saved_weights_path = os.path.join(cfg.BASE_DIR, 'models', 
                                     'weights.best.' + network + '.hdf5')
    checkpointer = ModelCheckpoint(filepath=saved_weights_path, 
                                   verbose=1, save_best_only=True)

    # clear possible pre-existing sessions. important!
    backend.clear_session()

    net_model = build_model(network)

    time_callback = TimeHistory()        
    net_model.fit(train_net, train_targets, 
                  validation_data=(valid_net, valid_targets),
                  epochs=num_epochs, batch_size=20, 
                  callbacks=[checkpointer, time_callback], verbose=1)

    mn = np.mean(time_callback.durations)
    std = np.std(time_callback.durations, ddof=1) if num_epochs > 1 else -1

    net_model.load_weights(saved_weights_path)
    net_predictions = [np.argmax(net_model.predict(np.expand_dims(feature, axis=0))) for feature in test_net]

    # report test accuracy
    test_accuracy = 100.*np.sum(np.array(net_predictions)==np.argmax(test_targets, axis=1))/float(len(net_predictions))
    print('[INFO] Test accuracy: %.4f%%' % test_accuracy)

    # generate a classification report for the model
    print("[INFO] Classification Report:")
    print(classification_report(np.argmax(test_targets, axis=1), 
                                net_predictions))
    # compute the raw accuracy with extra precision
    acc = accuracy_score(np.argmax(test_targets, axis=1), net_predictions)
    print("[INFO] score: {}".format(acc))

    # copy trained weights back to nextcloud
    dest_dir = cfg.Dog_RemoteStorage.rstrip('/') + '/models'
    print("[INFO] Upload %s to %s" % (saved_weights_path, dest_dir))
    dutils.rclone_copy(saved_weights_path, dest_dir)
    
    train_results = mutils.format_train(network, test_accuracy, num_epochs,
                                        data_size, time_prepare, mn, std)

    run_results["training"].append(train_results)

    return run_results


def get_train_args():

    train_args = cfg.train_args

    # convert default values and possible 'choices' into strings
    for key, val in train_args.items():
        val['default'] = str(val['default']) #yaml.safe_dump(val['default']) #json.dumps(val['default'])
        if 'choices' in val:
            val['choices'] = [str(item) for item in val['choices']]
        print(val['default'], type(val['default']))

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
        train(args)
        print("Elapsed time:  ", time.time() - start)
    else:
        get_metadata()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Model parameters')

    # get arguments configured for get_train_args()
    train_args = get_train_args()
    for key, val in train_args.items():
        parser.add_argument('--%s' % key,
                            default=val['default'],
                            type=type(val['default']), #may just put str
                            help=val['help'])
        print(key, val)
        print(type(val['default']))

    parser.add_argument('--method', type=str, default="get_metadata",
                        help='Method to use: get_metadata (default), \
                        predict_file, predict_data, predict_url, train')
    parser.add_argument('--file', type=str, help='File to do prediction on, full path')
    parser.add_argument('--url', type=str, help='URL with the image to do prediction on')

    args = parser.parse_args()
    print("Vars:", vars(args))
    
    main()

