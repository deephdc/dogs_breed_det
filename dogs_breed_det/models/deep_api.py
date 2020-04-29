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
import json
import argparse
import tempfile
import mimetypes 
import numpy as np
import pkg_resources
import dogs_breed_det.config as cfg
import dogs_breed_det.sys_info as sys_info
import dogs_breed_det.dataset.data_utils as dutils
import dogs_breed_det.dataset.make_dataset as mdata
import dogs_breed_det.models.model_utils as mutils
import dogs_breed_det.features.build_features as bfeatures

import keras
from keras import backend as K
#from keras import applications
#from keras.models import Model
#from keras import regularizers
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.layers.normalization import BatchNormalization

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

from aiohttp.web import HTTPBadRequest

## DEEPaaS wrapper to get e.g. UploadedFile() object
from deepaas.model.v2 import wrapper

from functools import wraps

## Authorization
from flaat import Flaat
flaat = Flaat()

# Switch for debugging in this script
debug_model = False 

def _catch_error(f):
    """Decorate function to return an error as HTTPBadRequest, in case
    """
    @wraps(f)    
    def wrap(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            raise HTTPBadRequest(reason=e)
    return wrap

def _fields_to_dict(fields_in):
    """
    Function to convert mashmallow fields to dict()
    """
    dict_out = {}
    
    for key, val in fields_in.items():
        param = {}
        param['default'] = val.missing
        param['type'] = type(val.missing)
        if key == 'files' or key == 'urls':
            param['type'] = str

        val_help = val.metadata['description']
        if 'enum' in val.metadata.keys():
            val_help = "{}. Choices: {}".format(val_help, 
                                                val.metadata['enum'])
        param['help'] = val_help

        try:
            val_req = val.required
        except:
            val_req = False
        param['required'] = val_req

        dict_out[key] = param
    return dict_out


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
    
    try:
        pkg = pkg_resources.get_distribution(module[0])
    except pkg_resources.RequirementParseError:
        # if called from CLI, try to get pkg from the path
        distros = list(pkg_resources.find_distributions(cfg.BASE_DIR, 
                                                        only=True))
        if len(distros) == 1:
            pkg = distros[0]
    except Exception as e:
        raise HTTPBadRequest(reason=e)

    # deserialize key-word arguments
    train_args = _fields_to_dict(get_train_args())
    # make 'type' JSON serializable
    for key, val in train_args.items():
        train_args[key]['type'] = str(val['type'])

    predict_args = _fields_to_dict(get_predict_args())
    # make 'type' JSON serializable
    for key, val in predict_args.items():
        predict_args[key]['type'] = str(val['type'])

    meta = {
        'name' : None,
        'version' : None,
        'summary' : None,
        'home-page' : None,
        'author' : None,
        'author-email' : None,
        'license' : None,
        'help-train' : train_args,
        'help-predict' : predict_args
    }

    for line in pkg.get_metadata_lines("PKG-INFO"):
        line_low = line.lower() # to avoid inconsistency due to letter cases
        for par in meta:
            if line_low.startswith(par.lower() + ":"):
                _, value = line.split(": ", 1)
                meta[par] = value
                
    return meta


def prepare_data(network='Resnet50'):
    """ Function to prepare data
    """
    mdata.prepare_data(network)


def build_model(network='Resnet50', num_classes=100):
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

    ### 'Standard' implementation based on bottlenecks
    #-train_net = bfeatures.load_features('train', network)
    net_model = Sequential()
    # add a global average pooling layer    
    #-net_model.add(GlobalAveragePooling2D(input_shape=train_net.shape[1:]))
    net_model.add(GlobalAveragePooling2D(input_shape=features_shape[network]))
    # add a fully-connected layer
    fc_layers = int(features_shape[network][2]/2.)
    net_model.add(Dense(fc_layers, activation='relu'))
    net_model.add(BatchNormalization())
    # add a classification layer
    net_model.add(Dense(num_classes, activation='softmax'))
    ###

    ### EXPERIMENTAL! # build the full ResNet50 network
    #base_model = applications.ResNet50(weights='imagenet', 
    #                                   include_top=False,
    #                                   input_shape=(224, 224, 3))
    #print('Model loaded.')
    #
    # build a classifier model to put on top of the convolutional model
    #top_model = Dropout(0.25)(base_model.output)
    #top_model = GlobalAveragePooling2D()(base_model.output)
    #top_model = Dense(128,
    #                  kernel_initializer='glorot_uniform',
    #                  kernel_regularizer=regularizers.l2(0.01),
    #                  activation='relu')(top_model)
    #top_model = Dropout(0.25)(top_model)
    #top_model = BatchNormalization()(top_model)
    #top_model = Dense(128, 
    #                  kernel_initializer='glorot_uniform',
    #                  kernel_regularizer=regularizers.l2(0.01),
    #                  activation='relu')(top_model)
    #top_model = BatchNormalization()(top_model)
    ##top_model = Dropout(0.5)(top_model)
    ##top_model = Flatten()(top_model)
    #predictions = Dense(num_classes, activation='softmax')(top_model)
    #
    #for layer in base_model.layers:
    #    layer.trainable = False    
    #
    #net_model = Model(inputs=base_model.input, outputs=predictions)
    ### END OF EXPERIMENTAL!

    print("__" + network+"__: ")
    net_model.summary()
    net_model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])
    
    return net_model

@_catch_error
def predict(**kwargs):
    
    print("predict(**kwargs) - kwargs: %s" % (kwargs)) if debug_model else ''

    if (not any([kwargs['urls'], kwargs['files']]) or
            all([kwargs['urls'], kwargs['files']])):
        raise Exception("You must provide either 'url' or 'data' in the payload")

    if kwargs['files']:
        kwargs['files'] = [kwargs['files']]  # patch until list is available
        return predict_data(kwargs)
    elif kwargs['urls']:
        kwargs['urls'] = [kwargs['urls']]  # patch until list is available
        return predict_url(kwargs)


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
    }

    # clear possible pre-existing sessions. IMPORTANT!
    K.clear_session()

    # check that all necessary data is there
    #prepare_data(network)

    weights_file = mutils.build_weights_filename(network)
    saved_weights_path = os.path.join(cfg.MODELS_DIR, weights_file) 

    # check if the weights file exists locally. if not -> try to download
    remote_dir = cfg.REMOTE_MODELS_DIR
    status_weights, _ = dutils.maybe_download_data(remote_dir,
                                                   local_dir=cfg.MODELS_DIR,
                                                   data_file=weights_file)

    # attempt to download default weights file                                                   
    if not status_weights and network in cfg.cnn_list:
        print("[INFO] Trying to download weights from the public link")
        url_weights = cfg.Dog_RemoteShare + weights_file
        status_weights, _ = dutils.url_download(
                                   url_path = url_weights, 
                                   local_dir=cfg.MODELS_DIR,
                                   data_file=weights_file)
                                   
    dog_names_file = cfg.Dog_LabelsFile.split('/')[-1]
    # check if the labels file exists locally. if not -> try to download
    remote_dir = cfg.REMOTE_DATA_DIR
    status_dog_names, _ = dutils.maybe_download_data(
                                                  remote_dir,
                                                  local_dir=cfg.DATA_DIR,
                                                  data_file=dog_names_file)

    # attempt to download labels file                                                   
    if not status_dog_names:
        print("[INFO] Trying to download labels from the public link")
        url_dog_names = cfg.Dog_RemoteShare + dog_names_file
        status_weights, _ = dutils.url_download(
                                   url_path = url_dog_names, 
                                   local_dir=cfg.DATA_DIR,
                                   data_file=dog_names_file)                                   

    if status_weights:
        dog_names  = dutils.dog_names_load()
        net_model = build_model(network, len(dog_names))
        net_model.load_weights(saved_weights_path)

        # extract bottleneck features
        bottleneck_feature = nets[network](dutils.path_to_tensor(img_path))
        if debug_model:
            print("[INFO] Bottleneck feature size: {}".format(
                                                     bottleneck_feature.shape))

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
        msg = "ERROR in predict_file(). No weights file found! " + \
              "Please first train the model with the " + network + " network!"
        msg = {"Error": msg}

    return msg


def predict_data(*args):
    """
    Function to make prediction on an uploaded file
    """
    print("predict_data(*args) - args: %s" % (args)) if debug_model else ''

    files = []

    for arg in args:
        file_objs = arg['files']
        for f in file_objs:
            files.append(f.filename)
            if debug_model:
                print("file_obj: name: {}, filename: {}, content_type: {}".format(
                                                               f.name,
                                                               f.filename,
                                                               f.content_type))
                print("File for prediction is at: {} \t Size: {}".format(
                                                  f.filename,
                                                  os.path.getsize(f.filename)))

        network = arg['network']

    prediction = []
    try:
        for imgfile in files:
            prediction.append(predict_file(imgfile, network))
            print("image: ", imgfile)
    except Exception as e:
        raise e
    finally:
        for imgfile in files:
            os.remove(imgfile)

    return prediction


def predict_url(*args):
    message = 'Not (yet) implemented in the model (predict_url())'
    message = {"Error": message}

    return message

@flaat.login_required() # Require only authorized people to do training
def train(**kwargs):
    """
    Train network (transfer learning)
    Parameters
    ----------
    https://docs.deep-hybrid-datacloud.eu/projects/deepaas/en/wip-api_v2/user/v2-api.html#deepaas.model.v2.base.BaseModel.train
    """
    print("train(**kwargs) - kwargs: %s" % (kwargs)) if debug_model else ''

    run_results = { "status": "ok",
                    "sys_info": [],
                    "training": [],
                  }

    # use the schema
    schema = cfg.TrainArgsSchema()
    # deserialize key-word arguments
    train_args = schema.load(kwargs)


    num_epochs = train_args['num_epochs']
    network = train_args['network']
    
    if debug_model:
        print("train_args:", train_args)
        print(type(train_args['num_epochs']), type(train_args['network']))
        print("Network:", train_args['network'], network)


    flag_sys_info = train_args['sys_info']
    print(flag_sys_info)
    if(flag_sys_info):
        sys_info.get_sys_info(cfg.machine_info)
        if debug_model:
            print("sys_info should be True: ", flag_sys_info)
            print(cfg.machine_info)
        run_results["sys_info"].append(cfg.machine_info)
    else:
        print("sys_info should be False: ", flag_sys_info) if debug_model else ''

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

    if debug_model:
        print('[INFO] Sizes of bottleneck_features (train, valid, test):')
        print(train_net.shape, valid_net.shape, test_net.shape)

    data_size = {
        'train': len(train_targets),
        'valid': len(valid_targets),
        'test': len(test_targets)
        }

    saved_weights_path = os.path.join(cfg.MODELS_DIR, 
                                      mutils.build_weights_filename(network))
    checkpointer = ModelCheckpoint(filepath=saved_weights_path, 
                                   verbose=1, save_best_only=True)

    # clear possible pre-existing sessions. important!
    K.clear_session()

    dog_names = dutils.dog_names_load()
    num_dog_breeds = len(dog_names)
    print('[INFO] Found %d classes (dogs breeds)' % num_dog_breeds)

    net_model = build_model(network, num_dog_breeds)

    before = K.get_session().run(net_model.trainable_weights)
    time_callback = TimeHistory()      
    net_model.fit(train_net, train_targets, 
                  validation_data=(valid_net, valid_targets),
                  epochs=num_epochs, batch_size=20, 
                  callbacks=[checkpointer, time_callback], verbose=1)
    after = K.get_session().run(net_model.trainable_weights)

    ## debug the neural network, can be removed from production    
    debug_here = False
    ii = 0
    for b, a in zip(before, after):
        if debug_here:
            print("[DEBUG] {} : ".format(net_model.trainable_weights[ii]))
            ii += 1
            if (b != a).any() and debug_here:
                print(" * ok, training (output does not match)")
            else:
                print(" * !!! output is the same, not training? !!!")
                print(" * Before: {} : ".format(b))
                print("")
                print(" * After: {} : ".format(a))
    ## end of "debug the neural network"

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
    dest_dir = cfg.REMOTE_MODELS_DIR
    print("[INFO] Upload %s to %s" % (saved_weights_path, dest_dir))
    dutils.rclone_copy(saved_weights_path, dest_dir)
    
    train_results = mutils.format_train(network, test_accuracy, num_epochs,
                                        data_size, time_prepare, mn, std)

    run_results["training"].append(train_results)

    return run_results


def get_train_args():
    """
    https://docs.deep-hybrid-datacloud.eu/projects/deepaas/en/wip-api_v2/user/v2-api.html#deepaas.model.v2.base.BaseModel.get_train_args
    https://marshmallow.readthedocs.io/en/latest/api_reference.html#module-marshmallow.fields
    :param kwargs:
    :return:
    """
    return cfg.TrainArgsSchema().fields


def get_predict_args():
    """
    https://docs.deep-hybrid-datacloud.eu/projects/deepaas/en/wip-api_v2/user/v2-api.html#deepaas.model.v2.base.BaseModel.get_predict_args
    :return:
    """
    return cfg.PredictArgsSchema().fields

# during development it might be practical 
# to check your code from the command line
def main():
    """
       Runs above-described functions depending on input parameters
       (see below an example)
    """

    if args.method == 'get_metadata':
        meta = get_metadata()
        print(json.dumps(meta))
        return meta 
    elif args.method == 'predict':
        ## use the schema
        #schema = cfg.PredictArgsSchema()
        #result = schema.load(vars(args))
    
        # TODO: change to many files ('for' itteration)
        if args.files:
            # create tmp file as later it will be deleted
            temp = tempfile.NamedTemporaryFile()
            temp.close()
            # copy original file into tmp file
            with open(args.files, "rb") as f:
                with open(temp.name, "wb") as f_tmp:
                    for line in f:
                        f_tmp.write(line)
        
            # create file object to mimic aiohttp workflow
            file_obj = wrapper.UploadedFile(name="data", 
                                            filename = temp.name,
                                            content_type=mimetypes.MimeTypes().guess_type(args.files)[0],
                                            original_filename=args.files)
            args.files = file_obj
        
        results = predict(**vars(args))
        print(json.dumps(results))
        return results        
    elif args.method == 'train':
        start = time.time()
        results = train(**vars(args))
        print("Elapsed time:  ", time.time() - start)
        print(json.dumps(results))
        return results


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Model parameters', 
                                     add_help=False)
    
    cmd_parser = argparse.ArgumentParser()    
    subparsers = cmd_parser.add_subparsers(
                            help='methods. Use \"model.py method --help\" to get more info', 
                            dest='method')

    get_metadata_parser = subparsers.add_parser('get_metadata', 
                                         help='get_metadata method',
                                         parents=[parser])

    # get train arguments configured
    train_parser = subparsers.add_parser('train', 
                                         help='commands for training',
                                         parents=[parser])
    train_args = _fields_to_dict(get_train_args())
    for key, val in train_args.items():
        train_parser.add_argument('--%s' % key,
                               default=val['default'],
                               type=val['type'], #may just put str
                               help=val['help'],
                               required=val['required'])

    # get predict arguments configured
    predict_parser = subparsers.add_parser('predict', 
                                           help='commands for prediction',
                                           parents=[parser])

    predict_args = _fields_to_dict(get_predict_args())
    for key, val in predict_args.items():
        predict_parser.add_argument('--%s' % key,
                               default=val['default'],
                               type=val['type'], #may just put str
                               help=val['help'],
                               required=val['required'])

    args = cmd_parser.parse_args()
   
    main()

