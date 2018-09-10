# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 21:29:57 2018

@author: valentin
"""

import os
import tempfile
import numpy as np
import werkzeug.exceptions as exceptions
import dogs_breed_det.config as cfg
import dogs_breed_det.dataset.data_utils as dutils
import dogs_breed_det.models.model_utils as mutils
import dogs_breed_det.features.build_features as bfeatures
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras import backend


def get_metadata():
    name = "Dogs_breed"
    d = {
        "id": "0",
        "name": name,
        "description": ("This is a model for dogs breed detection"
                        "(loaded '%s')" % name),
        "author": 'Valentin Kozlov',
        "version": "0.3.0",
    }
#    d = {
#        "author": None,
#        "description": None,
#        "url": None,
#        "license": None,
#        "version": None,
#    }

    return d
        

def build_model(network='Resnet50'):
    """
    Build network. Possible nets:
    Resnet50, Xception, InceptionV3, Resnet50, VGG19, VGG16
    """
    
    train_net, _, _ = bfeatures.build_features(network)

    net_model = Sequential()
    net_model.add(GlobalAveragePooling2D(input_shape=train_net.shape[1:]))
    net_model.add(Dense(133, activation='softmax'))
    
    net_model.summary()
    net_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    
    return net_model
        

def predict_file(img_path):
    """
    Function to make prediction which breed is closest
    :param img_path: image to classify, full path
    :return: most probable dogs breed
    """

    # clear possible pre-existing sessions. important!
    backend.clear_session()
    
    net_model = build_model('Resnet50')
    saved_weights_path = os.path.join(cfg.basedir,'models','weights.best.Resnet50.hdf5')
    net_model.load_weights(saved_weights_path)
    
    # extract bottleneck features
    bottleneck_feature = bfeatures.extract_Resnet50(dutils.path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = net_model.predict(bottleneck_feature)
    print("Sum:", np.sum(predicted_vector))
    # return dog breed that is predicted by the model
    idxs = np.argsort(predicted_vector[0])[::-1][:5] 
    dog_names  = dutils.dog_names_read(cfg.dogNamesFile)
    #dog_names_best = [ dog_names[i] for i in idxs ]
    dog_names_best = []
    probs_best = []
    for i in idxs:
        dog_names_best.append(dog_names[i])
        probs_best.append(predicted_vector[0][i])
        print(dog_names[i], " : ", predicted_vector[0][i]) 

    return mutils.format_prediction(dog_names_best, probs_best)


def predict_data(img):
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
            prediction.append(predict_file(imgfile))
    except Exception as e:
        raise e
    finally:
        for imgfile in filenames:
            os.remove(imgfile)

    return prediction


def predict_url(*args):
    message = 'Not (yet) implemented in the model)'
    return message
        

def train(nepochs=20, model='Resnet50'):
    """
    Train network (transfer learning)
    """
    dogImagesdir = os.path.join(cfg.basedir,'data','dogImages')
    _, train_targets = dutils.load_dataset(os.path.join(dogImagesdir,'train'))
    _, valid_targets = dutils.load_dataset(os.path.join(dogImagesdir,'valid'))
    _, test_targets = dutils.load_dataset(os.path.join(dogImagesdir,'test'))

    train_net, valid_net, test_net = bfeatures.build_features(model)
    data_size = {
        'train': len(train_targets),
        'valid': len(valid_targets),
        'test': len(test_targets)
        }
    
    saved_weights_path = os.path.join(cfg.basedir,'models','weights.best.Resnet50.hdf5')
    checkpointer = ModelCheckpoint(filepath=saved_weights_path, verbose=1, save_best_only=True)
 
    net_model = build_model(model)
        
    net_model.fit(train_net, train_targets, 
                  validation_data=(valid_net, valid_targets),
                  epochs=nepochs, batch_size=20, callbacks=[checkpointer], verbose=1)
    
    net_model.load_weights(saved_weights_path)
    net_predictions = [np.argmax(net_model.predict(np.expand_dims(feature, axis=0))) for feature in test_net]
    
    # report test accuracy
    test_accuracy = 100.*np.sum(np.array(net_predictions)==np.argmax(test_targets, axis=1))/float(len(net_predictions))
    print('Test accuracy: %.4f%%' % test_accuracy)
    

    return mutils.format_train("Resnet50", test_accuracy, nepochs, data_size)
