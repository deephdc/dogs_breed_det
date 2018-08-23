# -*- coding: utf-8 -*-
"""Functions to train neural network"""

import numpy as np
import dogs_breed_det.config as cfg
import dogs_breed_det.dataset.data_utils as dutils
import dogs_breed_det.models.model_utils as mutils
from dogs_breed_det.features.build_features import build_features
from dogs_breed_det.models.build_model import build_model
from keras.callbacks import ModelCheckpoint
from os import path


def train_model(network = 'Resnet50', nepochs=20):
    """
    Train network (transfer learning)
    """
    dogImagesdir = path.join(cfg.basedir,'data','dogImages')
    _, train_targets = dutils.load_dataset(path.join(dogImagesdir,'train'))
    _, valid_targets = dutils.load_dataset(path.join(dogImagesdir,'valid'))
    _, test_targets = dutils.load_dataset(path.join(dogImagesdir,'test'))

    train_net, valid_net, test_net = build_features(network)
    data_size = {
        'train': len(train_targets),
        'valid': len(valid_targets),
        'test': len(test_targets)
    }
    
    saved_weights_path = path.join(cfg.basedir,'models','weights.best.'+network+'.hdf5')
    checkpointer = ModelCheckpoint(filepath=saved_weights_path, verbose=1, save_best_only=True)
 
    net_model = build_model(network)  

    net_model.fit(train_net, train_targets, 
          validation_data=(valid_net, valid_targets),
          epochs=nepochs, batch_size=20, callbacks=[checkpointer], verbose=1)
    
    net_model.load_weights(saved_weights_path)
    net_predictions = [np.argmax(net_model.predict(np.expand_dims(feature, axis=0))) for feature in test_net]
    
    # report test accuracy
    test_accuracy = 100.*np.sum(np.array(net_predictions)==np.argmax(test_targets, axis=1))/float(len(net_predictions))
    print('Test accuracy: %.4f%%' % test_accuracy)
    
    
    return mutils.format_train(network, test_accuracy, nepochs, data_size)
