# -*- coding: utf-8 -*-
"""Functions to make predictions"""

import numpy as np
import tempfile
import os
import dogs_breed.config as cfg
import dogs_breed.dataset.data_utils as dutils
import dogs_breed.models.model_utils as mutils
import dogs_breed.features.build_features as bfeatures
from dogs_breed.models.build_model import build_model
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.applications.resnet50 import ResNet50
from keras import backend


# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')

def ResNet50_predict_labels(img_path):
    """
    returns prediction vector for image located at img_path based on Resnet50 neural network
    :param img_path: image to classify, full path
    :return: prediction vector
    """
    
    img = preprocess_input(dutils.path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

def dog_detector(img_path):
    """
    returns "True" if a dog is detected in the image stored at img_path
    :param img_path: image to classify, full path
    :return: True if dog detected, False otherwise
    """
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 

def predict_local(img_path, network='Resnet50'):
    """
    Function to make prediction which breed is closest
    :param img_path: image to classify, full path
    :param network: neural network to be used
    :return: most probable dogs breed
    """
    nets = {'VGG16': bfeatures.extract_VGG16,
            'VGG19': bfeatures.extract_VGG19,
            'Resnet50': bfeatures.extract_Resnet50,
            'InceptionV3': bfeatures.extract_InceptionV3,
            'Xception': bfeatures.extract_Xception,
    }

    # clear possible pre-existing sessions. important!
    backend.clear_session()
    
    net_model = build_model(network)  
    saved_weights_path = os.path.join(cfg.basedir,'models','weights.best.' + network + '.hdf5')
    net_model.load_weights(saved_weights_path)
    
    # extract bottleneck features
    bottleneck_feature = nets[network](dutils.path_to_tensor(img_path))
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

def predict_model(img):
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
            prediction.append(predict_local(imgfile))
    except Exception as e:
        raise e
    finally:
        for imgfile in filenames:
            os.remove(imgfile)

    return prediction

