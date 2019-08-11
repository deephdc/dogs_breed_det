# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 - 2019 Karlsruhe Institute of Technology - Steinbuch Centre for Computing
# This code is distributed under the MIT License
# Please, see the LICENSE file
#
"""
Created on Sat Aug 10 08:43:00 2019

@author: vykozlov
"""
import os
import unittest
import dogs_breed_det.config as cfg
import dogs_breed_det.models.model as dog_model

from keras.utils import plot_model

class TestModelFunc(unittest.TestCase):
    def test_predict_file(self):
        """
        Functional test of predict_file
        Also visualizes the neural network
        """
        img_path = os.path.join(cfg.BASE_DIR,'dogs_breed_det/tests/inputs/St_Bernard_wiki_3.jpg')
        network_image = os.path.join(cfg.BASE_DIR,'dogs_breed_det/tests/tmp/neural_net.png')
        network = 'Resnet50'
        prob_cut = 0.5
      
        pred_result = dog_model.predict_file(img_path, network)

        prob = 0.0
        for pred in pred_result["predictions"]:
            print("prob: ", pred["probability"])
            print("label: ", pred["label"])
            label = pred["label"]
            if label == 'Saint_bernard':
                prob = pred["probability"]
    
        model = dog_model.build_model(network)

        # comment for now to pass tests.        
        #plot_model(model, to_file=network_image)
        # print model summary
        model.summary()
        print("prob Saint_bernard: ", prob)
    
        assert prob > prob_cut

if __name__ == '__main__':
    unittest.main()
    #test_predict_file()