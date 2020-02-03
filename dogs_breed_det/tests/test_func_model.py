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

#from keras.utils import plot_model

debug = False

class TestModelFunc(unittest.TestCase):
    def setUp(self):
        self.img_path = os.path.join(
                            cfg.BASE_DIR,
                           'dogs_breed_det/tests/inputs/St_Bernard_wiki_3.jpg')
        self.network_image = os.path.join(
                            cfg.BASE_DIR,
                           'dogs_breed_det/tests/tmp/neural_net.png')
        self.network = 'Resnet50'
        self.prob_cut = 0.5

    def test_build_network(self):
        """
        Functional test if we can build the model (model_build())
        by checking if get_config() returns list
        """
        print("[test_build_network]")
        model = dog_model.build_model(self.network)

        # comment for now to pass tests.        
        #plot_model(model, to_file=network_image)
        config = model.get_config()
        self.assertTrue(type(config) is list)

        
    def test_predict_file(self):
        """
        Functional test of predict_file
        Also visualizes the neural network -> skip
        """
        print("[test_predict_file]")
        pred_results = dog_model.predict_file(self.img_path, 
                                              self.network)

        prob = 0.0
        for pred in pred_results:
            print("prob: ", pred["probability"]) if debug else ''
            print("label: ", pred["label"])  if debug else ''
            label = pred["label"]
            if label == 'Saint_bernard':
                prob = pred["probability"]

        print("prob Saint_bernard: ", prob)
    
        assert prob > self.prob_cut

if __name__ == '__main__':
    unittest.main()
    #test_predict_file()