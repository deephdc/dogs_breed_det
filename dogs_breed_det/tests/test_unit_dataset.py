# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 - 2019 Karlsruhe Institute of Technology - Steinbuch Centre for Computing
# This code is distributed under the MIT License
# Please, see the LICENSE file
#
"""
Created on Fri Aug  9 20:52:28 2019

@author: vykozlov
"""
import os
import numpy as np
import dogs_breed_det.config as cfg
import dogs_breed_det.dataset.data_utils as dutils
import dogs_breed_det.models.model_utils as mutils

import unittest

class TestDatasetMethods(unittest.TestCase):

    def test_url_download(self):
        url_path = cfg.Dog_RemoteShare
        data_dir = os.path.join(cfg.BASE_DIR, 'dogs_breed_det/tests/tmp')
        weights_file = mutils.build_weights_filename('Resnet50')
    
        dist_exist, error_out = dutils.url_download(url_path, data_dir, weights_file)
        if dist_exist:
            os.remove(os.path.join(data_dir, weights_file))
    
        assert dist_exist

    def test_load_data_files(self):
        path = os.path.join(cfg.BASE_DIR,'dogs_breed_det/tests/inputs/train')
        file_list_ref = np.array([os.path.join(path, '001.Affenpinscher/Affenpinscher_00001.jpg'),
                                  os.path.join(path, '001.Affenpinscher/Affenpinscher_00002.jpg'),
                                  os.path.join(path, '016.Beagle/Beagle_01129.jpg'),
                                  os.path.join(path, '016.Beagle/Beagle_01131.jpg')])
        file_list = dutils.load_data_files(path)
        print("Found: ", np.sort(file_list))
        print("Ref: ", file_list_ref)
    
        assert (file_list_ref == np.sort(file_list)).all()
    
    def test_dog_names_create(self):
        train_path = os.path.join(cfg.BASE_DIR,'dogs_breed_det/tests/inputs/train', '*')
        labels_path = os.path.join(cfg.BASE_DIR,'dogs_breed_det/tests/tmp/tmp_dog_names.txt')
        dog_names_ref = ['Affenpinscher', 'Beagle']
        dog_names = dutils.dog_names_create(train_path, labels_path)
        if os.path.exists(labels_path):
            os.remove(labels_path)

        print("Found: ", dog_names)
        print("Ref: ", dog_names_ref)
    
        self.assertEqual(dog_names, dog_names_ref)
    
    def test_dog_names_load(self):
        labels_path = os.path.join(cfg.BASE_DIR,'dogs_breed_det/tests/inputs/dog_names.txt')
        dog_names_ref = ['Affenpinscher', 'Beagle', 'Collie', 'Mastiff', 'Saint_bernard']
        dog_names = dutils.dog_names_load(labels_path)
        print("Found: ", dog_names)
        print("Ref: ", dog_names_ref)
    
        self.assertEqual(dog_names, dog_names_ref)


if __name__ == '__main__':
    unittest.main()