# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 - 2019 Karlsruhe Institute of Technology - Steinbuch Centre for Computing
# This code is distributed under the MIT License
# Please, see the LICENSE file
#
"""
Created on Sat Aug 10 08:47:51 2019

@author: vykozlov
"""
import unittest
import numpy as np
import dogs_breed_det.models.model as dog_model

debug = False

class TestModelMethods(unittest.TestCase):
    def test_model_variables(self):
        import keras.backend as K
        network = 'Resnet50'
        num_classes = 133

        model = dog_model.build_model(network, num_classes)
        print(model.trainable_weights)
        train_tensor = np.random.normal(size=(1, 1, 1, 2048))
        label_tensor = np.random.normal(size=(1, num_classes))

        before = K.get_session().run(model.trainable_weights)
        model.fit(train_tensor, label_tensor,
                  epochs=1, batch_size=1, 
                  verbose=1)
        after = K.get_session().run(model.trainable_weights)

        # Make sure something changed.
        i = 0
        for b, a in zip(before, after):
            # Make sure something changed.
            assert (b != a).any()

            if debug:
                print("[DEBUG] {} : ".format(model.trainable_weights[i]))
                i += 1
                if (b != a).any() and debug:
                    print(" * output does not match, i.e. training")
                else:
                    print(" * !!! output is the same, not training? !!!")

#test_model_variables()

if __name__ == '__main__':
    unittest.main()
