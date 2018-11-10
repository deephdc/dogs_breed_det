# -*- coding: utf-8 -*-
"""
   Module to prepare the dataset
"""
import logging
import argparse
from pathlib2 import Path
from dotenv import find_dotenv, load_dotenv
import dogs_breed_det.config as cfg
import dogs_breed_det.dataset.data_utils as dutils
import dogs_breed_det.features.build_features as bfeatures

def prepare_data(network='Resnet50'):
    """ Function to prepare data
    """
    
    # check if dog_names file exists locally, if not -> download,
    # if not downloaded -> dutils.dog_names_create()
    labels_file = cfg.Dog_LabelsFile.split('/')[-1]
    status_labels, _ = dutils.maybe_download_data(data_dir='/data', 
                                                  data_file=labels_file)

    if not status_labels:
        dutils.dog_names_create()
    else:
        print("[INFO] %s exists" % (cfg.Dog_LabelsFile))
                                                        
    # check if bottleneck_features file fexists locally
    # if not -> download it, if not downloaded -> try to build
    # train
    status = { True: "exists", False: "does not exist"}
    status_train = bfeatures.check_features_set('train', network)
    print("[INFO] Bottleneck file for train (%s) %s" % (network, status[status_train]))
    # valid
    status_valid = bfeatures.check_features_set('valid', network)
    print("[INFO] Bottleneck file for valid (%s) %s" % (network, status[status_valid]))
    # test
    status_test = bfeatures.check_features_set('test', network)
    print("[INFO] Bottleneck file for tests (%s) %s" % (network, status[status_test]))

def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed.
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    
    print("__%s__" % (args.network))
    prepare_data(args.network)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    
    parser = argparse.ArgumentParser(description='Model parameters')
    parser.add_argument('--network', type=str, default="Resnet50",
                        help='Neural network to use: Resnet50, InceptionV3,\
                        VGG16, VGG19, Xception')
    args = parser.parse_args()

    main()
