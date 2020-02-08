# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 - 2019 Karlsruhe Institute of Technology - Steinbuch Centre for Computing
# This code is distributed under the MIT License
# Please, see the LICENSE file
#
"""
   Module to prepare the dataset
"""
import os
import logging
import argparse
from pathlib2 import Path
from dotenv import find_dotenv, load_dotenv
import dogs_breed_det.config as cfg
import dogs_breed_det.dataset.data_utils as dutils
import dogs_breed_det.features.build_features as bfeatures


def check_targets(data_type):
    """Check if targets file exists locally
       Only one dataset, e.g. train, valid, test is checked
    """
    remote_dir = cfg.REMOTE_DATA_DIR   
    targets_file = 'Dogs_targets_' + data_type + '.npz'
    targets_path = os.path.join(cfg.DATA_DIR, targets_file)
    targets_exists, _ = dutils.maybe_download_data(remote_dir,
                                                   cfg.DATA_DIR,
                                                   data_file=targets_file)

    if not targets_exists:
        print("[INFO] %s was neither found nor downloaded. Trying to build .. "
              % targets_file)

        # check if directory with train, test, and valid images exists:
        dutils.maybe_download_and_unzip()
        dutils.build_targets(data_type)

        # Upload to nextcloud newly created file
        targets_exists = True if os.path.exists(targets_path) else False
        print("[INFO] Upload %s to %s" % (targets_path, remote_dir))    
        dutils.rclone_copy(targets_path, remote_dir)

    return targets_exists


def check_features(data_type, network='Resnet50'):
    """Check if features file exists locally
       Only one dataset, e.g. train, valid, test is checked
    """
    
    remote_dir=os.path.join(cfg.REMOTE_DATA_DIR, 'bottleneck_features')
    bottleneck_path = bfeatures.set_features_file(data_type, network)
    data_dir, bottleneck_file = os.path.split(bottleneck_path)
    bottleneck_exists, _ = dutils.maybe_download_data(remote_dir=remote_dir,
                                                      local_dir=data_dir,
                                                      data_file=bottleneck_file)

    if not bottleneck_exists:
        print("[INFO] %s was neither found nor downloaded. Trying to build. It may take time .. "
              % bottleneck_file)

        # check if directory with train, test, and valid images exists:
        dutils.maybe_download_and_unzip()
        bfeatures.build_features(data_type, network)
        
        # Upload to nextcloud newly created file
        bottleneck_exists = True if os.path.exists(bottleneck_path) else False
        print("[INFO] Upload %s to %s" % (bottleneck_path, remote_dir))
        dutils.rclone_copy(bottleneck_path, remote_dir)
        
    return bottleneck_exists


def prepare_data(network='Resnet50'):
    """ Function to prepare data
    """
   
    # check if dog_names file exists locally, if not -> download,
    # if not downloaded -> dutils.dog_names_create()
    remote_dir = os.path.join(cfg.Dog_RemoteStorage, 'data')
    dog_names_file = cfg.Dog_LabelsFile.split('/')[-1]
    status_dog_names, _ = dutils.maybe_download_data(remote_dir,
                                                     local_dir=cfg.DATA_DIR,
                                                     data_file=dog_names_file)

    if not status_dog_names:
        dutils.maybe_download_and_unzip()
        dutils.dog_names_create()
    else:
        print("[INFO] %s exists" % (cfg.Dog_LabelsFile))

    # check if bottleneck_features file exists locally
    # if not -> download it, if not downloaded -> try to build
    # train
    status = { True: "exists", False: "does not exist"}
    datasets = ['train', 'valid', 'test']
    for dset in datasets:
        status_targets = check_targets(dset)
        print("[INFO] Targets file for %s %s" % (dset, status[status_targets]))
        status_bottleneck = check_features(dset, network)
        print("[INFO] Bottleneck file for %s (%s) %s" % 
               (dset, network, status[status_bottleneck]))


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
