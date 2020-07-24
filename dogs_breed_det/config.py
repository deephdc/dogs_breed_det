# -*- coding: utf-8 -*-
"""
   Module to define CONSTANTS used across the project
"""
import os
from webargs import fields
from marshmallow import Schema, INCLUDE

# identify basedir for the package
BASE_DIR = os.path.dirname(os.path.normpath(os.path.dirname(__file__)))

# default location for input and output data, e.g. directories 'data' and 'models',
# is either set relative to the application path or via environment setting
IN_OUT_BASE_DIR = BASE_DIR
if 'APP_INPUT_OUTPUT_BASE_DIR' in os.environ:
    env_in_out_base_dir = os.environ['APP_INPUT_OUTPUT_BASE_DIR']
    if os.path.isdir(env_in_out_base_dir):
        IN_OUT_BASE_DIR = env_in_out_base_dir
    else:
        msg = "[WARNING] \"APP_INPUT_OUTPUT_BASE_DIR=" + \
        "{}\" is not a valid directory! ".format(env_in_out_base_dir) + \
        "Using \"BASE_DIR={}\" instead.".format(BASE_DIR)
        print(msg)

DATA_DIR = os.path.join(IN_OUT_BASE_DIR, 'data')
MODELS_DIR = os.path.join(IN_OUT_BASE_DIR, 'models')
    
Dog_RemoteSpace = 'rshare:/deep-oc-apps/dogs_breed_det/'
Dog_RemoteShare = 'https://nc.deep-hybrid-datacloud.eu/s/D7DLWcDsRoQmRMN/download?path=%2F&files='
Dog_DataDir = 'dogImages'
Dog_WeightsPattern = 'weights.best.NETWORK.3layers.hdf5'
Dog_LabelsFile = os.path.join(DATA_DIR, 'dog_names.txt')

REMOTE_DATA_DIR = os.path.join(Dog_RemoteSpace, 'data')
REMOTE_MODELS_DIR = os.path.join(Dog_RemoteSpace, 'models')

# FLAAT needs a list of trusted OIDC Providers. Here is an extended example:
#[
#'https://b2access.eudat.eu/oauth2/',
#'https://b2access-integration.fz-juelich.de/oauth2',
#'https://unity.helmholtz-data-federation.de/oauth2/',
#'https://login.helmholtz-data-federation.de/oauth2/',
#'https://login-dev.helmholtz.de/oauth2/',
#'https://login.helmholtz.de/oauth2/',
#'https://unity.eudat-aai.fz-juelich.de/oauth2/',
#'https://services.humanbrainproject.eu/oidc/',
#'https://accounts.google.com/',
#'https://aai.egi.eu/oidc/',
#'https://aai-dev.egi.eu/oidc/',
#'https://login.elixir-czech.org/oidc/',
#'https://iam-test.indigo-datacloud.eu/',
#'https://iam.deep-hybrid-datacloud.eu/',
#'https://iam.extreme-datacloud.eu/',
#'https://oidc.scc.kit.edu/auth/realms/kit/',
#'https://proxy.demo.eduteams.org'
#]
#
# we select following three providers:
Flaat_trusted_OP_list = [
'https://aai.egi.eu/oidc/',
'https://iam.deep-hybrid-datacloud.eu/',
'https://iam.extreme-datacloud.eu/',
]

machine_info = { 'cpu': '',
                 'gpu': '',
                 'memory_total': '',
                 'memory_available': ''
               }

cnn_list = ['Resnet50', 'InceptionV3', 'VGG16', 'VGG19']

# class / place to describe arguments for predict()
class PredictArgsSchema(Schema):
    class Meta:
        unknown = INCLUDE  # supports extra parameters

    network = fields.Str(
        required=False,
        missing=cnn_list[0],
        enum=cnn_list,
        description="Neural model to use for prediction"
    )
    
    files = fields.Field(
        required=False,
        missing=None,
        type="file",
        data_key="data",
        location="form",
        description="Select the image you want to classify."
    )
    
    urls = fields.Url(
        required=False,
        missing=None,
        description="Select an URL of the image you want to classify."
    )


# class / place to describe arguments for train()
class TrainArgsSchema(Schema):
    class Meta:
        unknown = INCLUDE  # supports extra parameters
        
    num_epochs = fields.Integer(
        required=False,
        missing=1,
        description="Number of training epochs")

    network = fields.Str(
        required=False,
        missing=cnn_list[0],
        enum=cnn_list,
        description="Neural model to use")

    sys_info = fields.Boolean(
        required=False,
        missing=False,
        enum=[True, False],
        description="Print information about the system (e.g. cpu, gpu, memory)")
