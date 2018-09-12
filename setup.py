# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    #setup_requires=['pbr>=1.8'],
    #pbr=True,
    
    name='dogs_breed_det',
    
    version='0.2.0',
    description='DEEP as a Service (DEEPaaS) is a REST API to expose a machine learning model through a REST API.',
    long_description=long_description,
    author='Valentin Kozlov',
    author_email='valentin.kozlov@gmail.com',
    license='MIT',
    url='http://github.com/vykozlov/dogs_breed_det',
    classifiers=[
        'Intended Audience :: Information Technology',
        'Intended Audience :: System Administrators',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.6'
    ],
    
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),

    entry_points={
        'deepaas.model': [
            'Dogs_Resnet50=dogs_breed_det.models.resnet50',
            'Dogs_VGG16=dogs_breed_det.models.vgg16',
            'Dogs_Xception=dogs_breed_det.models.xception',            
        ],
    },
    
)
