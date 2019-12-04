Dog's breed detector
==============================

[![Build Status](https://jenkins.indigo-datacloud.eu/buildStatus/icon?job=Pipeline-as-code%2FDEEP-OC-org%2Fdogs_breed_det%2Fmaster)](https://jenkins.indigo-datacloud.eu/job/Pipeline-as-code/job/DEEP-OC-org/job/dogs_breed_det/job/master/)


An application to identify Dog's breed, "Dogs breed detector", using deep learning. **133** breeds are known.

[DEEPaaS API](https://github.com/indigo-dc/DEEPaaS) is used to access the model functionality.

Dogs breed detector is originally forked from [udacity/dogs-project](https://github.com/udacity/dog-project), dataset comes from [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).

The project applies Transfer learning for dog's breed identification, implemented with Tensorflow and Keras:

From a pre-trained model (VGG16 | VGG19 | Resnet50 | InceptionV3 | Xception) the last layer is removed, then new FC classification layers are added, which is trained. All images first pass through the pre-trained network and converted into the tensor with the shape of the 'before-last' layer of the pre-trained network, into so-called 'bottleneck_features'. These bottleneck_features are used then as input for the FC classification network.



Project Organization
------------

    ├── LICENSE
    ├── README.md              <- The top-level README for developers using this project.
    ├── data                   <- Data placeholde
    │
    ├── docs                   <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models                 <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks              <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                             the creator's initials (if many user development),
    │                             and a short `_` delimited description, e.g.
    │                             `1.0-jqp-initial_data_exploration.ipynb`.
    │
    ├── references             <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports                <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures            <- Generated graphics and figures to be used in reporting
    │
    ├── requirements-dev.txt   <- The requirements file for the development environment
    │
    ├── test-requirements.txt  <- The requirements file for the test environment
    │
    ├── requirements.txt       <- The requirements file for reproducing the analysis environment, e.g.
    │                             generated with `pip freeze > requirements.txt`
    ├── setup.cfg              <- makes project pip installable (pip install -e .) so dogs_breed_det can be imported
    ├── setup.py               <- makes project pip installable (pip install -e .) so dogs_breed_det can be imported
    ├── dogs_breed_det    <- Source code for use in this project.
    │   ├── __init__.py        <- Makes dogs_breed_det a Python module
    │   │
    │   ├── dataset            <- Scripts to download or generate data
    │   │
    │   ├── features           <- Scripts to turn raw data into features for modeling
    │   │
    │   ├── models             <- Scripts to train models and then use trained models to make
    │   │                         predictions
    │   │
    │   └── tests              <- Scripts to perfrom code testing + pylint script
    │   │
    │   └── visualization      <- Scripts to create exploratory and results oriented visualizations
    │
    └── tox.ini                <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
