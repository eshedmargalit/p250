# PSYCH250 Final Project
## Identifying a V1-like layer in Deep Convolutional Neural Networks
#### Eshed Margalit

## Installation
Requirements:
```
python 2.7+
GPU with CUDA libraries installed to support tensorflow
```

Install the TNN library:
```bash
git clone https://github.com/neuroailab/tnn.git
cd tnn
pip install .
```

Install tfutils:
```bash
pip install git+https://github.com/neuroailab/tfutils.git
```

Install the p250 project:
```
git clone https://github.com/eshedmargalit/p250.git
cd p250
python setup.py install
```

## Using the scripts
Because this project is based on pre-trained models, you need access to the appropriate model checkpoints. Due to their size, they are not uploaded here. Similarly, `.tfrecord` files with test set stimuli are not uploaded. Please contact Eshed for access.

#### `scripts/01_get_features/extract_features.py`
Used to save an HDF5 file with model features. Example use case:
`python extract-features.py --gpu 1 --exp_id vgg19_slim --config_name sineff_20190507`

#### `scripts/02_compute_tuning_curves.py`
Given the outputs of the previous script, computes tuning curves for orientation, spatial frequency, color, and phase of a drifting Gabor test set.

#### `scripts/03_circular_variance.py`
Given the tuning curves, computes circular variance for orientation tuning and compares the distribution of each model layer to published data from Ringach et al., 2002

## Results
TODO

## Conclusions
TODO
