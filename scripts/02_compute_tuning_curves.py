"""
For each model, compute tuning curves from the sineff_20190507 stimulus set
"""

import os

import deepdish as dd
import numpy as np

from p250.utils.file_utils import get_features
from p250.utils.tuning_curves import compute_tuning_curves

## Resolve project path
PROJ_PATH = os.environ.get("P250_PROJ_PATH")
if PROJ_PATH is None:
    PROJ_PATH = "/mnt/fs6/eshedm"

# Globals: point to where tuning curves should be saved
TC_DIR = PROJ_PATH + '/tuning_curves_mean'


def compute_all_tc():
    print("\nComputing TC for vgg19_slim_all")
    print("-------------------------------------------------------------")
    vgg19_slim_all()

    print("\nComputing TC for alexnet_allrelu")
    print("-------------------------------------------------------------")
    alexnet_allrelu()

    print("\nComputing TC for tnn")
    print("-------------------------------------------------------------")
    tnn()

def vgg19_slim_all():
    model_name = 'vgg19_slim_step_0'
    feature_path = PROJ_PATH + '/extracted_features/%s_sineff_20190507_features_all.h5' % model_name

    save_dir = '%s/20190507_sineff/%s_all' % (TC_DIR, model_name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    layers = [
        "conv1_1",
        "conv1_2",
        "conv2_1",
        "conv2_2",
        "conv3_1",
        "conv3_2",
        "conv3_3",
        "conv3_4",
        "conv4_1",
        "conv4_2",
        "conv4_3",
        "conv4_4",
        "conv5_1",
        "conv5_2",
        "conv5_3",
        "conv5_4",
    ]

    layer_names = ["%s_relu" % x for x in layers]
    files_exist = [os.path.isfile("%s/%s.h5" % (save_dir, ln)) for ln in layer_names]
    if all(files_exist):
        print('\tAll tuning curves already computed')
        return
    else:
        print('\tAt least one tuning curve needs to be computed, proceeding')

    keys = ['labels']
    label_features = get_features(
        feature_path,
        group_name='sineff_20190507_features',
        keys=keys,
        out_keys=keys,
        show_keys=False,
        verbose=True,
    )
    labels = label_features['labels']
    param_names = [
        "angles",
        "sfs",
        "phases",
        "colors"
    ]

    # extract labels
    for layer_name in layer_names:
        print("Processing %s" % layer_name)

        keys = [layer_name]
        layer_features = get_features(
            feature_path,
            group_name='sineff_20190507_features',
            keys=keys,
            out_keys=keys,
            show_keys=False,
            verbose=True,
        )
        layer_feat = layer_features[layer_name]
        flat_layer = layer_feat.reshape((layer_feat.shape[0], -1))

        fpath = "%s/%s.h5" % (save_dir, layer_name)
        if os.path.isfile(fpath):
            print("Tuning curve dictionary already exists at %s" % fpath)
        else:
            print("Saving tuning curves to %s" % fpath)
            tuning_curves = compute_tuning_curves(labels,
                                                  flat_layer.T,
                                                  passing_indices=None,
                                                  agg_func=np.mean,
                                                  verbose=True,
                                                  param_names=param_names)
            dd.io.save(fpath, tuning_curves)

def alexnet_allrelu():
    model_name = 'alexnet-baseline-0-allrelu_step_115000'
    feature_path = PROJ_PATH + '/extracted_features/%s_sineff_20190507_features.h5' % model_name

    save_dir = '%s/20190507_sineff/%s' % (TC_DIR, model_name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    layer_names = ["conv%d_relu" % (x + 1) for x in range(5)]
    files_exist = [os.path.isfile("%s/%s.h5" % (save_dir, ln)) for ln in layer_names]
    if all(files_exist):
        print('\tAll tuning curves already computed')
        return
    else:
        print('\tAt least one tuning curve needs to be computed, proceeding')

    keys = [
        'labels',
        'conv1_relu',
        'conv2_relu',
        'conv3_relu',
        'conv4_relu',
        'conv5_relu',
    ]

    features = get_features(
        feature_path,
        group_name='sineff_20190507_features',
        keys=keys,
        out_keys=keys,
        show_keys=True,
        verbose=True,
    )

    # make a list of features to use
    layers = [features[x] for x in layer_names]
    flat_layers = [x.reshape((x.shape[0], -1)) for x in layers]

    # extract labels
    labels = features["labels"]
    param_names = [
        "angles",
        "sfs",
        "phases",
        "colors"
    ]

    for layer_name, flat_layer in zip(layer_names, flat_layers):
        print("Processing %s" % layer_name)

        fpath = "%s/%s.h5" % (save_dir, layer_name)

        if os.path.isfile(fpath):
            print("Tuning curve dictionary already exists at %s" % fpath)
        else:
            print("Saving tuning curves to %s" % fpath)
            tuning_curves = compute_tuning_curves(labels,
                                                  flat_layer.T,
                                                  passing_indices=None,
                                                  agg_func=np.mean,
                                                  verbose=True,
                                                  param_names=param_names)
            dd.io.save(fpath, tuning_curves)

def tnn():
    model_name = 'tnn_step_0'
    feature_path = PROJ_PATH + '/extracted_features/%s_sineff_20190507_features_tnn.h5' % model_name

    save_dir = '%s/20190507_sineff/%s' % (TC_DIR, model_name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    layer_names = ["conv%d" % x for x in range(1, 11)]
    files_exist = [os.path.isfile("%s/%s.h5" % (save_dir, ln)) for ln in layer_names]
    if all(files_exist):
        print('\tAll tuning curves already computed')
        return
    else:
        print('\tAt least one tuning curve needs to be computed, proceeding')

    keys = ['labels']
    label_features = get_features(
        feature_path,
        group_name='sineff_20190507_features_tnn',
        keys=keys,
        out_keys=keys,
        show_keys=False,
        verbose=True,
    )
    labels = label_features['labels']
    param_names = [
        "angles",
        "sfs",
        "phases",
        "colors"
    ]

    # extract labels
    for layer_name in layer_names:
        print("Processing %s" % layer_name)

        keys = [layer_name]
        layer_features = get_features(
            feature_path,
            group_name='sineff_20190507_features_tnn',
            keys=keys,
            out_keys=keys,
            show_keys=False,
            verbose=True,
        )
        layer_feat = layer_features[layer_name]
        flat_layer = layer_feat.reshape((layer_feat.shape[0], -1)) + 1 # special to deal with elu -- set minimum to 1

        fpath = "%s/%s.h5" % (save_dir, layer_name)
        if os.path.isfile(fpath):
            print("Tuning curve dictionary already exists at %s" % fpath)
        else:
            print("Saving tuning curves to %s" % fpath)
            tuning_curves = compute_tuning_curves(labels,
                                                  flat_layer.T,
                                                  passing_indices=None,
                                                  agg_func=np.mean,
                                                  verbose=True,
                                                  param_names=param_names)
            dd.io.save(fpath, tuning_curves)

if __name__ == "__main__":
    compute_all_tc()
