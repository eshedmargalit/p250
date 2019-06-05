"""
Gets pre-formatted configurations for model evaluation
"""
from __future__ import absolute_import, division, print_function

import re
from collections import defaultdict

import numpy as np
import tensorflow as tf
from tfutils import utils  # pylint: disable=import-error

from p250.models import model_functions
from p250.data_providers.imagenet_data import ImageNet
from p250.data_providers.sineff_data import SineFF
from p250.utils.utils import loss_and_in_top_k


def get_load_config(exp_id, step, valname, seed=0, port=26016):
    """
    Returns a configuration dictionary for the experiment specified

    Inputs:
        exp_id (str): the key from the experiments dictionary to get the configuration for
        step (int): global step of model checkpoint to use
        valname (str): the name of the validation configuration to use, e.g., 'sineff_20190507'
        seed (int): for model functions that take a seed argument
        port (int): mongodb port to load results from
    """

    # define vgg keys here programatically
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

    vgg19_keys = {}
    for layer in layers:
        base_tensor_name = (
            "model_0/validation/"
            + valname
            + "/model_0/__var_copy_0/__GPU0__/vgg_19/"
            + layer.split("_")[0]
            + "/"
            + layer
        )

        # vgg19_keys[layer] = base_tensor_name + '/Conv2D:0'
        vgg19_keys[layer + "_relu"] = base_tensor_name + "/Relu:0"

    experiment_configs = {
        "alexnet-baseline-0-allrelu": {
            "dbname": "tpu-scl",
            "collname": "noscl",
            "exp_id": "alexnet-baseline-0",
            "from_ckpt": "/mnt/fs6/eshedm/gs_models/alexnet-baseline-0/model.ckpt-"
            + str(step),
            "model_params": {
                "func": model_functions.alexnet_wrapper,
                "seed": seed,
                "cfg_final": None,
                "return_params": True,  # true for tfutils, false for TPU
            },
            "to_extract": {
                "conv1_relu": "model_0/validation/"
                + valname
                + "/model_0/__var_copy_0/__GPU0__/relu:0",
                "conv2_relu": "model_0/validation/"
                + valname
                + "/model_0/__var_copy_0/__GPU0__/relu_1:0",
                "conv3_relu": "model_0/validation/"
                + valname
                + "/model_0/__var_copy_0/__GPU0__/relu_2:0",
                "conv4_relu": "model_0/validation/"
                + valname
                + "/model_0/__var_copy_0/__GPU0__/relu_3:0",
                "conv5_relu": "model_0/validation/"
                + valname
                + "/model_0/__var_copy_0/__GPU0__/relu_4:0",
            },
            "step": step,
            "port": port,
        },
        "vgg19_slim": {
            "dbname": "null",
            "collname": "null",
            "exp_id": "null",
            "from_ckpt": "/mnt/fs6/tf-slim/checkpoints/vgg_19.ckpt",
            "model_params": {
                "func": model_functions.vgg_19_wrapper,
                "num_classes": 1000,
                "cfg_final": None,
                "return_params": True,  # true for tfutils, false for TPU
            },
            "to_extract": vgg19_keys,
            "step": None,
            "port": port,
        },
        "tnn": {  # tnn model fn requires that all conv layer outputs used
            "dbname": "null",
            "collname": "null",
            "exp_id": "null",
            "from_ckpt": "/mnt/fs6/eshedm/checkpoints/model.ckpt-1940300",
            "model_params": {
                "func": model_functions.tnn_model_func,
                "cfg_final": None,
                "json_fpath": "/mnt/fs6/eshedm/checkpoints/ff_128_neuralfit.json",
                "batch_size": 128,
            },
            "step": None,
            "port": port,
        },
    }
    config = experiment_configs.get(exp_id)
    return config


def get_extraction_target(inputs, outputs, to_extract, to_print="", **loss_params):
    if not to_print == "":
        names = [
            [x.name for x in op.values()]
            for op in tf.get_default_graph().get_operations()
        ]
        names = [y for x in names for y in x]

        regex = re.compile(r"__GPU__\d/")
        _targets = defaultdict(list)

        matching_names = list()
        for name in names:
            name_without_gpu_prefix = regex.sub("", name)
            if to_print in name_without_gpu_prefix:
                matching_names.append(name)
        # ipdb.set_trace()

    targets = {
        k: tf.get_default_graph().get_tensor_by_name(v) for k, v in to_extract.items()
    }
    targets["labels"] = inputs["labels"]
    targets["images"] = inputs["images"]
    return targets


def tnn_target_func(inputs, outputs):
    outputs["labels"] = inputs["labels"]
    outputs["images"] = inputs["images"]
    return outputs


def get_validation_config(name, load_config, prep_type="resnet"):
    """
    Returns a configuration dictionary for the validation data specified
    """
    # point to raw tfrecord data files
    imagenet_dir = "/mnt/fs1/Dataset/TFRecord_Imagenet_standard/image_label_full"
    sineff_20190507_dir = "/mnt/fs6/eshedm/tfrecords/sineff_20190507"

    targdict = {
        "func": get_extraction_target,
        "to_extract": load_config.get("to_extract"),
        "to_print": "",
    }
    tnn_targdict = {"func": tnn_target_func}

    val_params = {
        "imagenet_performance": {
            "func": ImageNet(imagenet_dir, prep_type, drop_remainder=True).dataset_func,
            "file_pattern": "validation-*",
            "batch_size": 256,
            "crop_size": 224,
            # 'num_steps': int(ImageNet.VAL_LEN // 256),
            "num_steps": 20,
            "targets": {"func": loss_and_in_top_k, "target": "labels"},
            "agg_func": lambda x: {k: np.mean(v) for k, v in x.items()},
            "online_agg_func": utils.online_agg,
        },
        "imagenet_features_tnn": {
            "func": ImageNet(imagenet_dir, prep_type, drop_remainder=True).dataset_func,
            "file_pattern": "validation-*",
            "batch_size": 256,
            "crop_size": 224,
            "num_steps": 10,
            "online_agg_func": utils.append_and_return,
            "agg_func": utils.identity_func,
            "targets": tnn_targdict,
        },
        "imagenet_features": {
            "func": ImageNet(imagenet_dir, prep_type).dataset_func,
            "file_pattern": "validation-*",
            "batch_size": 256,
            "crop_size": 224,
            # 'num_steps': int(ImageNet.VAL_LEN // 256),
            "num_steps": 20,
            "online_agg_func": utils.append_and_return,
            "agg_func": utils.identity_func,
            "targets": targdict,
        },
        "sineff_20190507_features": {
            "func": SineFF(sineff_20190507_dir, prep_type, n_label_cols=4).dataset_func,
            "file_pattern": "sine_ff_*",
            "batch_size": 128,
            "crop_size": 224,
            "num_steps": 5,
            "online_agg_func": utils.append_and_return,
            "agg_func": utils.identity_func,
            "targets": targdict,
        },
        "sineff_20190507_features_tnn": {
            "func": SineFF(
                sineff_20190507_dir, prep_type, n_label_cols=4, drop_remainder=True
            ).dataset_func,
            "file_pattern": "sine_ff_*",
            "batch_size": 128,
            "crop_size": 224,
            "num_steps": 5,
            "online_agg_func": utils.append_and_return,
            "agg_func": utils.identity_func,
            "targets": tnn_targdict,
        },
    }
    params = val_params.get(name)
    return params
