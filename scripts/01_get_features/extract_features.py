"""
Model evaluation and feature extraction script
"""

from __future__ import division, print_function, absolute_import

import argparse
import os
import pdb

import h5py
import numpy as np
from configs import get_load_config, get_validation_config

from tfutils import base

## Resolve project path
PROJ_PATH = os.environ.get("P250_PROJ_PATH")
if PROJ_PATH is None:
    PROJ_PATH = "/mnt/fs6/eshedm"

def get_params(load_config, data_config):
    params = {}
    params["model_params"] = load_config["model_params"]
    params["save_params"] = {"do_save": False}
    params["load_params"] = {
        "host": "localhost",
        "port": FLAGS.port,
        "dbname": load_config["dbname"],
        "collname": load_config["collname"],
        "exp_id": load_config["exp_id"],
        "cache_dir": None,
        "from_ckpt": load_config["from_ckpt"],
    }
    params["validation_params"] = data_config
    return params


def make_params_from_cfg(cfg, load_config):
    params = {}

    params["targets"] = cfg["targets"]
    params["num_steps"] = cfg["num_steps"]
    params["agg_func"] = cfg.get("agg_func")
    params["online_agg_func"] = cfg.get("online_agg_func")

    params["data_params"] = {
        "func": cfg["func"],
        "file_pattern": cfg["file_pattern"],
        "is_train": False,
        "q_cap": cfg["batch_size"],
        "batch_size": cfg["batch_size"],
    }

    return params


def get_validation_params(provider_names, load_config, prep_type="resnet"):
    validation_params = {}
    for name in provider_names:
        val_cfg = get_validation_config(name, load_config, prep_type=prep_type)
        if val_cfg is None:
            raise Exception("Validation parameters for %s not found" % name)
        params = make_params_from_cfg(val_cfg, load_config)

        validation_params[name] = params
    return validation_params


def save_results(res, save_path, to_skip=["imagenet_performance"]):
    something_was_written = False
    with h5py.File(save_path, "w") as f:
        for val in res[0].keys():
            print("Val: %s" % val)
            # dont save results from imagenet validation
            if val in to_skip:
                print("Skipping saving of %s" % val)
                continue
            else:
                something_was_written = True
            g = f.create_group(val)

            batched_targets = [
                targ
                for targ in res[0][val]["result"][0].keys()
                if "singular" not in targ
            ]
            singular_targets = [
                targ for targ in res[0][val]["result"][0].keys() if "singular" in targ
            ]

            for targ in singular_targets:
                print("Singular target: %s" % targ)
                g[targ] = res[0][val]["result"][0][targ]

            for targ in batched_targets:
                print("Batched target: %s" % targ)
                dics = res[0][val]["result"]

                # initialize array F with the first batch
                print("\tWriting batch 1 of %d" % len(dics))
                F = dics[0][targ]

                # add all of the other batches
                for i in range(1, len(dics)):
                    print("\tWriting batch %d of %d" % (i + 1, len(dics)))
                    F = np.append(F, dics[i][targ], axis=0)

                g[targ] = F

    if something_was_written:
        print("Finished writing to %s" % save_path)


def main():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

    load_config = get_load_config(
        FLAGS.exp_id, FLAGS.step, FLAGS.config_name, seed=0, port=FLAGS.port
    )

    validations = [FLAGS.config_name]
    data_config = get_validation_params(validations, load_config, FLAGS.prep_type)
    params = get_params(load_config, data_config)

    res = base.test_from_params(**params)

    save_base = PROJ_PATH + "/extracted_features/"
    save_path = (
        save_base
        + FLAGS.exp_id
        + "_step_" + str(FLAGS.step)
        + "_" + FLAGS.config_name
        + FLAGS.identifier
        + ".h5"
    )
    save_results(res, save_path)


if __name__ == "__main__":
    # parse input arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu",
                        type=str,
                        default="9",
                        help="which GPU number to use")

    parser.add_argument("--exp_id",
                        type=str,
                        help="experiment id")

    parser.add_argument("--prep_type",
                        type=str,
                        default="resnet",
                        help="which preprocessing to use")

    parser.add_argument("--step",
                        type=int,
                        help="training step")

    parser.add_argument("--port",
                        type=int,
                        default=26116,
                        help="mongodb port")

    parser.add_argument(
        "--config_name",
        type=str,
        default="imagenet_performance",
        help="name of config to use",
    )

    parser.add_argument(
        "--identifier",
        type=str,
        default="",
        help="extra identifier for save path",
    )

    FLAGS, _ = parser.parse_known_args()
    main()
