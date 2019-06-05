"""
Utilities for dealing with file I/O
"""
from __future__ import print_function, division, absolute_import

import os
import pprint
import h5py


def get_features(
    validation_path,
    group_name="imagenet_features",
    keys=("images"),
    out_keys=None,
    show_keys=False,
    verbose=False,
):
    """
    Returns features from HDF5 DataSet

    Inputs
        validation_path (str): where to find the HDF5 dataset
        group_name (str): the group name used for the particular validation
        keys (list of strs): which keys to extract from the group
        out_keys (list of strs): keys for the output dict
    """
    assert os.path.isfile(validation_path), "%s is not a file" % (validation_path)
    assert len(keys) == len(
        out_keys
    ), "Number of keys does not match number of output keys"
    if out_keys is None:
        out_keys = keys

    out = {}
    with h5py.File(validation_path, "r") as open_file:
        if show_keys:
            keys_to_print = open_file[group_name].keys()
            print("Keys in dataset:")
            pprint.pprint(keys_to_print)
        for in_key, out_key in zip(keys, out_keys):
            out[out_key] = open_file[group_name][in_key][:]
            if verbose:
                print("Extracted %s:" % out_key, out[out_key].shape)

    return out
