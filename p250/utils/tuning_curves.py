"""
Tuning curve utility functions
"""

import numpy as np



def parse_labels(labels, param_names=None):
    """
    Takes n_units x n_params matrix and converts to
        dictionary indexed by parameter name
    """

    if param_names is None:
        param_names = ["angles", "sfs", "phases"]

    label_dict = {}
    unique = {}
    for i, param_name in enumerate(param_names):
        label_dict[param_name] = labels[:, i]
        unique[param_name] = np.unique(labels[:, i])

    return label_dict, unique


def compute_tuning_curves(labels,
                          features_flat,
                          passing_indices=None,
                          agg_func=np.max,
                          verbose=False,
                          param_names=None):
    """
    Computes tuning curve for each label type

    Inputs
        labels (n_images, n_params): image labels
        features_flat (n_units, n_images): image features
        passing_indices (n_units,): which image indices to keep
        agg_func: method used to collapse across tuning-curve
            orthogonal directions
        param_names (list): strings indexing columns of "labels"
    """

    assert labels.shape[0] == features_flat.shape[1], \
        "mismatched number of images and labels"

    label_dict, unique_labels = parse_labels(labels, param_names)
    if passing_indices is None:
        n_units = features_flat.shape[0]
        passing_indices = np.arange(n_units)

    tuning_curves = {}
    for param_name, unique_vals in unique_labels.items():
        if verbose:
            print("Computing tuning curves for %s" % param_name)

        labels = label_dict[param_name]

        # map each unique parameter value to the image indices it matches
        vals2index = {val: np.where(labels == val)[0] for val in unique_vals}
        n_vals = unique_vals.shape[0]

        collapsed = np.zeros((passing_indices.shape[0], n_vals))

        # for each unique parameter value, extract the corresponding features
        for val_num, val in enumerate(sorted(unique_vals)):
            indices = np.array(vals2index[val])

            # subselect features that match this particular value
            subslice = features_flat[passing_indices, :]
            subslice = subslice[:, indices]

            if len(subslice.shape) > 1:
                collapsed[:, val_num] = agg_func(subslice, axis=1)
            elif len(subslice.shape) == 1:
                collapsed[:, val_num] = subslice
            else:
                collapsed[:, val_num] = 0

        tuning_curves[param_name] = collapsed

    return tuning_curves

def circular_variance(tuning_curve, angles=None):
    """
    Computes circular variance as defined in Ringach et al., 2002

    Inputs:
        tuning_curve (1D array): responses to each orientation
        angles (1D array): angles, in radians, of tuning_curve
            If angles is None, assume it spans 0 to pi but stops short of pi

    Outputs:
        circular variance (scalar)
    """
    tuning_curve = np.squeeze(tuning_curve)
    assert len(tuning_curve.shape) == 1, "input cannot be squeezed to 1D"

    # set angles if not provided
    if angles is None:
        angles = np.linspace(0, np.pi, tuning_curve.shape[0] + 1)[:-1]

    # compute "R"
    numerator = np.sum(tuning_curve * np.exp(angles * 2 * 1j))
    denominator = np.sum(tuning_curve)
    R = numerator / denominator

    # compute circular variance
    CV = 1 - np.abs(R)

    return CV

