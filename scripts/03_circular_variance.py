"""
Computes circular variance for units in all Alexnet conv layers.
Here, we remove units who have no mean response above a fixed
activity threshold. This is kind of like "not recording" neurons
that aren't visually responsibe and makes comparison to neural data
a bit more fair.
"""
from __future__ import division

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pdb
import os
import argparse
import pprint

import numpy as np
import scipy.stats as stats
import deepdish as dd

from p250.utils.tuning_curves import circular_variance

## Resolve project path
PROJ_PATH = os.environ.get("P250_PROJ_PATH")
if PROJ_PATH is None:
    PROJ_PATH = "/mnt/fs6/eshedm"

# globals
# define values from Ringach et al., 2002
RINGACH_VALS = np.array([
    3,
    21,
    26,
    17,
    25,
    20,
    23,
    21,
    30,
    32,
    33,
    36,
    14
]).astype(np.float)

TC_DIR = PROJ_PATH + '/tuning_curves_mean'

def get_circular_variances(fpath):
    """
    Extract tuning curves and compute circular variance for each unit
    """
    tuning_curves = dd.io.load(fpath)
    angle_tc = tuning_curves['angles']
    live_ind = np.where(angle_tc.sum(axis=1) > FLAGS.activity_threshold)[0]
    live_angle_tc = angle_tc[live_ind, :]

    circular_variances = np.array([circular_variance(tc) for tc in live_angle_tc])
    return circular_variances, live_angle_tc

def histogram_plot(normalized_counts_list, save_path, layer_names, axes):
    """
    Plot and save circular variances
    """
    for normalized_counts, ax, layer_name in zip(normalized_counts_list, axes, layer_names):
        bin_edges = np.linspace(0, 1, 14)
        bin_width = bin_edges[1] - bin_edges[0]
        bin_centers = bin_edges[:-1] + (bin_width/2)
        ax.bar(bin_centers,
               normalized_counts,
               facecolor='k', width=0.07)
        ax.set_xlabel('Circular Variance')
        ax.set_xlim((0, 1))
        ax.set_ylabel('Normalized Unit Count')
        ax.set_title(layer_name)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.savefig(save_path, dpi=150)

def example_tc_plot(cvs, angle_tcs, save_path, title=None):
    """
    Plot and save example tuning curves and their corresponding
    circular variances
    """
    fig = plt.figure(figsize=(10, 5))
    sort_ind = np.argsort(cvs)

    sorted_cv = cvs[sort_ind]
    sorted_tcs = angle_tcs[sort_ind]
    steps = np.floor(np.linspace(0, cvs.shape[0]-1, 10)).astype(int)
    representative_tcs = sorted_tcs[steps]
    
    angle_values = np.floor(np.linspace(0, 180, representative_tcs.shape[1] + 1)[:-1])
    angle_labels = angle_values

    fig = plt.figure(figsize=(20, 10))
    for i, tc in enumerate(representative_tcs):
        plt.subplot(2, 5, i + 1)
        plt.plot(angle_values, tc, color='k', linewidth=5)
        plt.title('CV = %.2f' % sorted_cv[steps[i]], fontsize=16)

    axes = fig.get_axes()

    min_val = np.min(representative_tcs.ravel())
    max_val = np.max(representative_tcs.ravel())

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticks(angle_values[::4])
        ax.set_xticklabels(angle_labels[::4])
        ax.set_ylim((min_val, max_val))
        ax.set_ylabel('Mean Response')
        ax.set_xlabel('Orientation (degrees)')

    if title is not None:
        plt.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

def plot_layer_corr(histogram_list, save_path, layer_names):
    """
    Plots correlation with Ringach data by layer
    """

    ringach_correlations = [stats.pearsonr(x, RINGACH_VALS)[0] for x in histogram_list]
    n_layers = len(layer_names)
    fig, ax = plt.subplots(figsize=(n_layers * 1.5, 4))

    ax.plot(np.arange(n_layers),
            ringach_correlations,
            c='k',
            linewidth=2)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xticks(np.arange(n_layers))
    ax.set_xticklabels(layer_names)
    ax.set_ylabel('Correlation with Ringach 2002 Data')

    # save
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

def main(model_name, load_dir, layers, layer_names, subplot_axes):
    """
    Main function, entry point for script
    """
    # make sure we have the files we need
    all_files = ["%s/%s.h5" % (load_dir, ln) for ln in layer_names]
    all_files_exist = all([os.path.isfile(f) for f in all_files])
    assert all_files_exist, "Go back and run computing_tuning_curves/compute_sineff_20190507_mean_tuning_curves.py first"

    save_dir = "%s/%s" % (SAVE_BASE, model_name)
    save_dir_tc = "%s/example_tuning_curves" % save_dir

    for sd in [save_dir, save_dir_tc]:
        if not os.path.isdir(sd):
            os.makedirs(sd)

    histogram_list = [] 
    count_save_path = PROJ_PATH + "/results_cache/cv_hist_counts/%s.npz" % model_name
    histogram_list_exists = os.path.isfile(count_save_path)
    for fpath, layer_name in zip(all_files, layer_names):
        print("Processing %s..." % layer_name)
        example_tc_save_path = "%s/%s_example_tcs.png" % (save_dir_tc, layer_name)

        plots_exist = os.path.isfile(example_tc_save_path)
        if plots_exist and histogram_list_exists and not FLAGS.override_skip: 
            print("All figures exist, skipping!")
            continue

        # get cvs and tuning curves, make plots
        cvs, angle_tcs = get_circular_variances(fpath)

        # make plots
        example_tc_plot(cvs, angle_tcs, example_tc_save_path, title=layer_name)

        # correlate with Ringach
        counts, _ = np.histogram(cvs, bins=np.linspace(0, 1, 14))
        normalized_counts = counts / counts.sum()
        histogram_list.append(normalized_counts)

    # plot correlation with ringach values across layers
    correlation_by_layer_save_path = "%s/correlation_by_layer.png" % save_dir
    histogram_save_path = "%s/histograms.png" % (save_dir)

    # catch corner case where no histograms are available
    if not histogram_list_exists or FLAGS.override_skip:
        np.savez(count_save_path, histogram_list=np.array(histogram_list))
    else:
        print("Loading histogram list from saved cache file")
        histogram_list = np.load(count_save_path)['histogram_list']

    plot_layer_corr(histogram_list, correlation_by_layer_save_path, layer_names)
    histogram_plot(histogram_list, histogram_save_path, layer_names, subplot_axes)
    return histogram_list

def plot_ringach_hist(ax):
    bin_edges = np.linspace(0, 1, 14)
    bin_width = bin_edges[1] - bin_edges[0]
    bin_centers = bin_edges[:-1] + (bin_width/2)
    ax.bar(bin_centers,
           RINGACH_VALS/RINGACH_VALS.sum(),
           facecolor='k',
           width=0.07)
    ax.set_xlabel('Circular Variance')
    ax.set_xlim((0, 1))
    ax.set_ylabel('Normalized Neuron Count')
    ax.set_title('Ringach 2002')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def make_plots_for_all_models():
    # alexnet
    alexnet_layers = [
        "conv1",
        "conv2",
        "conv3",
        "conv4",
        "conv5",
    ]
    layer_names = ["%s_relu" % x for x in alexnet_layers]
    model_name = 'alexnet-baseline-0-allrelu_step_115000'
    load_dir = '%s/20190507_sineff/%s' % (TC_DIR, model_name)

    fig, axes = plt.subplots(figsize=(15, 10), nrows=2, ncols=3)
    axes = axes.ravel()
    plot_ringach_hist(axes[0])

    print("Processing Alexnet...")
    alexnet_histograms = main(model_name, load_dir, alexnet_layers, layer_names, axes[1:])
    plt.close(fig)

    # VGG19
    vgg_layers = [
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
    layer_names = ["%s_relu" % x for x in vgg_layers]
    model_name = 'vgg19_slim_step_0'
    load_dir = '%s/20190507_sineff/%s_all' % (TC_DIR, model_name)
    fig, axes = plt.subplots(figsize=(20, 16), nrows=4, ncols=5)
    axes = axes.ravel()
    plot_ringach_hist(axes[0])

    print("Processing VGG...")
    vgg_histograms = main(model_name, load_dir, vgg_layers, layer_names, axes[1:])
    plt.close(fig)

    # TNN
    tnn_layers = ["conv%d" % x for x in range(1, 11)]
    layer_names =  tnn_layers
    model_name = 'tnn_step_0'
    load_dir = '%s/20190507_sineff/%s' % (TC_DIR, model_name)
    fig, axes = plt.subplots(figsize=(16, 12), nrows=3, ncols=4)
    axes = axes.ravel()
    plot_ringach_hist(axes[0])

    print("Processing TNN...")
    tnn_histograms = main(model_name, load_dir, tnn_layers, layer_names, axes[1:])
    plt.close(fig)

    print("Plotting final correlation figure...")
    plt.rcParams.update({'font.size': 16})
    alexnet_color = "#052F5F"
    vgg_color = "#06A77D"
    tnn_color = "#F1A208"
    
    # append layer identifiers
    alexnet_layers = ["alexnet_" + x for x in alexnet_layers]
    vgg_layers = ["vgg_" + x for x in vgg_layers]
    tnn_layers = ["tnn_" + x for x in tnn_layers]

    all_histograms = np.concatenate([alexnet_histograms, vgg_histograms, tnn_histograms])
    all_layers = np.concatenate([alexnet_layers, vgg_layers, tnn_layers])
    all_colors = np.concatenate([
        np.tile(alexnet_color, alexnet_histograms.shape[0]), 
        np.tile(vgg_color, vgg_histograms.shape[0]), 
        np.tile(tnn_color, tnn_histograms.shape[0]), 
    ])

    alexnet_alphas = np.linspace(0.2, 1, len(alexnet_layers))
    vgg_alphas = np.linspace(0.2, 1, len(vgg_layers))
    tnn_alphas = np.linspace(0.2, 1, len(tnn_layers))
    all_alphas = np.concatenate([alexnet_alphas, vgg_alphas, tnn_alphas])

    # sort by correlation to data
    ringach_correlations = np.array([stats.pearsonr(x, RINGACH_VALS)[0] for x in all_histograms])
    sort_ind = np.argsort(-ringach_correlations) # negate to sort in descending order
    sorted_correlations = ringach_correlations[sort_ind]
    sorted_layers = all_layers[sort_ind]
    sorted_colors = all_colors[sort_ind]
    sorted_alphas = all_alphas[sort_ind]

    fig, ax = plt.subplots(figsize=(25, 8))
    for idx, (corr, color, alpha) in enumerate(zip(sorted_correlations, sorted_colors, sorted_alphas)):
        ax.bar(idx, corr, facecolor=color, alpha=alpha)
    ax.set_xticks(np.arange(sorted_correlations.shape[0]))
    ax.set_xticklabels(sorted_layers, rotation=30)
    ax.set_ylabel('Correlation with Ringach Data')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim([-0.3, 1.0])

    save_path = "%s/all_model_all_layer_sorted_correlations.png" % SAVE_BASE
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    # parse input arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--override_skip",
                        action="store_true",
                        help="whether or not to override the automatic skipping of things already saved"
                        )

    parser.add_argument("--activity_threshold",
                        dest="activity_threshold",
                        type=float,
                        default=1.0)
    FLAGS, _ = parser.parse_known_args()
    SAVE_BASE = PROJ_PATH + "/figures/circular_variance_thr_%.2f" % FLAGS.activity_threshold
    if not os.path.isdir(SAVE_BASE):
        os.makedirs(SAVE_BASE)

    make_plots_for_all_models()

