import os
from os.path import join, dirname, exists
from subprocess import call

import fire
import matplotlib
import numpy as np
from Orange.data import Table

# Use Agg backend so we can generate images inside the docker image
from matplotlib.cbook import mkdirs
from sklearn.metrics import silhouette_score, normalized_mutual_info_score

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from induce_graph import induce_graph, sbm_clustering_nmi_silhouette, \
    DISTANCES, save_to_file

RESULTS_DIR = join(dirname(__file__), 'results')


def plot_threshold_sbm_components(dataset, split_into=10):
    """This plots the difference between the silhouette score and NMI on a
    dataset when clustering with the threshold SBM."""
    dataset_ = Table(dataset)
    data, y = dataset_.X, dataset_.Y

    graph = induce_graph(data)
    weights = graph.edge_properties['weights'].get_array()

    nmi_scores, silhouettes_scores = [], []
    thresholds = np.linspace(0, weights.max(), split_into)
    for threshold in thresholds:
        nmi, silhouette = sbm_clustering_nmi_silhouette(data, y, threshold)
        nmi_scores.append(nmi)
        silhouettes_scores.append(silhouette)

    plt.plot(thresholds, nmi_scores, label='NMI')
    plt.plot(thresholds, silhouettes_scores, label='Silhouette')
    plt.title('Clustering with SBM with thresholding on %s'
              % dataset.title())
    plt.xlabel('Distance threshold')
    plt.ylabel('Scores')
    plt.legend()
    plt.savefig('%s/threshold_clustering_%s.png' % (RESULTS_DIR, dataset))


def plot_threshold_sbm_distance_metrics(dataset, split_into=20):
    dataset_ = Table(dataset)
    data, y = dataset_.X, dataset_.Y

    for metric in DISTANCES:
        graph = induce_graph(data, distance='%s_invexp' % metric)
        weights = graph.edge_properties['weights'].get_array()

        silhouette_scores = []
        thresholds = np.linspace(0, weights.max(), split_into)
        for threshold in thresholds:
            _, silhouette = sbm_clustering_nmi_silhouette(data, y, threshold)
            silhouette_scores.append(silhouette)

        plt.plot(thresholds, silhouette_scores, label=metric)
        plt.title('Clustering with SBM after thresholding on %s' %
                  dataset.title())
        plt.xlabel('Distance threshold')
        plt.ylabel('Silhouette score')
    plt.legend()
    plt.savefig('%s/threshold_clustering_metrics_%s.png' % (
        RESULTS_DIR, dataset))


def plot_blocks_wsbm_evaluation_metrics(dataset, test_range=tuple(range(2, 15))):
    """This plots the difference between the silhouette score and NMI on a
    dataset when clustering with the WSBM."""
    dataset_ = Table(dataset)
    data, y = dataset_.X, dataset_.Y

    graph = induce_graph(data)
    fname = save_to_file(graph, force_integer=True)

    output_directory = '_labellings'
    if not exists(output_directory):
        mkdirs(output_directory)

    cwd = os.getcwd()
    os.chdir('YWWTools/target')

    nmi_scores, silhouettes_scores = [], []
    for num_blocks in test_range:
        result_fname = '../../%s_blocks_%d' % (fname, num_blocks)
        call(' '.join(
            ['java',
             '-cp YWWTools.jar:deps.jar yang.weiwei.Tools',
             '--tool wsbm',
             '--nodes %d' % graph.num_vertices(),
             '--blocks %d' % num_blocks,
             '--graph ../../%s' % fname,
             '--output %s' % result_fname,
             '--no-verbose',
             ]
        ), shell=True)

        blocks = np.genfromtxt(result_fname)
        os.remove(result_fname)
        silhouettes_scores.append(silhouette_score(data, blocks))
        nmi_scores.append(normalized_mutual_info_score(y, blocks))

    os.chdir(cwd)
    os.remove(fname)

    plt.plot(test_range, nmi_scores, label='NMI')
    plt.plot(test_range, silhouettes_scores, label='Silhouette')
    plt.title('Clustering with WSBM on %s'
              % dataset.title())
    plt.xlabel('Blocks')
    plt.ylabel('Scores')
    plt.legend()
    plt.savefig('%s/wsbm_blocks_%s.png' % (RESULTS_DIR, dataset))


if __name__ == '__main__':
    fire.Fire()
