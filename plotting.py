from os.path import join, dirname

import matplotlib
import numpy as np
from Orange.data import Table

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from induce_graph import induce_graph, sbm_clustering_nmi_silhouette, DISTANCES

RESULTS_DIR = join(dirname(__file__), 'results')


def plot_threshold_sbm_components(dataset, name, split_into=10):
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
              % name.title())
    plt.xlabel('Distance threshold')
    plt.ylabel('Scores')
    plt.legend()
    plt.savefig('%s/threshold_clustering_%s.png' % (RESULTS_DIR, name))


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


if __name__ == '__main__':
    fire.Fire()
