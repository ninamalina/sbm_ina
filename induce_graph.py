from functools import partial
from pprint import pprint

import matplotlib
from Orange.data import Table

matplotlib.use('Agg')


import fire
import graph_tool as gt
import graph_tool.topology as tp
import numpy as np
from graph_tool.inference import minimize_blockmodel_dl
from scipy.spatial.distance import pdist, squareform
from sklearn import datasets
from sklearn.metrics import normalized_mutual_info_score, silhouette_score

import matplotlib.pyplot as plt

DISTANCES = {
    'chebyshev': partial(pdist, metric='chebyshev'),
    'euclidean': partial(pdist, metric='euclidean'),
    'manhattan': partial(pdist, metric='cityblock'),
}


def induce_graph(data, distance='manhattan_invexp', invexp_factor=1):
    """Induce a graph from the dataset with distances as weights."""
    # Prepare an empty undirected graph with weights
    graph = gt.Graph(directed=False)
    weights = graph.new_edge_property('double')
    graph.edge_properties['weights'] = weights

    # Split the distance string to detecet
    distance, *inv = distance.split('_')

    # Compute the pairwise distances
    distances = DISTANCES[distance](data)
    # If we want to take the inverse exponent, apply that as well
    if inv and inv[0] == 'invexp':
        distances = invexp_factor * np.exp(-invexp_factor * distances)
    # Convert the distances from the upper triangular form
    distances = squareform(distances)

    # Finally, create a full graph with distances
    for i in range(distances.shape[0]):
        for j in range(i + 1, distances.shape[0]):
            graph.add_edge(i, j)
            weights[i, j] = distances[i][j]

    return graph


def cutoff(graph, threshold):
    assert threshold >= 0, 'The threshold must be greater than 0.'

    graph = gt.Graph(graph)

    weights = graph.edge_properties['weights']
    edges_to_remove = [e for e in graph.edges() if weights[e] <= threshold]

    for edge in edges_to_remove:
        graph.remove_edge(edge)

    return graph


def calculate_cutoff_components(graph, n_tries=10):
    working_graph = gt.Graph(graph)
    weights = graph.edge_properties['weights'].get_array()

    max_weight = weights.max()

    results = []
    for th in np.linspace(0, max_weight, num=n_tries):
        cutoff(working_graph, th)
        results.append(tp.label_components(working_graph))

    pprint(results)


class ClusteringWithCutoff:
    def fit_predict(self, data, check_interval=5):
        graph = induce_graph(data)

        result_blocks = []

        weights = graph.edge_properties['weights'].get_array()
        for threshold in np.linspace(0, weights.max(), check_interval):
            working_graph = cutoff(graph, threshold)
            blocks = minimize_blockmodel_dl(working_graph)
            blocks = blocks.get_blocks().get_array()

            # Silhouette doesn't work if there's only one cluster label
            if len(np.unique(blocks)) > 1:
                cutoff_score = silhouette_score(data, blocks)
                result_blocks.append((cutoff_score, blocks))

        return np.array(max(result_blocks)[1])


def test2():
    predictor = ClusteringWithCutoff()
    iris = datasets.load_iris()
    labels = predictor.fit_predict(iris.data)
    print(labels)


def clustering_nmi_silhouette(data, y, threshold, n_tries=10):
    graph = induce_graph(data)
    graph = cutoff(graph, threshold)

    nmi_scores, silhouette_scores = [], []
    for _ in range(n_tries):
        blocks = minimize_blockmodel_dl(graph)
        blocks = blocks.get_blocks().get_array()

        nmi_scores.append(normalized_mutual_info_score(y, blocks))
        silhouette_scores.append(silhouette_score(data, blocks)
                                 if len(np.unique(blocks)) > 1 else 0)

    return np.mean(nmi_scores), np.mean(silhouette_scores)


def plot_threshold_components(dataset, name, split_into=10):
    dataset_ = Table(dataset)
    data, y = dataset_.X, dataset_.Y

    graph = induce_graph(data)
    weights = graph.edge_properties['weights'].get_array()

    nmi_scores, silhouettes_scores = [], []
    thresholds = np.linspace(0, weights.max(), split_into)
    for threshold in thresholds:
        nmi, silhouette = clustering_nmi_silhouette(data, y, threshold)
        nmi_scores.append(nmi)
        silhouettes_scores.append(silhouette)

    plt.plot(thresholds, nmi_scores, label='NMI')
    plt.plot(thresholds, silhouettes_scores, label='Silhouette')
    plt.title('Clustering using connected components after thresholding on %s'
              % name.title())
    plt.xlabel('Distance threshold')
    plt.ylabel('Scores')
    plt.legend()
    plt.savefig('threshold_clustering_%s.png' % name)


def plot_data_distance_metrics(dataset, split_into=20):
    dataset_ = Table(dataset)
    data, y = dataset_.X, dataset_.Y

    for metric in DISTANCES:
        graph = induce_graph(data, distance='%s_invexp' % metric)
        weights = graph.edge_properties['weights'].get_array()

        silhouette_scores = []
        thresholds = np.linspace(0, weights.max(), split_into)
        for threshold in thresholds:
            _, silhouette = clustering_nmi_silhouette(data, y, threshold)
            silhouette_scores.append(silhouette)

        plt.plot(thresholds, silhouette_scores, label=metric)
        plt.title('Clustering using connected components after thresholding '
                  'on %s' % dataset.title())
        plt.xlabel('Distance threshold')
        plt.ylabel('Silhouette score')
    plt.legend()
    plt.savefig('threshold_clustering_metrics_%s.png' % dataset)


if __name__ == '__main__':
    fire.Fire()
