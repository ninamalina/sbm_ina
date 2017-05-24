from functools import partial
from pprint import pprint

import fire
from sklearn import datasets

import graph_tool as gt
import graph_tool.topology as tp
from graph_tool.inference import minimize_blockmodel_dl
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import normalized_mutual_info_score

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


def test(threshold, n_tries=10):
    iris = datasets.load_iris()
    G = induce_graph(iris.data)
    G = cutoff(G, threshold)

    results = []
    for _ in range(n_tries):
        blocks = minimize_blockmodel_dl(G)
        blocks = blocks.get_blocks().get_array()

        results.append(normalized_mutual_info_score(iris.target, blocks))

    print(np.mean(results))


if __name__ == '__main__':
    fire.Fire()
