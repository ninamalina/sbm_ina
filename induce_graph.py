from functools import partial

import graph_tool as gt
import numpy as np
from scipy.spatial.distance import pdist, squareform

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
