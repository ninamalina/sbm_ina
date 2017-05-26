from functools import partial

import fire
import graph_tool as gt
import graph_tool.topology as tp
import numpy as np
from graph_tool.inference import minimize_blockmodel_dl
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import normalized_mutual_info_score, silhouette_score

DISTANCES = {
    'chebyshev': partial(pdist, metric='chebyshev'),
    'euclidean': partial(pdist, metric='euclidean'),
    'manhattan': partial(pdist, metric='cityblock'),
}


def induce_graph(data, distance='manhattan_inv', invexp_factor=1):
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
    if inv and inv[0] == 'inv':
        distances = invexp_factor * np.exp(-invexp_factor * distances)
    # Convert the distances from the upper triangular form
    distances = squareform(distances)

    # Finally, create a full graph with distances
    for i in range(distances.shape[0]):
        for j in range(i + 1, distances.shape[0]):
            graph.add_edge(i, j)
            weights[i, j] = distances[i][j]

    return graph


def cutoff(graph, threshold, inplace=False):
    """Remove all edges under a certain threshold in a graph."""
    assert threshold >= 0, 'The threshold must be greater than 0.'

    if not inplace:
        graph = gt.Graph(graph)

    weights = graph.edge_properties['weights']
    edges_to_remove = [e for e in graph.edges() if weights[e] <= threshold]

    for edge in edges_to_remove:
        graph.remove_edge(edge)

    return graph


def calculate_cutoff_components(graph, n_split_threshold=10):
    """Apply thresholding to a graph gradually and return the component labels
    for each cutoff."""
    working_graph = gt.Graph(graph)
    weights = graph.edge_properties['weights'].get_array()

    max_weight = weights.max()

    results = []
    for th in np.linspace(0, max_weight, num=n_split_threshold):
        cutoff(working_graph, th, inplace=True)
        results.append(tp.label_components(working_graph))

    return results


def sbm_clustering_nmi_silhouette(data, y, threshold, n_tries=10):
    """Apply the SBM model to graphs with different thresholds.

    Due to sbm randomness, repeat the process `n` times and return the mean
    silhouette score.

    """
    graph = induce_graph(data)
    graph = cutoff(graph, threshold, inplace=True)

    nmi_scores, silhouette_scores = [], []
    for _ in range(n_tries):
        # Apply the sbm to the pruned graph
        blocks = minimize_blockmodel_dl(graph)
        blocks = blocks.get_blocks().get_array()
        # Compute the nmi and silhouette score for evaluation
        nmi_scores.append(normalized_mutual_info_score(y, blocks))
        silhouette_scores.append(
            silhouette_score(data, blocks) if len(np.unique(blocks)) > 1 else 0
        )

    return np.mean(nmi_scores), np.mean(silhouette_scores)


if __name__ == '__main__':
    fire.Fire()
