import matplotlib as mpl

mpl.use("TkAgg")
from scipy.spatial.distance import cdist
import networkx as nx
import numpy as np
import graph_tool as gt


def nx2gt(nx_graph):
    """Convert a networkx graph into a graph_tool graph object."""
    gt_graph = gt.Graph(directed=nx_graph.is_directed())

    # Add weights property map to new graph
    weights = gt_graph.new_edge_property('double')
    gt_graph.edge_properties['weights'] = weights

    nx_weights = nx.get_edge_attributes(nx_graph, 'weight')
    for edge in nx_graph.edges():
        gt_graph.add_edge(*edge)
        weights[edge] = nx_weights[edge]

    return gt_graph


def build_graph(X, distance, alpha=1):
    D = np.array([])

    if (distance == "chebyshev"):
        D = cdist(X, X, 'chebyshev')
        D = alpha * np.exp(-alpha * D)
    elif (distance == "manhattan"):
        D = cdist(X, X, 'cityblock')
        D = alpha * np.exp(-alpha * D)
    # TODO: other distances - euclidean, tanimoto...

    G = nx.Graph()

    for i in range(D.shape[0]):
        for j in range(i + 1, D.shape[0]):
            G.add_edge(i, j, weight=D[i][j])

    return G
