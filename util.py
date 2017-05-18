import matplotlib as mpl
mpl.use("TkAgg")
from scipy.spatial.distance import cdist
import networkx as nx
import numpy as np


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

