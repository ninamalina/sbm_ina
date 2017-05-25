import numpy as np
from graph_tool.inference import minimize_blockmodel_dl
from sklearn.metrics import silhouette_score

from induce_graph import induce_graph, cutoff


class ClusteringWithCutoff:
    """Perform clustering using the stochastic block model.

    Since we're working with full graphs, different threshold rates are checked
    and the one that produces the best silhouette score is selected.

    """

    @staticmethod
    def fit_predict(data, check_interval=10):
        graph = induce_graph(data)

        result_blocks = []

        weights = graph.edge_properties['weights'].get_array()
        for threshold in np.linspace(0, weights.max(), check_interval):
            working_graph = cutoff(graph, threshold, inplace=True)
            # Apply the sbm to the pruned graph
            blocks = minimize_blockmodel_dl(working_graph)
            blocks = blocks.get_blocks().get_array()

            # Silhouette doesn't work if there's only one cluster label
            if len(np.unique(blocks)) > 1:
                cutoff_score = silhouette_score(data, blocks)
                result_blocks.append((cutoff_score, blocks))

        return np.array(max(result_blocks)[1])
