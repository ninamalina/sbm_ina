import numpy as np
from graph_tool.inference import minimize_blockmodel_dl
from sklearn.metrics import silhouette_score

from induce_graph import induce_graph, cutoff


class SilhoutteSelection:
    """Wrap clustering methods that can't determine the optimal number of
    clusters by themselves.

    Try the method with several parameter for `n_clusters` and return the
    labelling with the best silhouette score.

    """
    def __init__(self, method, n_clusters=10, *args, **kwargs):
        self.method = method
        self.n_clusters = n_clusters
        self.args = args
        self.kwargs = kwargs

    def fit_predict(self, data):
        scored_labels = []
        # Try different values for `n_clusters`
        for n_clusters in range(2, self.n_clusters):
            method = self.method(n_clusters, *self.args, **self.kwargs)
            labels = method.fit_predict(data)
            score = silhouette_score(data, labels)
            scored_labels.append((score, labels))
        # Find the labelling with the best score and return that
        return np.array(max(scored_labels)[1])


class ClusteringWithCutoff:
    """Perform clustering using the stochastic block model.

    Since we're working with full graphs, different threshold rates are checked
    and the one that produces the best silhouette score is selected.

    """

    def __init__(self, metric='manhattan_inv', threshold_interval=10):
        self.metric = metric
        self.cutoff_interval = threshold_interval

    def fit_predict(self, data):
        graph = induce_graph(data, distance=self.metric)

        result_blocks = []

        weights = graph.edge_properties['weights'].get_array()
        for threshold in np.linspace(0, weights.max(), self.cutoff_interval):
            working_graph = cutoff(graph, threshold, inplace=True)
            # Apply the sbm to the pruned graph
            blocks = minimize_blockmodel_dl(working_graph)
            blocks = blocks.get_blocks().get_array()

            # Silhouette doesn't work if there's only one cluster label
            if len(np.unique(blocks)) > 1:
                cutoff_score = silhouette_score(data, blocks)
                result_blocks.append((cutoff_score, blocks))

        return np.array(max(result_blocks)[1])
