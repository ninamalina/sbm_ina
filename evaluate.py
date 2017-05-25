from collections import defaultdict
from pprint import pprint
import numpy as np

import fire
from Orange.data import Table
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import normalized_mutual_info_score, silhouette_score

from cutoff_clustering import ClusteringWithCutoff


class Data:
    def __init__(self, f_name):
        file = open(f_name)
        X = []
        Y = []
        for line in file:
            line = line.strip().split(",")
            X.append(line[:-1])
            Y.append(line[-1])
        self.X = np.array(X).astype(float)
        self.Y = np.array(Y).astype(int)


DATASETS = {
    'iris': Table('iris'),
    'ecoli': Table('ecoli'),
    'glass': Table('glass'),
    'wine': Table('wine'),
    'zoo': Table('zoo'),
    'circular': Table('datasets/circular'),
    'ina': Table('datasets/ina'),
    'two_moons': Table('datasets/two_moons'),
    'movements': Data('datasets/movement_libras.data'),
}

CLUSTERING_METHODS = {
    'k-means': KMeans(n_clusters=3, init='random', n_jobs=-1),
    'Hierarchical clustering': AgglomerativeClustering(n_clusters=3),
    'Connected components': ClusteringWithCutoff(),
}


def run():
    nmi_scores = defaultdict(dict)
    silhouette_scores = defaultdict(dict)

    for dataset_name, dataset in DATASETS.items():
        for method_name, method in CLUSTERING_METHODS.items():
            labels = method.fit_predict(dataset.X)

            nmi_scores[dataset_name][method_name] = \
                normalized_mutual_info_score(dataset.Y, labels)
            silhouette_scores[dataset_name][method_name] = \
                silhouette_score(dataset.X, labels)

    pprint(nmi_scores)
    pprint(silhouette_scores)


if __name__ == '__main__':
    fire.Fire()
    # for ds in DATASETS:
    #     print(ds, "&", DATASETS[ds].X.shape[0], "&", DATASETS[ds].X.shape[1],
    #           "&", len(set(DATASETS[ds].Y)))
