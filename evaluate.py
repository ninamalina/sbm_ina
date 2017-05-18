from collections import defaultdict
from pprint import pprint

import fire
from Orange.data import Table
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import normalized_mutual_info_score, silhouette_score

DATASETS = {
    'iris': Table('iris'),
    # 'ecoli': Table('ecoli'),
}

CLUSTERING_METHODS = {
    'k-means': KMeans(n_clusters=3, init='random', n_jobs=-1),
    'Hierarchical clustering': AgglomerativeClustering(
        n_clusters=3, linkage='complete'
    ),
    'Spectral clustering': SpectralClustering(n_clusters=3, n_jobs=-1),
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
