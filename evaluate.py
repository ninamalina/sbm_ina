from collections import defaultdict

import fire
import numpy as np
from Orange.data import Table
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import normalized_mutual_info_score, silhouette_score, \
    adjusted_rand_score

from clustering import ClusteringWithCutoff, SilhoutteSelection, WSBM


class Data:
    def __init__(self, f_name):
        with open(f_name) as file:
            X = []
            Y = []
            for line in file:
                line = line.strip().split(",")
                X.append(line[:-1])
                Y.append(line[-1])
            self.X = np.array(X).astype(float)
            self.Y = np.array(Y).astype(int)


DATASETS = {
    # 'iris': Table('iris'),
    'ecoli': Table('ecoli'),
    # 'glass': Table('glass'),
    # 'wine': Table('wine'),
    # 'zoo': Table('zoo'),
    # 'circular': Table('datasets/circular'),
    # 'ina': Table('datasets/ina'),
    # 'two_moons': Table('datasets/two_moons'),
    # 'movements': Data('datasets/movement_libras.data'),
}

CLUSTERING_METHODS = {
    # Traditional clustering approaches
    'k-means': SilhoutteSelection(KMeans, n_jobs=-1),
    'Hierarchical clustering': SilhoutteSelection(AgglomerativeClustering),

    # Complex networks approaches
    # TODO

    # Stochastic block model with thresholding
    'SBM (Manhattan)': ClusteringWithCutoff('manhattan_inv'),
    'SBM (Euclidean)': ClusteringWithCutoff('euclidean_inv'),
    'SBM (Chebyshev)': ClusteringWithCutoff('chebyshev_inv'),

    # Weighted stochastic block models
    'WSBM (Manhattan)': WSBM('manhattan_inv'),
    'WSBM (Euclidean)': WSBM('euclidean_inv'),
    'WSBM (Chebyshev)': WSBM('chebyshev_inv'),
}


def run():
    nmi_scores = defaultdict(dict)
    silhouette_scores = defaultdict(dict)
    ari_scores = defaultdict(dict)
    n_clusters = defaultdict(dict)

    for dataset_name, dataset in DATASETS.items():
        for method_name, method in CLUSTERING_METHODS.items():
            labels = method.fit_predict(dataset.X)

            nmi_scores[dataset_name][method_name] = \
                normalized_mutual_info_score(dataset.Y, labels)
            silhouette_scores[dataset_name][method_name] = \
                silhouette_score(dataset.X, labels)
            ari_scores[dataset_name][method_name] = \
                adjusted_rand_score(dataset.Y, labels)
            n_clusters[dataset_name][method_name] = np.unique(labels).shape[0]
            print('Finished evaluating `%s` with `%s`.' % (
                dataset_name, method_name))

    # Print header
    print(' & '.join([
        'Dataset', 'Clustering method', 'Silhouette score', 'NMI score',
        'ARI', '$n$ clusters',
    ]), r'\\ \hline')
    # TODO Change this when we agree on a better format for the results table
    for dataset_name in DATASETS:
        for method_name in CLUSTERING_METHODS:
            print(' & '.join([
                dataset_name, method_name,
                '%.4f' % silhouette_scores[dataset_name][method_name],
                '%.4f' % nmi_scores[dataset_name][method_name],
                '%.4f' % ari_scores[dataset_name][method_name],
                '%d' % n_clusters[dataset_name][method_name],
            ]), r'\\')
        print(r'\hline')


def get_labels():
    for dataset_name, dataset in DATASETS.items():
        for method_name, method in CLUSTERING_METHODS.items():
            labels = method.fit_predict(dataset.X)
            np.savetxt('_labellings/%s_%s.csv' %
                       (dataset_name, method_name), labels.astype(np.int16))


if __name__ == '__main__':
    fire.Fire()
    # for ds in DATASETS:
    #     print(ds, "&", DATASETS[ds].X.shape[0], "&", DATASETS[ds].X.shape[1],
    #           "&", len(set(DATASETS[ds].Y)))
