from sklearn import datasets
from util import build_graph

iris = datasets.load_iris()
G = build_graph(iris.data, "manhattan")
print(len(list(G.nodes())))
print(len(list(G.edges())))