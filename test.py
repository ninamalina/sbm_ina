from sklearn import datasets
from util import build_graph, nx2gt

iris = datasets.load_iris()
G = build_graph(iris.data, "manhattan")
print(len(list(G.nodes())))
print(len(list(G.edges())))

print(nx2gt(G))
