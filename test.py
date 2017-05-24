from graph_tool.inference import minimize_blockmodel_dl
from sklearn import datasets

from util import build_graph, nx2gt

iris = datasets.load_iris()
G = build_graph(iris.data, "manhattan")
print(len(list(G.nodes())))
print(len(list(G.edges())))

graph = nx2gt(G)
sbm = minimize_blockmodel_dl(graph, deg_corr=False)
