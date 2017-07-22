import matplotlib
from Orange.data import Table

matplotlib.use('Agg')

from graph_tool.draw import sfdp_layout, graph_draw

from induce_graph import induce_graph, cutoff

iris = Table('iris')
graph = induce_graph(iris)

graph_1 = cutoff(graph, 0.1)
graph_2 = cutoff(graph, 0.2)
graph_3 = cutoff(graph, 0.3)

vertex_layout = sfdp_layout(graph_2)

rgb_colors = ([[70, 190, 250]] * 50) + ([[237, 70, 47]] * 50) + ([[170, 242, 43]] * 50)
rgb_colors = [[r / 255, g / 255, b / 255] for r, g, b in rgb_colors]

colors = graph_1.new_vertex_property('vector<double>')
for node, color in zip(graph_1.vertices(), rgb_colors):
    colors[node] = color

graph_draw(graph_1, vertex_layout, vertex_fill_color=colors, output='iris_threshold_01.png', output_size=(600, 400))
graph_draw(graph_2, vertex_layout, vertex_fill_color=colors, output='iris_threshold_02.png', output_size=(600, 400))
graph_draw(graph_3, vertex_layout, vertex_fill_color=colors, output='iris_threshold_03.png', output_size=(600, 400))
