# -*- coding: utf-8 -*-

from __future__ import (print_function, division, absolute_import,
                        unicode_literals)


import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def draw_MEA_graph(mea, node_color='r', node_size=400, font_size=12,
                     node_width=2, edge_width=2, fig=None):
    """ Draw a graph from MEA data """
    MEADict = dict()
    key = 0
    MEA = nx.Graph()
    corMEA = nx.Graph()
    activeMEA = nx.Graph()
    bothMEA = nx.Graph()
    DiMEA = nx.DiGraph()
    DiCorMEA = nx.DiGraph()
    fullMEA = nx.DiGraph()
    layout = dict()

    for j in range(2, 8):
        fullMEA.add_node(key)
        layout[key] = [1 - 0.1 + 0.2*(j % 2), 9 - j + 0.1]
        MEADict[key] = 10 + j
        key += 1

    for i in range(2, 8):
        for j in range(1, 9):
            fullMEA.add_node(key)
            layout[key] = [i - 0.1 + 0.2*(j % 2), 9 - j - 0.1 + 0.2*(i % 2)]
            MEADict[key] = ((10*i) + j)
            key += 1

    for j in range(2, 8):
        fullMEA.add_node(key)
        layout[key] = [8 - 0.1 + 0.2*(j % 2), 9 - j - 0.1]
        MEADict[key] = 80 + j
        key += 1

    if fig is None:
        plt.figure()

    xy = np.asarray([layout[v] for v in list(fullMEA)])
    plt.scatter(xy[:, 0], xy[:, 1], s=node_size, c='white', edgecolors='k',
                linewidths=node_width, zorder=2)

    xy = np.asarray([layout[v] for v in list(mea)])
    if len(xy) > 0:
        plt.scatter(xy[:, 0], xy[:, 1], s=node_size, c=node_color,
                    edgecolors='k', linewidths=node_width, zorder=2)

    nx.draw_networkx_edges(mea, layout, width=edge_width)
    nx.draw_networkx_labels(fullMEA, layout, font_size=font_size)
    plt.axis('off')


if __name__ == '__main__':
    graph = nx.Graph()
    graph.add_edge(0, 22)
    draw_MEA_graph(graph)
    plt.show()
