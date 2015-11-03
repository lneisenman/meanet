# -*- coding: utf-8 -*-

from __future__ import (print_function, division, absolute_import,
                        unicode_literals)


import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def draw_MEA_graph(mea, node_color='r', node_size=400, font_size=12,
                     node_width=2, edge_width=2):
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

    nx.draw_networkx(fullMEA, layout, node_color='w', node_size=node_size,
                     linewidths=node_width, font_size=font_size)
    nx.draw_networkx(mea, layout, node_color=node_color, node_size=node_size,
                     linewidths=node_width, font_size=font_size,
                     width=edge_width)


if __name__ == '__main__':
    graph = nx.Graph()
    graph.add_edge(0, 22)
    draw_MEA_graph(graph)
    plt.axis('off')
    plt.show()
