# -*- coding: utf-8 -*-

from __future__ import (print_function, division, absolute_import,
                        unicode_literals)


import networkx as nx
import numpy as np


def corr_matrix_to_graph(corr, density, directed=False, tol=0.01):
    """ Convert a correlation matrix to a NetworkX graph

    This function finds a threshold level such that creating a connection
    matrix where a connection exists if the correlation is greater or equal to
    the threshold results in a network with the specified density of
    connections.

    corr: NxN 2-D numpy array where N is the number of nodes in the network.
          Values should all be between 0 and 1

    density: value between 0 and 100 representing the percentage of possible
             connections that should be created

    directed: boolean. If true, return a directed network

    returns: a tuple consisting of the resulting Networkx graph and the
             corresponding threshold

    """

    N = corr.shape[0]
    num_connections = N * (N - 1) * density / 100
    if not directed:
        num_connections /= 2

    low, high = 0, 1
    while (high - low >= tol):
        threshold = (low + high) / 2
        adj_matrix = np.zeros_like(corr, dtype=np.int)
        index = np.where(corr >= threshold)
        adj_matrix[index] = 1
        graph = nx.from_numpy_matrix(adj_matrix)
        num_edges = graph.number_of_edges()
        if num_edges == num_connections:
            return graph, threshold

        if num_edges > num_connections:
            # threshold is too low
            low = threshold
        else:
            # threshold is too high
            high = threshold

    return graph, threshold
