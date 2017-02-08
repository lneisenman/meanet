# -*- coding: utf-8 -*-

from __future__ import (print_function, division, absolute_import,
                        unicode_literals)

import networkx as nx
import numpy as np

from .meanet import corr_matrix_to_graph


def analyse_data(data, threshold=0.66):
    """ perform graph theory analysis on data

    Parameters
    ----------
    data:   dict
        the keys are the names of the datasets
        and the values are dicts that include 'corr' which represents
        the corr matrix from which to derive the graph

    Returns
    -------
    result: dict of graph theory results
        the keys are the names of the datasets
        the values are another dict containing
        'L' - the average shortest path length
        'CC' - the average clustering coefficient
        'DD' - the degree histogram
        'Nodes' - the number of nodes in the graph
        'Edges' - the number of edges in the graph
    """
    result = dict()
    for label, dataset in data.items():
        summary = dict()
        corr = dataset['corr']
        graph, _ = corr_matrix_to_graph(corr, threshold=threshold)
        summary['L'] = nx.average_shortest_path_length(graph)
        summary['CC'] = nx.average_clustering(graph)
        summary['DD'] = nx.degree_histogram(graph)
        summary['Nodes'] = graph.number_of_nodes()
        summary['Edges'] = graph.number_of_edges()
        result[label] = summary

    return result


def _distance_matrix(G):
    """ create a numpy 2-d array of distances between nodes

    Parameters
    ----------
    G : NetworkX undirected graph


    Returns
    -------
    matrix: ndarray
        numpy 2-d array of distances between nodes

    """

    size = len(G)
    matrix = np.zeros((size, size))
    nodes = nx.nodes(G)
    for i, node1 in enumerate(nodes):
        for j, node2 in enumerate(nodes):
            try:
                matrix[i, j] = nx.shortest_path_length(G, node1, node2)
            except:
                pass

    return matrix


def old_average_shortest_path_length(G):
    """ Compute the average shortest path length (L) of the graph
        assuming that the pathlength between unconnected nodes is equal to zero

    Parameters
    ----------
    G : NetworkX undirected graph

    Returns
    -------
    L: float
        the average shortest path length (L) of the graph

    Notes
    -----
    This is based on the old NetworkX behavior. The current behavior is to
    raise an exception if there are unconnected nodes
    """

    # test for correct type of input
    if not isinstance(G, nx.classes.graph.Graph):
        raise TypeError('This function only works for undirected graphs')

    # make sure the Graph isn't empty
    if len(G) == 0:
        raise ValueError('The graph is empty')

    # create a numpy 2-d array of distances between nodes called matrix
    matrix = _distance_matrix(G)

    # calculate L
    size = matrix.shape[0]
    L = matrix.sum()/(size*(size - 1))
    return L


def bullmore_average_shortest_path_length(G):
    """ Compute the average shortest path length (L) of the graph
        assuming that the pathlength between unconnected nodes is equal to
        the longest path length between connected nodes in the network

    Parameters
    ----------
    G : NetworkX undirected graph

    Returns
    -------
    L: float
        the average shortest path length (L) of the graph

    Notes
    -----
    This is based on Fornito et al Front Syst Neurosci 4:22 2010 and references
    therein
    """

    # test for correct type of input
    if not isinstance(G, nx.classes.graph.Graph):
        raise TypeError('This function only works for undirected graphs')

    # make sure the Graph isn't empty
    if len(G) == 0:
        raise ValueError('The graph is empty')

    # create a numpy 2-d array of distances between nodes called matrix
    matrix = _distance_matrix(G)

    # set all zero distances to the max distance in matrix
    maxdist = np.nanmax(matrix)
    indices = np.where(matrix == 0)
    matrix[indices] = maxdist

    # reset distances from each node to itself back to zero
    np.fill_diagonal(matrix, 0)

    # calculate L
    size = matrix.shape[0]
    L = matrix.sum()/(size*(size - 1))
    return L


def small_world_random(G):
    """ Compute the average clustering coefficient and average shortest path
        length of a random network with the same number of nodes and edges as G

    Parameters
    ----------
    G: Network X undirected graph

    Returns
    -------
    C: float
        the random network clustering coefficient
    L: float
        the random network average shortest path length

    Notes
    -----
    Formulas from Albert and Barabasi 2002
    """

    N = len(G)
    d = 2 * G.number_of_edges() / N
    C = d / N
    L = np.log(N) / np.log(d)
    return C, L
