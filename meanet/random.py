# -*- coding: utf-8 -*-

from __future__ import (print_function, division, absolute_import,
                        unicode_literals)


import matplotlib.pyplot as plt
import numpy as np

from .mea import MEA
from .meanet import corr_matrix_to_graph, calc_cfp_from_MEA


def shuffle(data):
    """ Return a random spiketrain by shuffling the ISIs in data

    Parameters
    ----------
        data: numpy array

    Returns
    -------
        shuffled: numpy array
    """

    isi = np.hstack([data[0], np.diff(data)])
    return np.cumsum(np.random.permutation(isi[::-1]))


def shuffle_MEA(mea, seed=None):
    np.random.seed(seed)
    shuffled = MEA()
    shuffled.dur = mea.dur
    for key in mea.keys():
        if len(mea[key]) > 0:
            shuffled[key] = shuffle(mea[key])

    return shuffled


def _cfp_corr(mea, **kwargs):
    cfp = calc_cfp_from_MEA(mea, **kwargs)
    return cfp['corr']


def bootstrap_test(mea, N=10, threshold=0.66, corr_fcn=_cfp_corr,
                   colors=['#0072B2', '#D55E00'], **kwargs):
    corr = corr_fcn(mea, **kwargs)
    graph, _ = corr_matrix_to_graph(corr, threshold=threshold)
    n_true = graph.number_of_nodes()
    e_true = graph.number_of_edges()
    nodes = np.zeros(N)
    edges = np.zeros(N)
    for i in range(N):
        shuffled = shuffle_MEA(mea)
        corr = corr_fcn(shuffled, **kwargs)
        graph, _ = corr_matrix_to_graph(corr, threshold=threshold)
        nodes[i] = graph.number_of_nodes()
        edges[i] = graph.number_of_edges()

    n_min = nodes.min()
    n_max = nodes.max()
    nodes_hist = plt.figure()
    plt.hist(nodes, bins=n_max - n_min, align='left', color=colors[0])
    plt.title('Nodes Histogram')
    plt.axvline(x=n_true, linewidth=8, color=colors[1])

    edges_hist = plt.figure()
    e_min = edges.min()
    e_max = edges.max()
    plt.hist(edges, bins=e_max - e_min, align='left', color=colors[0])
    plt.title('Edges Histogram')
    plt.axvline(x=e_true, linewidth=8, color=colors[1])
    return nodes_hist, edges_hist


def calc_random_threshold(mea, N=10, sd=3, corr_fcn=_cfp_corr, **kwargs):
    """ calculate thresholds for graph edges

    Shuffle mea data N times and calculate CFP for each shuffled mea
    thresholds are defined as the mean CFP + sd*(standard deviation)
    returns numpy array of thresholds
    """
    shuffled_corrs = np.zeros((60, 60, N))
    for i in range(N):
        shuffled = shuffle_MEA(mea)
        corr = corr_fcn(shuffled, **kwargs)
        shuffled_corrs[:, :, i] = corr

    threshold = np.mean(shuffled_corrs, axis=2)
    threshold += sd * np.std(shuffled_corrs, axis=2)
    return threshold
