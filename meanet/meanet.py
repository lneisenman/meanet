# -*- coding: utf-8 -*-

from __future__ import (print_function, division, absolute_import,
                        unicode_literals)


import networkx as nx
import numpy as np
import scipy.optimize as opt


def _threshold_corr_matrix_to_graph(corr, threshold, directed=False):
    """ Convert a correlation matrix to a NetworkX graph by thresholding
        the correlation matrix. This version preserves the
        identity of the contacts.
    """
    adj = _threshold_corr_matrix(corr, threshold)
    N = adj.shape[0]
    graph = nx.Graph()
    if directed:
        graph = nx.DiGraph()

    for i in range(N):
        for j in range(N):
            if adj[i][j] == 1:
                graph.add_edge(i, j)

    return graph


def _threshold_corr_matrix(corr, threshold, directed=False):
    """ Convert a correlation matrix to an adjacency matrix by thresholding
        the correlation matrix.
    """

    adj_matrix = np.zeros_like(corr, dtype=np.int)
    index = np.where(corr >= threshold)
    adj_matrix[index] = 1
    return adj_matrix


def corr_matrix_to_graph(corr, threshold=0.5, density=None,
                           directed=False, tol=0.01):
    """ Convert a correlation matrix to a NetworkX graph

    This function finds a threshold level such that creating a connection
    matrix where a connection exists if the correlation is greater or equal to
    the threshold results in a network with the specified density of
    connections.

    Parameters
    ----------
    corr: NxN 2-D numpy array where N is the number of nodes in the network.
          Values should all be between 0 and 1

    threshold: value between zero and 1 such that an edge is present if the
               corresponding value of corr is >= threshold

    density: value between 0 and 100 representing the percentage of possible
             connections that should be created. If None, the network is
             created based on the value of density

    directed: boolean. If true, return a directed network

    Returns
    --------
    (graph, threshold): a tuple consisting of the resulting Networkx graph and
        the corresponding threshold

    """

    if density is None:
        return (_threshold_corr_matrix_to_graph(corr, threshold, directed),
                threshold)

    N = corr.shape[0]
    num_connections = N * (N - 1) * density / 100
    if not directed:
        num_connections /= 2
    num_connections = int(num_connections)

    low, high = 0, 1
    while (high - low >= tol):
        threshold = (low + high) / 2
        adj = _threshold_corr_matrix(corr, threshold, directed)
        graph = nx.Graph(adj)
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


def check_small_world_bias(corr, dmin=3, dmax=30, dnum=10,
                             randomizations=20, seed=1):
    """ Test to see if a method for defining edges produces networks with
        excess clustering

        This is based on Zalesky et al. Neuroimage 60:2096 2012
        which demonstrated that standard correlation applied to random time
        series resulted in networks with unexpectedly high clustering. The
        cause is the observation that if A is correlated with B and B is
        correlated with C, there is a high probability that A is correlated
        with C

    """

    np.fill_diagonal(corr, 0)   # no self connections
#    L_null = np.zeros(10)
    C_null = np.zeros(10)
#    L_rand = np.zeros(10)
    C_test = np.zeros(10)
    thresholds = np.zeros(10)
    densities = np.linspace(3, 30, 10)
    for i, density in enumerate(densities):
        graph, threshold = corr_matrix_to_graph(corr, density, tol=0.0001)
        if len(graph) > 1:
#            L_test[i] = nx.average_shortest_path_length(graph)
            C_test[i] = nx.average_clustering(graph)
            thresholds[i] = threshold
#            L = np.zeros(randomizations)
            C = np.zeros(randomizations)
            for r in range(randomizations):
                random = nx.expected_degree_graph(graph.degree().values(),
                                                  selfloops=False,
                                                  seed=seed + density + 100*r)
#                L[r] = nx.average_shortest_path_length(random)
                C[r] = nx.average_clustering(random)

#            L_null[i] = np.average(L)
            C_null[i] = np.average(C)

    return C_test, C_null, densities


def _cfp_function(x, maxi, delay, width, offset):
    """ function to which the cfp result is fit

        cfp = (max/(1 + ((x - delay)/width)**2)) + offset
    """

    return (maxi/(1 + ((x - delay)/width)**2)) + offset


def _cfp_cost(p, y, x):
    """ cost function for use in opt.minimize for fitting _cfp_function """

    maxi, delay, width, offset = p
    residual = y - _cfp_function(x, maxi, delay, width, offset)
    residual_squared = np.square(residual)
    return np.sum(residual_squared)


def conditional_firing_probability(train1, train2, min_points=0,
                                   method=None):
    """ Calculate the conditional firing probability between two lists of
        spike times

        train1: numpy array of spike times in msec
        train2: numpy array of spike times in msec
        threshold: minimum amplitude of fit
        delay: minimum peak time of fit in msec

        For LeFeber the fit was performed with the following limits
            0 <= max < 1
            0 <= delay <= 500
            1 <= width <= 100
            0 <= offset <= 0.5

        The followng was required to declare a connection
            max/offset >= 2
            5<= delay <= 250
            width > 5

    """
    class Empty_Fit():
        def __init__(self):
            self.x = np.zeros(4)
            self.x[-1] = 1

    psth = np.zeros(500)
    max2 = train2.shape[0]
    indices = np.searchsorted(train2, train1, side='right')
    for i, index in enumerate(indices):
        j = index
        while (j < max2) and (train2[j] - train1[i] < 500):
            psth[int(train2[j] - train1[i])] += 1
            j += 1

    if np.sum(psth) < min_points:
        return Empty_Fit(), psth

    psth /= train1.shape[0]
    xdata = np.arange(500)
    offset = np.average(psth)
    maxi = np.max(psth) - offset
    delay = np.argmax(psth)
    width = delay/2 if delay/2 > 5 else 5
    p0 = (maxi, delay, width, offset)
    bounds = ((0, 1), (0, 500), (1, 100), (0, 0.5))
    if method == 'DE':
        fit = opt.differential_evolution(_cfp_cost, bounds=bounds, seed=1,
                                         args=(psth, xdata), maxiter=100,
                                         polish=False)
        p0 = fit.x

    fit = opt.minimize(_cfp_cost, p0, bounds=bounds, args=(psth, xdata),
                       method='L-BFGS-B')
    return fit, psth


def calc_cfp_from_MEA(mea, min_points=0, method=None):
    cfp = conditional_firing_probability
    active = [key for key in range(60) if len(mea[key]) >= 250]
    N = len(active)
    peak = np.zeros((60, 60))
    delay = np.zeros((60, 60))
    width = np.zeros((60, 60))
    baseline = np.zeros((60, 60))
    num = np.zeros((60, 60))
    psth = np.zeros((60, 60, 500))
    for i in range(N-1):
        for j in range(i+1, N):
            ai = active[i]
            aj = active[j]
            fit, psth[ai, aj, :] = cfp(mea[ai]*1000, mea[aj]*1000, min_points,
                                       method)
            peak[ai, aj] = fit.x[0]
            delay[ai, aj] = fit.x[1]
            width[ai, aj] = fit.x[2]
            baseline[ai, aj] = fit.x[3]
            num[ai, aj] = len(mea[ai])

            fit, psth[aj, ai, :] = cfp(mea[aj]*1000, mea[ai]*1000, min_points,
                                       method)
            peak[aj, ai] = fit.x[0]
            delay[aj, ai] = fit.x[1]
            width[aj, ai] = fit.x[2]
            baseline[aj, ai] = fit.x[3]
            num[aj, ai] = len(mea[aj])

    corr = _calc_corr_matrix(peak, delay, width, baseline)
    return {'peak': peak, 'delay': delay, 'width': width, 'baseline': baseline,
            'psth': psth, 'corr': corr, 'num': num}


def _calc_corr_matrix(peak, delay, width, baseline):
    """
    5<= delay <= 250
    width > 5
    """
    corr = np.zeros_like(peak)
    index = np.where(peak > 0)
    corr[index] = peak[index] / (peak[index] + baseline[index])
    index = np.where(delay < 5)
    corr[index] = 0
    index = np.where(delay > 250)
    corr[index] = 0
    index = np.where(width < 5)
    corr[index] = 0
    return corr
