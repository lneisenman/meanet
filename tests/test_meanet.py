#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_meanet
----------------------------------

Tests for `meanet` module.
"""

from __future__ import (print_function, absolute_import,
                        division, unicode_literals)

import matplotlib.pyplot as plt
import numpy as np


import meanet


def test_corr_matrix_to_graph():
    """ test corr_matrix_to_graph function """

    corr = np.asarray([[0, 0.2, 0.4, 0.6, 0.8],
                       [0.2, 0, 0.4, 0.6, 0.8],
                       [0.4, 0.4, 0, 0.2, 0.6],
                       [0.6, 0.6, 0.2, 0, 0.4],
                       [0.8, 0.8, 0.6, 0.4, 0]])

    graph, threshold = meanet.corr_matrix_to_graph(corr, density=20)
    assert threshold == 0.75
    assert graph.edges() == [(0, 4), (1, 4)]

    graph, threshold = meanet.corr_matrix_to_graph(corr, density=90)
    assert threshold == 0.1953125
    assert graph.edges() == [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3),
                             (1, 4), (2, 3), (2, 4), (3, 4)]


def test_check_small_world_bias(num_nodes=90, num_pnts=64, seed=1):
    """ test check_small_world_bias function """

    np.random.seed(seed)
    data = np.random.normal(0, 1, (num_nodes, num_pnts))
    corr = np.corrcoef(data)
    CCnet, CCrand, densities = meanet.check_small_world_bias(corr, seed=seed)
    print(CCnet)
    print(CCrand)
    print(densities)


def test_conditional_firing_probability():
    """ test conditional_firing_probability function """

    np.random.seed(1)
    train1 = np.linspace(1000, 100000, 100)
    train2 = train1 + np.random.normal(20, 5, train1.shape[0])

    train1 = np.asarray([0.01, 1, 1.5, 1.8]) * 1000
    train2 = np.asarray([0.02, 0.05, 1, 1.06, 1.08, 1.6, 1.7, 1.82]) * 1000
    result, psth = meanet.conditional_firing_probability(train1, train2)
    xdata = np.arange(500)
    fit = meanet.meanet._cfp_function(xdata, *result.x)
#    print(train1)
#    print(train2)
#    print(psth)
    print(result.x)
#    plt.plot(psth, '-g')
#    plt.plot(xdata, fit, '-b')
#    plt.show()


def test_shuffle():
    def isi(dat):
        return np.hstack([dat[0], dat[1:] - dat[0:-1]])

    data = np.cumsum(np.arange(1, 10))
    test = meanet.shuffle(data)
    print(data)
    print(test)
    assert np.allclose(np.sort(isi(data)), np.sort(isi(test)))


if __name__ == '__main__':
    test_shuffle()
