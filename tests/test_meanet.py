#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_meanet
----------------------------------

Tests for `meanet` module.
"""

from __future__ import (print_function, absolute_import,
                        division, unicode_literals)


import numpy as np


import meanet


def test_corr_matrix_to_graph():
    """ test corr_matrix_to_graph function """

    corr = np.asarray([[0, 0.2, 0.4, 0.6, 0.8],
                       [0.2, 0, 0.4, 0.6, 0.8],
                       [0.4, 0.4, 0, 0.2, 0.6],
                       [0.6, 0.6, 0.2, 0, 0.4],
                       [0.8, 0.8, 0.6, 0.4, 0]])

    graph, threshold = meanet.corr_matrix_to_graph(corr, 20)
    assert threshold == 0.75
    assert graph.edges() == [(0, 4), (1, 4)]

    graph, threshold = meanet.corr_matrix_to_graph(corr, 90)
    assert threshold == 0.1953125
    assert graph.edges() == [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3),
                             (1, 4), (2, 3), (2, 4), (3, 4)]


if __name__ == '__main__':
    test_corr_matrix_to_graph()
