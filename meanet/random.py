# -*- coding: utf-8 -*-

from __future__ import (print_function, division, absolute_import,
                        unicode_literals)


import numpy as np


def shuffle(data):
    """ Return a random spiketrain by shuffling the ISIs in data

    Parameters
    ----------
        data: numpy array

    Returns
    -------
        shuffled: numpy array
    """

    isi = data[1:] - data[0:-1]
    isi = np.hstack([data[0], isi])
    return np.cumsum(np.random.permutation(isi[::-1]))
