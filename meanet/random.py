# -*- coding: utf-8 -*-

from __future__ import (print_function, division, absolute_import,
                        unicode_literals)


import numpy as np

from .mea import MEA


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


def shuffle_MEA(mea):
    shuffled = MEA()
    shuffled.dur = mea.dur
    for key in mea.keys():
        if len(mea[key]) > 0:
            shuffled[key] = shuffle(mea[key])

    return shuffled
