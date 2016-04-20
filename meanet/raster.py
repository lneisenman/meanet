# -*- coding: utf-8 -*-

from __future__ import (print_function, division, absolute_import,
                        unicode_literals)


import matplotlib.pyplot as plt

from mea import MEA


def draw_raster(mea):
    """ Draw a raster plot from MEA data """
    fig = plt.figure()
    for i in range(60):
        plt.eventplot(mea[i], lineoffsets=i+0.5, colors=[(0, 0, 0)])

    plt.xlim(0, mea.dur)
    plt.ylim(0, 60)
    return fig

if __name__ == '__main__':
    mea = MEA()
    mea.dur = 20
    mea[0] = range(3, 19, 3)
    mea[15] = range(1, 12)
    mea[30] = range(2, 18, 2)
    fig = draw_raster(mea)
    plt.show()
