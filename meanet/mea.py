# -*- coding: utf-8 -*-

from __future__ import (print_function, division,
                        absolute_import, unicode_literals)

import numpy as np

import h5shelve


class MEA():
    """ This class implements a dictionary of numpy arrays that can be indexed
    by a string composed of the row and column number of the corresponding MEA
    or by an integer index between zero and 59. It also has a field ('dur')for
    the duration of the data collection.
    """

    def __init__(self, filename=None):

        self._ordered_keys = ['12', '13', '14', '15', '16', '17',
                              '21', '22', '23', '24', '25', '26', '27', '28',
                              '31', '32', '33', '34', '35', '36', '37', '38',
                              '41', '42', '43', '44', '45', '46', '47', '48',
                              '51', '52', '53', '54', '55', '56', '57', '58',
                              '61', '62', '63', '64', '65', '66', '67', '68',
                              '71', '72', '73', '74', '75', '76', '77', '78',
                              '82', '83', '84', '85', '86', '87']

        self.dur = 0
        self.spike_times = {}
        for key in self._ordered_keys:
            self.spike_times[key] = np.zeros(0)

    def __len__(self):
        return len(self.spike_times)

    def __getitem__(self, key):
        if key in self.spike_times.keys():
            return self.spike_times[key]
        elif (isinstance(key, int)) and (0 <= key < 60):
            return self.spike_times[self._ordered_keys[key]]
        else:
            raise KeyError

    def __setitem__(self, key, value):
        if key in self.spike_times.keys():
            self.spike_times[key] = value
        elif (isinstance(key, int)) and (0 <= key < 60):
            self.spike_times[self._ordered_keys[key]] = value
        else:
            raise KeyError

    def __iter__(self):
        for key in self._ordered_keys:
            yield self.spike_times[key]

    def iterkeys(self):
        return iter(self.keys())

    def keys(self):
        return self.spike_times.keys()

    def __contains__(self, item):
        if item in self.spike_times.keys():
            return True
        elif (isinstance(item, int)) and (0 <= item < 60):
            return True
        else:
            return False


def time_window(mea, start_time, end_time):
    """ return a new MEA with spikes that occur at or after 'start' and before
        'end' """
    assert start_time >= 0
    assert end_time <= mea.dur

    windowed_mea = MEA()
    windowed_mea.dur = mea.dur
    for i in range(60):
        window = np.where(np.logical_and(start_time <= mea[i],
                                         mea[i] < end_time))
        windowed_mea[i] = mea[i][window].copy()

    return windowed_mea
