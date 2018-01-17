#!/usr/bin/env python
"""
This script is used to convert the float values given as the
output of the neural network into integer-valued spike counts.
"""

from __future__ import print_function
from __future__ import division

from spikefinder import load, score
import numpy as np

for i in range(1, 11):
    name_1 = '/tmp/%d.train.spikes.csv' % i
    name_2 = 'data/spikefinder.train/%d.train.spikes.csv' % i
    a = load(name_1)
    b = load(name_2)

    a_num = np.nan_to_num(a)
    mask = np.isnan(a)

    best_mean, best_median = 0, 0
    mean_cut, med_cut = 0, 0

    for cutoff in np.arange(0.1, 0.7, 0.01):
        a_cutoff = np.round(a / cutoff)

        s = score(a_cutoff, b)

        mean_v = np.nanmean(s)
        med_v = np.nanmedian(s)

        if mean_v > best_mean:
            best_mean = mean_v
            mean_cut = cutoff

        if med_v > best_median:
            best_median = med_v
            med_cut = cutoff

    print('train %d: [' % i,
          'mean = %f, cutoff = %f;' % (best_mean, mean_cut),
          'median = %f, cutoff = %f' % (best_median, med_cut),
          ']')

