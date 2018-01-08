import numpy as np
from scipy import signal
from utils import rebin
import copy
from theano import config
import itertools

def data_timesplit(traces, spikes, trainshare = 0.6):

    n_cells = len(traces)
    traces_train = [[] for _ in range(n_cells)]
    traces_val = [[] for _ in range(n_cells)]
    spikes_train = [[] for _ in range(n_cells)]
    spikes_val = [[] for _ in range(n_cells)]

    for c in range(n_cells):
        for i in range(len(traces[c])):
            traces_train[c].append(traces[c][i][:int(trainshare*len(traces[c][i]))])
            traces_val[c].append(traces[c][i][int(trainshare*len(traces[c][i])):])
            spikes_train[c].append(spikes[c][i][:int(trainshare*len(spikes[c][i]))])
            spikes_val[c].append(spikes[c][i][int(trainshare*len(spikes[c][i])):])

    return traces_train,traces_val,spikes_train,spikes_val

def data_resamp(traces, spikes ,fps, spikefps, resample, superres = 1):

    traces_test = copy.deepcopy(traces)
    spikes_test = copy.deepcopy(spikes)

    for i in range(len(traces)):
        for j in range(len(traces[i])):

            traces_test[i][j] = signal.resample(traces_test[i][j], int(len(traces_test[i][j])*resample/fps[i])).astype(config.floatX)
            spikes_test[i][j] = rebin(spikes_test[i][j], int(spikefps/resample/superres))[0].astype(config.floatX)
    return traces_test,spikes_test


def data_chop(timebins, traintraces, trainspikes, sv_inds=None, fluorfps=None, spikefps=60, resample=None, superres=1, fb=True):
    
    n_cells = len(traintraces)

    if sv_inds is None: sv_inds = list(np.arange(n_cells))

    traces_train = [[] for x in range(n_cells)]
    spikes_train = [[] for x in range(n_cells)]

    for i in range(n_cells):

        spikebool = isinstance(trainspikes, (np.ndarray, list)) and i in sv_inds

        for j in range(len(traintraces[i])):

            if resample:
                length = int(len(traintraces[i][j]) * resample / fluorfps[i])
                trace = signal.resample(traintraces[i][j], length)
                if spikebool: spike = (rebin(trainspikes[i][j], int(spikefps / (resample * superres)))).flatten()
            else:
                trace = traintraces[i][j]
                if spikebool: spike = trainspikes[i][j]

            trace = trace.astype(config.floatX)
            p = len(trace) // timebins
            if spikebool:
                if len(trace) * superres > len(spike):
                    p = len(spike) // superres // timebins

            trace_f = trace[:p * timebins].reshape([-1, timebins])
            trace_b = trace[-p * timebins:].reshape([-1, timebins])
            if fb: trace_f = np.concatenate((trace_f, trace_b), axis=0)
            traces_train[i].append(trace_f)

            if spikebool:

                spike_f = spike[:p * timebins * superres].reshape([-1, timebins * superres])
                spike_b = spike[-p * timebins * superres:].reshape([-1, timebins * superres])
                if fb: spike_f = np.concatenate((spike_f, spike_b), axis=0)
                spikes_train[i].append(spike_f)

        
            traces_train[i] = np.vstack(traces_train[i]).astype(config.floatX)
            if spikebool: spikes_train[i] = np.vstack(spikes_train[i]).astype(config.floatX)

    return traces_train, spikes_train

def DatasetMiniBatchIterator(model, traces, spikes):
    """ Basic mini-batch iterator """

    sv_inds = np.where(np.array([len(s) for s in spikes]) > 0)[0]
    probs = np.array([len(f) for f in traces])

    probs = probs / np.sum(probs)

    n_cells = len(traces)
    us_ind = []

    for i in range(n_cells):
        us_ind.append(list(np.arange(len(traces[i]))))

    rem = list(np.arange(n_cells))

    for _ in range(model.print_freq):

        c = model.rng.choice(rem, 1, p=probs)[0]
        choice = model.rng.choice(np.array(us_ind[c]), model.batchSize, replace=False)

        truth = np.array([0], ndmin=2, dtype=config.floatX)

        if len(spikes[c]) > 0:
            truth = spikes[c][choice]

        yield traces[c][choice], c, truth, choice