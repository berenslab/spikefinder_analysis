from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style("white")

from theano import config
import numpy as np
from matplotlib import gridspec
import pickle
import os
from data_funcs import data_resamp
from utils import rebin

def plot_preds_bl(model, data, cell=0, pred=False, ts=[0, 1000], trace=0, figsize=[30,10]):
    
    n_samples = 30
    fig = plt.figure(figsize=figsize)

    cp = sns.hls_palette(11, l=.4, s=.8)
    try:
        sr = model.superres
    except KeyError:
        sr = 1

    start = 0
    end = - 1

    fluor, spikes = data_resamp(data['traces_test'], data['spikes_test'], data['fps'], data['spike_fps'], model.resample, model.superres)
    fluor = np.array(fluor[cell][trace][ts[0]:ts[1]], ndmin=2).astype(config.floatX)
    spikes = np.array(spikes[cell][trace][sr * ts[0]:sr * ts[1]], ndmin=2).astype(config.floatX)

    pred_spikes = model.mrec.getSample(fluor, n_samples)
    pred_prob = pred_spikes[:, sr * start:sr * end:sr].mean(axis=0)
    pred_spikes = pred_spikes[:, sr * start:sr * end:sr]

    s_spikes = np.sum(np.squeeze(pred_prob))
    r_spikes = np.sum(spikes[0, sr * start:sr * end])
    corr = np.corrcoef(rebin(np.squeeze(pred_prob), 3), rebin(spikes[0, sr * start:sr * end], 3))[0, 1]

    pred_inds = [np.where(pred_spikes[i])[0] for i in range(n_samples)]

    gs = gridspec.GridSpec(3, 1, height_ratios=([2, 1, 1]))

    axes = []
    for i in range(3):
        axes.append(plt.Subplot(fig, gs[i]))
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        fig.add_subplot(axes[i])

    gs.tight_layout(fig, h_pad=-1.8, rect=(0, 0, 0.7, 1))

    where = np.where(spikes[0, sr * start:sr * end] == 1)[0]
    where2 = np.where(spikes[0, sr * start:sr * end] > 1)[0]

    reals = []

    dt = 1 / model.resample / sr

    c = 1
    t = np.arange(len(fluor[0, start:end])) * dt
    t_sr = np.arange(len(fluor[0, start:end]) * sr) * dt / sr

    for i in where:
        i *= dt
        if c == 1: axes[0].axvline(i, ymin=0., ymax=.1, color='black', label='True Spikes'); c = 0;
        axes[0].axvline(i, ymin=0., ymax=.1, color='black')  # , linestyle='dotted')
        axes[1].axvline(i, color='0.75')
    axes[0].set_xlim([0, t[-1]])
    axes[1].set_xlim([0, t[-1]])
    axes[1].set_ylim([0, 1])

    for i in where2:
        i *= dt
        axes[0].axvline(i, ymin=0., ymax=.2, color='black')
        axes[1].axvline(i, color='0.75')

    for ith, trial in enumerate(pred_inds):
        axes[2].vlines(trial * dt, ith + .5, ith + 1.5, color=cp[0])
    axes[2].set_xlim([0, t[-1]])

    axes[1].set_ylabel('Predicted Probability', fontsize=14)
    axes[2].set_ylabel('Sampled Spike Trains', fontsize=14)
    axes[1].set_xlabel('Time in Seconds', fontsize=14)
    axes[1].plot(t_sr, pred_prob, color=cp[6])
    axes[0].plot(t, fluor[0, start:end], label="Trace", color=cp[4])
    axes[0].legend(loc='upper left', fontsize=14)
    axes[1].text(0.1 * t[-1], 0.85, 'Corr: ' + '{:0.2f}'.format(corr), fontsize=14)
    axes[1].text(0.1 * t[-1], 0.65, 'Spikes: ' + '{:0.2f}'.format(s_spikes) + ' / ' + str(r_spikes), fontsize=14)