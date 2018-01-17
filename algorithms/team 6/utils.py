import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import keras
import keras.backend as K
from sys import path
path.append('../../')
import spikefinder_eval
from oasis.oasis_methods import oasisAR2
from oasis.functions import estimate_parameters


# definitions for artefact corrections as manual preprocessing step
def butter_highpass(cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scipy.signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff=1., fs=100, order=5):
    # cutoff and fs in Hz
    b, a = butter_highpass(cutoff, fs, order=order)
    y = scipy.signal.filtfilt(b, a, data)
    return y


def correct_artefacts(dataset):
    Y = np.array(pd.read_csv('../' + str(dataset) + '.train.calcium.csv')).T
    if dataset == 1:
        Y[2, :900] += 2.5 / 900 * np.arange(-900, 0)
    elif dataset == 2:
        Y[7, :1960] = butter_highpass_filter(Y[7])[:1960]
        Y[12, 4870:6500] = butter_highpass_filter(Y[12])[4870:6500]
        Y[13, :1200] = .6 + butter_highpass_filter(Y[13])[:1200]
    elif dataset == 3:
        Y[5, :1500] -= 2.5 * .997**np.arange(0, 1500)
    elif dataset == 7:
        Y[11, 13000:] = np.nan
        Y[15, 19000:] = np.nan
    elif dataset == 8:
        Y[2, :1400] = .6 + butter_highpass_filter(Y[2], cutoff=2.)[:1400]
        Y[14, :500] -= .7 * .993**np.arange(0, 500)
    elif dataset == 9:
        Y[3, :4000] += 3.1 / 4000 * np.arange(-4000, 0)
        Y[4, :4000] += 3.1 / 4000 * np.arange(-4000, 0)
        Y[7, :2900] += 3.7 / 2900 * np.arange(-2900, 0)
        Y[8, :3700] += 2.3 / 3700 * np.arange(-3700, 0)
        Y[18, :1000] = Y[18, 1000:2000]
    elif dataset == 10:
        Y[15, 17200:17300] = .75
    return Y


def load_data(dataset, artefacts=True, train=True):
    if train:
        Y = correct_artefacts(dataset) if artefacts else \
            np.array(pd.read_csv('../%d.train.calcium.csv' % dataset)).T
        S = np.array(pd.read_csv('../%d.train.spikes.csv' % dataset)).T
    else:
        Y = np.array(pd.read_csv('../%d.test.calcium.csv' % dataset)).T
        S = None
    return Y, S


def runOASIS(Y, d=0.992799, r=0.0422006, perc=26, window=5000, lam_sn=45.5894, mu=-0.564023):
    prep = [y[~np.isnan(y)] - scipy.ndimage.filters.percentile_filter(
        y[~np.isnan(y)], perc, window) for y in Y]
    S = np.nan * np.zeros_like(Y)
    for i, y in enumerate(Y):
        preprocessedy = prep[i]
        y = y[~np.isnan(y)]
        # decimate to estimate noise (upsampling of spikefinder data produced artefacts)
        ydec = y[:len(y) // 10 * 10].reshape(-1, 10).mean(1)
        g, sn = estimate_parameters(ydec, 1, lags=10, fudge_factor=.97, method='mean')
        S[i, :len(preprocessedy)] = oasisAR2(
            preprocessedy - mu, d + r, -d * r, lam=lam_sn * sn *
            np.sqrt((1 + d * r) / ((1 - d * d) * (1 - r * r) * (1 - d * r))))[1]
    return S


# define pearson correlation loss
def pearson_loss(x, y):
    x_ = x - K.mean(x, axis=1, keepdims=True)
    y_ = y - K.mean(y, axis=1, keepdims=True)
    corr = K.sum(x_ * y_, axis=1) / K.sqrt(K.sum(K.square(x_), axis=1) *
                                           K.sum(K.square(y_), axis=1) + 1e-12)
    return -corr


# Use Keras to build a simple shallow 1 layer NN
def make_decoder(w, optimizer='adadelta'):
    F = keras.layers.Input(shape=(None, 1))
    s = keras.layers.Convolution1D(1, len(w), activation='relu', border_mode='same',
                                   weights=[w[:, None, None], np.array([0])])(F)
    decoder = keras.models.Model(input=F, output=s)
    decoder.compile(optimizer=optimizer, loss=pearson_loss)
    return decoder


def score(S, trueS):
    return spikefinder_eval.score(pd.DataFrame(S.T), pd.DataFrame(trueS.T))


def learn_params(infS, trueS, tau=30, b=False, verbose=0, optimizer='adadelta'):
    k = []
    for n in range(len(trueS)):
        s = trueS[n]
        s = np.hstack([np.zeros(tau // 2), s[~np.isnan(s)]])
        T = len(s)
        infs = np.hstack([infS[n], np.zeros(tau // 2)])[:T]
        ss = np.zeros((tau, T))
        for i in range(tau):
            ss[i, i:] = infs[:T - i]
        ssm = ss - ss.mean() if b else ss
        k.append(np.linalg.lstsq(np.nan_to_num(ssm.T), s)[0])
    km = np.mean(k, 0)
    k = [kk * km.dot(kk) / kk.dot(kk) for kk in k]  # normalize
    km = np.mean(k, 0)
    # convolve to obtain result
    S = np.maximum(np.array([np.convolve(s, km)[tau // 2:-tau // 2 + 1]
                             for i, s in enumerate(infS)]), 0)
    cor1 = np.mean(np.nan_to_num(score(S, trueS)))
    # REFINE postprocessing kernel by optimizing for correlation instead of least squares
    # (hardly helps)
    decoder = make_decoder(km[::-1], optimizer)  # initialize weights with least-squares solution
    history = decoder.fit(np.nan_to_num(infS[..., None]), np.nan_to_num(trueS[..., None]),
                          batch_size=len(trueS), epochs=100, verbose=verbose)
    w, b = history.model.get_weights()
    pred = np.squeeze(decoder.predict(np.nan_to_num(infS[..., None]).astype('float32')))
    cor2 = np.mean(np.nan_to_num(score(pred, trueS)))
    # return whichever is better
    return (km, 0) if cor1 > cor2 else (w.ravel()[::-1], b)


def postprocess(S, w=[0.06079751, 0.00788163, 0.05314323, 0.06572068, 0.02574128,
                      0.09723502, 0.0937969, 0.06506694, 0.08910812, 0.15414132,
                      0.17046386, 0.17425694, 0.2532278, 0.34371623, 0.43407378,
                      0.43582311, 0.41132095, 0.37028062, 0.22911859, 0.17894797,
                      0.15937749, 0.10619271, 0.09882098, 0.11790272, 0.09062019,
                      0.07682828, 0.10484864, 0.05008423, 0.05693898, 0.07668038],
                b=-0.01724054):
    return np.maximum([(np.convolve(y, w) + b)[len(w) // 2:-len(w) // 2 + 1] for y in S], 0)


def normalize(trace):
    return trace / np.nanmax(trace)


def plot_inference(F, s, spikes, neuron_idx):
    plt.figure(figsize=(8, 6))
    plt.plot(normalize(F[neuron_idx]), label='calcium fluorescence')
    plt.plot(normalize(s[neuron_idx]) - 1, label='true spikes')
    plt.plot(normalize(spikes[neuron_idx]) - 2, label='Infered spikes')
    plt.axis('off')
    plt.legend(loc='best', ncol=3)
