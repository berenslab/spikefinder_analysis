from collections import defaultdict

from scipy.interpolate import interp1d
import scipy.stats as stat
import scipy.signal as signal
import scipy.linalg as linalg
import pandas as pd
import numpy as np
import keras
import os

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    print('Plotting libraries not imported.')

import spikefinder

# Which indicator was used in each dataset
INDICATORS = ['OGB', 'OGB', 'GCaMP6S', 'OGB', 'GCaMP6S',
              'GCaMP5K', 'GCaMP6F', 'GCaMP6S', 'jRCaMP', 'jRGECO']

# Index of model AR order in decoder filename
AR_IDX = 11

# Spikefinder downsamples data by a factor of 4 before evaluation
SPIKEFINDER_DOWNSAMPLE = 4


def infer_spikes(F, s=None, factor=1, name='Models/default.hdf5'):
    """
    Evaluate fluorescence data using a default model

    Parameters
    ----------
    F : pandas.DataFrame or np.array, shape (T,N)
        Fluorescence traces. This is the only required argument.
    s : pandas.DataFrame or np.array, shape (T,N), optional, default=None
        Spike trains (for computing accuracy, optional)
    name : string containing path to model to use, default=None
        Name/path to the model to use.
    factor : bool, optional, default=1
        Downsample data before estimating parameters/running inference?
        This is recommended for most spikefinder datasets, but probably
        not in most actual usage scenarios.

    Returns
    -------
    results_df : pandas.DataFrame
        Dataframe containing many fields and all spike inference results
    spikes_df : pandas.DataFrame
        List dataframes of size (N,T) containing inference from each model
    """

    # Make F and s into dataframes if they aren't already
    if type(F) == np.ndarray:
        F = get_dataframe(F)
    if s is not None and type(s) == np.ndarray:
        s = get_dataframe(s)

    # Load model
    model = keras.models.load_model(name)

    # Process each dataset
    out_n = []
    for neuron_idx in range(np.shape(F)[1]):
        Fc = F[str(neuron_idx)][np.isfinite(F[str(neuron_idx)])]
        Fc = np.reshape(Fc, (1, -1))
        Fc_ = downsample(Fc, factor)

        # If we need to append covariates, then do it
        Fc_ = make_data_stack(Fc_, factor=factor, p=2)
        pred_n = np.squeeze(model.predict(Fc_))
        pred_n = upsample_inference(pred_n, np.shape(Fc)[1])

        # Round output a bit to get some actual zeros and speed things up
        pred_n = np.round(pred_n, 2)
        out_n.append(pred_n)

    # Save out the results (and correlations with ground truth)
    spikes_df = get_dataframe(out_n)

    # Compute error rate if spikes are provided
    scores_df = None
    if s is not None:
        scores_df = spikefinder.score(spikes_df, s)
        return spikes_df, scores_df
    return spikes_df


def axcov(data, maxlag=5):
    """ From OASIS. """
    data = data - np.mean(data)
    T = len(data)
    exponent = 0
    while 2 * T - 1 > np.power(2, exponent):
        exponent += 1
    xcov = np.fft.fft(data, np.power(2, exponent))
    xcov = np.fft.ifft(np.square(np.abs(xcov)))
    xcov = np.concatenate([xcov[np.arange(xcov.size - maxlag, xcov.size)],
                           xcov[np.arange(0, maxlag + 1)]])
    return np.real(xcov / T)


def estimate_noise_scale(y, range_ff=[0.25, 0.5], method='mean'):
    """ From OASIS. """
    ff, Pxx = signal.welch(y)
    ind1 = ff > range_ff[0]
    ind2 = ff < range_ff[1]
    ind = np.logical_and(ind1, ind2)
    Pxx_ind = Pxx[ind]
    sn = lambda Pxx_ind: np.sqrt(np.mean(Pxx_ind / 2))
    sn = sn(Pxx_ind)
    return sn


def estimate_time_constant(y, p, sn, lags=5, fudge_factor=1.):
    """ From OASIS. """
    lags += p
    xc = axcov(y, lags)
    xc = xc[:, np.newaxis]

    l_i = np.eye(lags, p)
    A = linalg.toeplitz(xc[lags + np.arange(lags)],
                        xc[lags + np.arange(p)]) - sn**2 * l_i
    g = np.linalg.lstsq(A, xc[lags + 1:])[0]
    gr = np.roots(np.concatenate([np.array([1]), -g.flatten()]))
    gr = (gr + gr.conjugate()) / 2.
    gr[gr > 1] = 0.95 + np.random.normal(0, 0.01, np.sum(gr > 1))
    gr[gr < 0] = 0.15 + np.random.normal(0, 0.01, np.sum(gr < 0))
    g = np.poly(fudge_factor * gr)
    g = -g[1:]

    return g.flatten()


def downsample(signal, factor):
    """ Downsample signal by factor. From C2S. """
    if factor < 2:
        return np.asarray(signal)
    return np.convolve(np.asarray(signal).ravel(),
                       np.ones(factor), 'valid')[::factor]


def _run_nn(Fc, model, high_resolution=False,
            covariates=False, factor=7, fixed=None, p=1):
    """ Preprocess Fc and run the NN in model. """

    # Do inference on downsampled data if desired
    Fc_ = Fc if high_resolution else downsample(Fc, 4)

    # If we need to append covariates, then do it
    Fc_ = make_data_stack(Fc_, factor=factor, p=p,
                          fixed=fixed) if covariates else Fc_
    pred = np.squeeze(model.predict(Fc_))
    return pred


def plot_results(results_df, kind='box', x='Dataset'):
    """
    Plot the results dataframe.

    Parameters
    ----------
    results_df : pandas.DataFrame
        Dataframe containing many fields and all spike inference results
    kind : string, optional, default 'box'
        Choose 'box' or 'bar' for boxplot or barplot, respectively.
    x : string, optional, default 'Dataset'
        Choose 'Indicator' or 'Dataset' to plot either on the x axis.
    """
    plt.figure(figsize=(20, 5))
    if kind == 'box':
        sns.boxplot(data=results_df, x=x, y='Correlation', hue='Decoder')
    else:
        sns.barplot(data=results_df, x=x, y='Correlation', hue='Decoder')
    plt.ylim([0, 1])
    plt.legend(loc='best', ncol=2)


def load_evaluation_data(ds, train=True):
    """
    Load the spike finder dataset at 100 Hz.

    Parameters
    ----------
    ds : int between 1 and 10
        Which dataset to load
    train : bool, optional, default True
        Load training or testing data

    Returns
    -------
    calcium : pandas.DataFrame, shape (T,N)
        Fluorescence traces
    spikes : pandas.DataFrame, shape (T,N)
        Spike trains
    """

    kind = 'train' if train else 'test'
    spikes = []

    # Load up the data from the CSV files
    current_path = os.path.dirname(os.path.abspath(__file__))
    base_path = current_path + os.path.sep + 'Data' + os.path.sep
    calcium_path = base_path + str(ds) + '.' + kind + '.calcium.csv'
    spike_path = base_path + str(ds) + '.' + kind + '.spikes.csv'

    calcium = spikefinder.load(calcium_path)
    if train:
        spikes = spikefinder.load(spike_path)

    return calcium, spikes


def make_data_stack(F, p=1, factor=7, fixed=None):
    """
    This processes a single fluorescence vector and appends
    a constant vector with the gamma (decay) estimate to it.

        Parameters
        ----------
        F : (T,) sized np.array
            Fluorescence from a single neuron
        p : int, default 1
            Order of AR filter (must be 1 or 2)
        factor : int, default 7
            Factor to downsample data by to compute noise scale.
            7 is a reasonable value assuming the datasets were really
            sampled at ~15 Hz but data is being supplied at 100 Hz.
        fixed : int, default None
            [sn, gamma] manually estimated.

        Returns
        -------
        stack : (2,T) sized np.array
            Matrix containing F and gamma
    """
    F = np.squeeze(F)
    if np.ndim(F) > 1:
        raise ValueError('Pass a single vector, not a matrix of F')
    sn = np.ones(len(F))
    g1 = np.ones(len(F))
    g2 = np.ones(len(F))

    if fixed is not None:
        sn[:] = fixed[0]
        if len(fixed) > 1:
            gam_ = fixed[1:]
        else:
            gam_ = estimate_time_constant(F, p, sn[0])
    else:
        # We need to separately estimate sigma on decimated data
        F_ = F[::factor]
        sn[:] = estimate_noise_scale(F_)

        # Now estimate gamma on the full fluorescence trace
        gam_ = estimate_time_constant(F, p, sn[0])

    g1[:] = gam_[0]
    stack = np.stack([F, sn, g1]).T
    if p > 1:
        g2[:] = gam_[1]
        stack = np.stack([F, sn, g1, g2]).T

    return np.expand_dims(stack, 0)


def upsample_inference(inference, output_length):
    """
    Upsample a spike inference vector to a desired output_length
    using nearest neighbor interpolation.

    Parameters
    ----------
    spikes : (T,) sized np.array
        Spike inference from a single neuron
    output_length: int
        Size to upsample spikes to.

    Returns
    -------
    spikes : (output_length) sized np.array
        Upsampled spike vector
    """
    if np.ndim(inference) > 1:
        raise ValueError('Pass a single vector, not a matrix of inference')
    xn = np.linspace(0, len(inference), output_length)
    x = np.linspace(0, len(inference), len(inference))
    inference = np.array(inference)
    interp = interp1d(x, inference, 'nearest')
    return interp(xn)


def get_dataframe(spikes):
    """
    Upsample a spike inference vector to a desired output_length
    using nearest neighbor interpolation.

    Parameters
    ----------
    spikes : (N,T) sized np.array
        Spike inference from a all neurons evaluated with a model

    Returns
    -------
    df : pandas.DataFrame
        Spike inference dataframe of size (T,N)
    """
    df = pd.DataFrame(spikes)
    df = df.T
    df.columns = df.columns.astype(str)
    return df


def evaluate_dataset(F, s, models, names, dataset=-1, high_resolution=False,
                     p=1, covariates=False, train=True, factor=7,
                     fixed=None, kernel_scale=None, kernel_size=17):
    """
    Evaluate fluorescence data vs. true fluorescence and
    all provided neural network models.

    Parameters
    ----------
    F : pandas.DataFrame, shape (T,N)
        Fluorescence traces
    s : pandas.DataFrame, shape (T,N)
        Spike trains
    models : list of Keras models
        All decoder models to evaluate data with
    names : list of strings
        Name of each model in models
    dataset : int, optional, default=-1
        Dataset number currently being processed
    high_resolution : bool, optional, default=False
        Should NNs process data at 25 Hz or at 100 Hz?
    covariates : bool, optional, default=False
        Should we append parameter estimates to NN model input?
    fixed : list of len == 2
        Pass in [sigma, gamma] or [sigma] instead of estimating parameters.
    factor : int, optional, default=7
        Decimating rate for estimating parameters.

    Returns
    -------
    results_df : pandas.DataFrame
        Dataframe containing many fields and all spike inference results
    spikes_df : pandas.DataFrame
        List dataframes of size (N,T) containing inference from each model
    """
    out_n = defaultdict(list)
    spikes_df = defaultdict(list)
    results_df = pd.DataFrame()

    # Process each dataset
    for neuron_idx in range(np.shape(F)[1]):
        Fc = F[str(neuron_idx)][np.isfinite(F[str(neuron_idx)])]
        Fc = np.reshape(Fc, (1, -1))

        for model_idx, model in enumerate(models):
            if len(p) > 1:
                p_ = p[model_idx]
            pred_n = _run_nn(Fc, model, high_resolution,
                             covariates, factor, fixed, p=p_)

            if kernel_scale is not None:
                dist = stat.norm(scale=kernel_scale)
                kernel = dist.pdf(np.linspace(-5, 5, kernel_size))
                pred_n = np.convolve(pred_n, kernel, 'same')

            # If inference was done at 25 Hz, upsample the output
            if not high_resolution:
                pred_n = upsample_inference(pred_n, np.shape(Fc)[1])

            # Round output a bit to get some actual zeros and speed things up
            pred_n = np.round(pred_n, 2)
            out_n[model_idx].append(pred_n)

    # Save out the results (and correlations with ground truth)
    scores_models = []
    for model_idx in range(len(models)):
        frame = get_dataframe(out_n[model_idx])
        spikes_df[model_idx].append(frame)
        if train:
            scores_models.append(spikefinder.score(frame, s))

    # Save out a big dict with metadata
    if train:
        for neuron_idx in range(np.shape(F)[1]):
            metadata = {'Neuron': neuron_idx, 'Dataset': dataset,
                        'Fluorescence': F[str(neuron_idx)],
                        'Spikes': s[str(neuron_idx)],
                        'Indicator': INDICATORS[dataset-1]}

            neuron = []

            for model_idx in range(len(out_n)):
                out_n_curr = out_n[model_idx][neuron_idx]
                nn = {'Decoder': names[model_idx],
                      'Correlation': scores_models[model_idx][neuron_idx],
                      'Inference': out_n_curr/max(out_n_curr)}
                neuron.append(dict(nn, **metadata))

            results_df = results_df.append(neuron, ignore_index=True)

    # Return the summary dataframe
    return results_df, spikes_df


def get_best_models(results_df, names, run='Unknown', display=True,
                    plot=True, mean=True):
    """
    Figure out which NN was best for each dataset. Plot + print results.

    Parameters
    ----------
    results_df : pandas.DataFrame
        Dataframe containing many fields and all spike inference results
    names : list of strings
        Name of each model in models
    run : string, optional, default='Unknown'
        Name of this parameter sweep (for naming saved plots)
    dataset : int, optional, default=-1
        Dataset number currently being processed
    display : bool, optional, default=True
        Should we print out the results?
    plot : bool, optional, default=True
        Should we make a plot showing performance of the best models?
    mean : bool, optional, default=True
        Should we plot the mean or a median value?

    Returns
    -------
    best_models : list of strings
        Names of best models for each dataset.
    best_corrs : list of float
        Correlation values for best model at each dataset.
    """
    best_corrs = []
    best_models = []
    x = np.arange(1, 11)
    for ds in x:
        brief = results_df.loc[results_df['Dataset'] == ds]
        best_corr = 0
        best_name = 'Nothing'
        for name in brief['Decoder'].unique():
            decoder_sort = brief.loc[brief['Decoder'] == name]
            if mean:
                corr = decoder_sort.mean()['Correlation']
            else:
                corr = decoder_sort.median()['Correlation']
            if corr > best_corr:
                best_corr = corr
                best_name = name
        best_corrs.append(best_corr)
        best_models.append(best_name)

    if plot:
        plt.plot(x, best_corrs, label='Best NN')

        plt.ylim([.2, .8])
        plt.legend(loc='best')
        plt.xlabel('Dataset')
        plt.ylabel('Correlation')
        plt.savefig('Plots/' + run + '-scale' + '-best-all.pdf')

    if display:
        print(len(best_models), len(np.unique(best_models)))
        inds = []
        for ds_idx, n in enumerate(best_models):
            model_idx = names.index(n)
            print(ds_idx+1, model_idx, INDICATORS[ds_idx], n)
            inds.append(model_idx)

    return best_models, best_corrs


def plot_inference(ds, model, ind=slice(0, 4000, 1), name='[Name]',
                   covariates=False, high_resolution=False, train=True,
                   p=1, fixed=None, factor=7, neuron_idx=1,
                   kernel_scale=None, kernel_size=17):
    """
    Evaluate fluorescence data vs. true fluorescence and
    all provided neural network models on a single dataset, then plot it.

    Parameters
    ----------
    ds : int between 1 and 10
        Training dataset to load (and plot)
    model : Keras model
        Single decoder models to evaluate data with
    ind : Slice, optional, default=slice(0, 4000, 1)
        Range of indices to plot
    name : string
        Name of model
    high_resolution : bool, optional, default=False
        Should NNs process data at 25 Hz or at 100 Hz?
    covariates : bool, optional, default=False
        Should we append parameter estimates to NN model input?
    fixed : list of len == 2
        Pass in [sigma, gamma] or [sigma] instead of estimating parameters.
    factor : int, default 7
        Factor to downsample data by to compute noise scale.
    train : bool, default True
        Plot ground truth spikes and correlation coefficient on training data.
    """
    F, s = load_evaluation_data(ds, train)
    F = np.squeeze(F)
    corr_n_all = []

    Fc = F[str(neuron_idx)][np.isfinite(F[str(neuron_idx)])]
    Fc = np.reshape(Fc, (1, -1))

    # Perform spike inference on the current neuron
    out_n = _run_nn(Fc, model, high_resolution,
                    covariates, factor, fixed, p=p)

    # If desired, convolve all outputs with a specified kernel
    if kernel_scale is not None:
        nn = stat.norm(scale=kernel_scale)
        kernel = nn.pdf(np.linspace(-5, 5, kernel_size))
        out_n = np.convolve(out_n, kernel, 'same')
    out_n = downsample(out_n, SPIKEFINDER_DOWNSAMPLE)

    # Downsample the data for plotting
    F_down = downsample(np.squeeze(Fc), SPIKEFINDER_DOWNSAMPLE)

    # Compute correlations
    title = ('Performance on ' + INDICATORS[ds-1] +
             ' data (dataset ' + str(ds) + ', neuron ' +
             str(neuron_idx))
    if train:
        s_down = downsample(s[str(neuron_idx)], SPIKEFINDER_DOWNSAMPLE)
        corr_n = np.corrcoef(s_down[:len(out_n)], out_n)[0, 1]
        title = (title + ', corr n ' + str(corr_n) + ')')
        corr_n_all.append(corr_n)

    # Make a plot
    plt.figure(figsize=(8, 6))
    plt.plot(F_down[ind]*.5, linewidth=1)
    if train:
        plt.plot(s_down[ind]-3, label='True spikes')
    plt.plot(out_n[ind]/max(out_n[ind])*2-12, label='NN')
    plt.axis('off')
    plt.legend(loc='best', ncol=3)
    plt.title(title)
    return corr_n_all
