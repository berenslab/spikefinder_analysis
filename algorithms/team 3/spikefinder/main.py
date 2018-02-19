from scipy import corrcoef
from scipy.stats import spearmanr
from numpy import percentile, asarray, arange, zeros, where, repeat, sort, cov, mean, std, ceil
from numpy import vstack, hstack, argmin, ones, convolve, log, linspace, min, max, square, sum, diff
from numpy import array, eye, dot, empty, seterr, isnan, any, zeros_like
from scipy.optimize import minimize
from scipy.stats import poisson
from scipy.interpolate import interp1d
from pandas import read_csv
from warnings import warn

def load(file):
    """
    Load results from a file or string.
    """
    return read_csv(file)
        
def score(a, b, method='corr', downsample=4):
    """
    Estimate similarity score between two reslts.
    """
    methods = {
      'loglik': _loglik, 
      'info': _info, 
      'corr': _corr, 
      'auc': _auc, 
      'rank': _rank
    }
    if method not in methods.keys():
      raise Exception('scoring method not one of: %s' % ' '.join(methods.keys()))

    func = methods[method]

    result = []
    for column in a:
        x = a[column]
        y = b[column]
        naninds = isnan(x) | isnan(y)
        x = x[~naninds]
        y = y[~naninds]
        x = _downsample(x, downsample)
        y = _downsample(y, downsample)
        if not len(x) == len(y):
            raise Exception('mismatched lengths %s and %s' % (len(x), len(y)))
        result.append(func(x, y))
    return result

def _corr(x, y):
    return corrcoef(x, y)[0,1]

def _info(x, y):
    # this is absolute info gain, we report mean(info)/mean(entropy) across all cells in the paper
    loglik, info = _infolik(x, y)
    return info

def _loglik(x, y):
    loglik, info = _infolik(x, y)
    return loglik

def _rank(x, y):
    return spearmanr(x, y).correlation        

def _auc(x, y):
     pass

def _downsample(signal, factor):
    """
    Downsample signal by averaging neighboring values.
    @type  signal: array_like
    @param signal: one-dimensional signal to be downsampled
    @type  factor: int
    @param factor: this many neighboring values are averaged
    @rtype: ndarray
    @return: downsampled signal
    """

    if factor < 2:
        return asarray(signal)
        
    return convolve(asarray(signal).ravel(), ones(factor), 'valid')[::factor]
    
def _infolik(spikes, predictions, fps=25):
    """
    Computes log likelihood of the data and the information gain
    
    adapted from lucas theis, c2s package
    https://github.com/lucastheis/c2s   
    """
    factor = 1   ### DITO
    
    # find optimal point-wise monotonic function
    f = optimize_predictions(predictions, spikes, num_support=10, regularize=5e-8, verbosity=2)

    # for conversion into bit/s
    factor = 1. / factor / log(2.)     

    # average firing rate (Hz) over all cells
    firing_rate = mean(spikes) * fps;

    # estimate log-likelihood and marginal entropies
    loglik = mean(poisson.logpmf(spikes, f(predictions))) * fps * factor
    entropy = -mean(poisson.logpmf(spikes, firing_rate / fps)) * fps * factor

    return loglik, loglik + entropy
    
def optimize_predictions(predictions, spikes, num_support=10, regularize=5e-8, verbosity=1):
    """
    Fits a monotonic piecewise linear function to maximize the Poisson likelihood of
    firing rate predictions interpreted as Poisson rate parameter.
    @type  predictions: array_like
    @param predictions: predicted firing rates
    @type  spikes: array_like
    @param spikes: true spike counts
    @type  num_support: int
    @param num_support: number of support points of the piecewise linear function
    @type  regularize: float
    @param regularize: strength of regularization for smoothness
    @rtype: interp1d
    @return: a piecewise monotonic function
    
    adapted from lucas theis, c2s package
    https://github.com/lucastheis/c2s
    """

    if num_support < 2:
        raise ValueError('`num_support` should be at least 2.')

    if any(predictions < 0.):
        warn('Some firing rate predictions are smaller than zero.')
        predictions[predictions < 0.] = 0.

    if any(isnan(predictions)):
        warn('Some predictions are NaN.')
        predictions[isnan(predictions)] = 0.

    # support points of piece-wise linear function
    if num_support > 2:
        F = predictions
        if F.sum() > 0:
            F = F[F > (max(F) - min(F)) / 100.]
        x = list(percentile(F, range(0, 101, num_support)[1:-1]))
        x = asarray([0.] + x + [max(F)])

        for i in range(len(x) - 1):
            if x[i + 1] - x[i] < 1e-6:
                x[i + 1] = x[i] + 1e-6
    else:
        x = asarray([min(predictions), max(predictions)])

        if x[1] - x[0] < 1e-6:
                x[1] += 1e-6

    def objf(y):
        # construct piece-wise linear function
        f = interp1d(x, y)

        # compute predicted firing rates
        l = f(predictions) + 1e-8

        # compute negative log-likelihood (ignoring constants)
        K = mean(l - spikes * log(l))

        # regularize curvature
        z = (x[2:] - x[:-2]) / 2.
        K = K + regularize * sum(square(diff(diff(y) / diff(x)) / z))

        return K

    class MonotonicityConstraint:
        def __init__(self, i):
            self.i = i

        def __call__(self, y):
            return y[self.i] - y[self.i - 1]

    # monotonicity and non-negativity constraint
    constraints = [{'type': 'ineq', 'fun': MonotonicityConstraint(i)} for i in range(1, x.size)]
    constraints.extend([{'type': 'ineq', 'fun': lambda y: y[0]}])

    # fit monotonic function
    settings = seterr(invalid='ignore')
    res = minimize(
		fun=objf,
		x0=x + 1e-6,
		method='SLSQP',
		tol=1e-9,
		constraints=constraints,
		options={'disp': 0, 'iprint': verbosity})
    seterr(invalid=settings['invalid'])

    # construct monotonic piecewise linear function
    return interp1d(x, res.x, bounds_error=False, fill_value=res.x[-1])