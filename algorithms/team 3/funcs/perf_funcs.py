import seaborn as sns
sns.set_style("white")

import numpy as np
import scipy.stats
import pickle
from scipy.stats import poisson

from utils import rebin
import math
import os

import collections

def last(x):
    return next(reversed(x.values()))

def total_best_exp(folder, sort, hi, prints = None, exp_list = None):

    the_dict = collections.OrderedDict([])

    subdirs = next(os.walk(folder))[1]
    for subdir in subdirs:
        dec = 3-len(subdir)
        if exp_list:
            if int(subdir[dec:]) not in exp_list: continue
        for _,_,file in os.walk(folder+subdir):
            for f in file:
                if 'dicts' in f:
                    with open(folder+subdir+'/'+f, 'rb') as f:
                        try: odict = pickle.load(f)
                        except EOFError: continue

                    if not isinstance(odict[sort],list):
                        if hi == 0: the_ind = min(odict[sort], key=odict[sort].get)
                        if hi == 1: the_ind = max(odict[sort], key=odict[sort].get)
                        the_dict[odict[sort][the_ind]] = f.name.split('/')[-2:] + [odict[a][the_ind] for a in prints]
                    else:
                        if hi == 0: the_ind = min(odict[sort][0], key=odict[sort][0].get)
                        if hi == 1: the_ind = max(odict[sort][0], key=odict[sort][0].get)
                        the_dict[odict[sort][0][the_ind]] = f.name.split('/')[-2:] + [odict[a][the_ind] for a in prints]

    if hi == 0: real_dict = collections.OrderedDict((key, the_dict[key]) for key in sorted(the_dict))
    if hi == 1: real_dict = collections.OrderedDict((key, the_dict[key]) for key in sorted(the_dict, reverse=True))
    return real_dict

def current_best_exp(folder, sort, hi, prints = None, exp_list = None):

    the_dict = collections.OrderedDict([])

    subdirs = next(os.walk(folder))[1]
    for subdir in subdirs:
        if exp_list:
            if int(subdir[-1]) not in exp_list: continue
        for _,_,file in os.walk(folder+subdir):
            for f in file:
                if 'dicts' in f:
                    with open(folder+subdir+'/'+f, 'rb') as f:
                        try: odict = pickle.load(f)
                        except EOFError: continue

                    the_dict[last(odict[sort])] = f.name.split('/')[-2:] + [last(odict[a]) for a in prints]
    if hi == 0: real_dict = collections.OrderedDict((key, the_dict[key]) for key in sorted(the_dict))
    if hi == 1: real_dict = collections.OrderedDict((key, the_dict[key]) for key in sorted(the_dict, reverse=True))
    return real_dict

def Corr_Sig(pred,truth):
    ccc = []
    for i in range(len(pred)):
        cc = np.corrcoef(pred[i],truth[i])[0,1]
        if not math.isnan(cc): ccc.append(cc)
    return np.mean(ccc),scipy.stats.sem(np.array(ccc))

def Corr(pred,truth):
    ccc = []
    for i in range(len(pred)):
        cc = np.corrcoef(pred[i],truth[i])[0,1]
        if not math.isnan(cc): ccc.append(cc)
    return np.mean(ccc)

def Loglike(pred, truth):
    loglike = []
    for k in range(len(pred)):
        loglike.append(np.mean(poisson.logpmf(truth[k], pred[k])))
    return np.mean(loglike)

def CrossEnt(pred, truth):
    
    truth = np.clip(truth,0,1)
    return -(truth * np.log(pred) + (1.0 - truth) * np.log(1.0 - pred))

def RMSE(pred, truth):

    return np.sqrt(np.sum((pred-truth)**2)/truth.sum())

def MSE(pred, truth):

    return (np.sum((pred-truth)**2)/truth.sum())

def RMSE_Sig(pred, truth):

    rrr = []
    for i in range(len(pred)):
        rr = np.sqrt(np.sum((pred[i]-truth[i])**2)/truth[i].sum())
        rrr.append(rr)
    return np.mean(rrr),scipy.stats.sem(np.array(rrr))

def MAE(pred, truth):

    return np.sum(np.abs(pred-truth))/truth.sum()


def binned_PM(func, facs):
    '''Takes a function and a list of factors and returns a new function for perf. measurements with different binnings'''
    def wrap_func(pred,truth, facs = facs):

        pms = []
        for f in facs:
            pms.append(func(rebin(pred,f),rebin(truth,f)))
        return pms

    return wrap_func

def eval_performance(model_or_reconstruction, spikes, traces = None, m_func = Corr, buffers = [100,100], facs = None, det = False):
    '''Evaluates performance for a model/prediction and a given performance measurement.
    Works on 3d arrays (N_cells, N_traces, N_t)'''

    if traces is None:

        preds = model_or_reconstruction

        if np.ndim(preds) == 3 or isinstance(preds,list):

            preds = np.vstack(preds)
            spikes = np.vstack(spikes)

    else:

        if np.ndim(traces) == 3 or isinstance(traces,list):

            traces = np.vstack(traces)
            spikes = np.vstack(spikes)

        if not det: preds = model_or_reconstruction.mrec.recfunc(traces)
        else:       preds = model_or_reconstruction.mrec.detfunc(traces)

    if facs:
        eval_func = binned_PM(m_func,facs)
    else:
        eval_func = m_func

    return eval_func(preds[:,buffers[0]:-buffers[-1]], spikes[:,buffers[0]:-buffers[-1]])

def eval_all(pred_prob, spikes, buffers, facs, det = True):

    RMSEs = eval_performance(pred_prob, spikes, traces = None, m_func = RMSE, buffers = buffers, facs = facs, det = det)
    Corrs = eval_performance(pred_prob, spikes, traces = None, m_func = Corr, buffers = buffers, facs = facs, det = det)
    # LogLs = eval_performance(pred_prob, spikes, traces = None, m_func = CrossEnt, buffers = buffers, facs = facs, det = det)

    if np.ndim(pred_prob) == 3 or isinstance(pred_prob, list):
        pred_prob = np.vstack(pred_prob)
        spikes = np.vstack(spikes)

    Factor = pred_prob[:,buffers[0]:-buffers[-1]].sum(-1)/spikes[:,buffers[0]:-buffers[-1]].sum(-1)
    Factor = np.mean(Factor[Factor<1e8])

    return RMSEs, Corrs, Factor
