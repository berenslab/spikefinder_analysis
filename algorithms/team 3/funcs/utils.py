import numpy as np
import theano.tensor as T
from theano import config

import sys
sys.path.append('../funcs/')
import layers as ll
import lasagne.nonlinearities as lnl
import lasagne

def rebin(arr, factor, method = 'sum'):
    '''Rebins a spiketrain by a given factor, shortens the array to enable division'''
    if np.ndim(arr) == 1: arr = np.expand_dims(arr, axis=0)

    arr = arr[:,:(arr.shape[1]//factor)*factor]
    if method == 'sum': return arr.reshape((arr.shape[0],arr.shape[1]//factor,factor)).sum(-1)
    else: return arr.reshape((arr.shape[0],arr.shape[1]//factor,factor)).mean(-1)

def softplus(x):
    return(np.log(1+np.exp(x)))

def arrayrize(tracelist,timebins):

    traces = [[] for x in range(len(tracelist))]

    for i in range(len(tracelist)):
        for j in range(len(tracelist[i])):
            traces[i].append(np.reshape(tracelist[i][j][:len(tracelist[i][j])//timebins * timebins],[-1,timebins]))
        traces[i] = np.vstack(traces[i]).astype(config.floatX)
    return traces

def set_rec_net(num_filters,filtsize,superres = 1,nonlin = lnl.LeakyRectify(0.2), AR = False, n_rnn_units = 128, n_features = 13):

    input_l = ll.InputLayer((None, None))
    rec_nn = ll.DimshuffleLayer(input_l, (0, 'x', 1))
    hevar = np.sqrt(np.sqrt(2/(1+0.2**2)))

    if nonlin == lnl.tanh:
        init = lasagne.init.GlorotUniform()
    else:
        init = lasagne.init.HeNormal(hevar)

    for num,size in zip(num_filters,filtsize):

        rec_nn = (ll.Conv1DLayer(rec_nn, num_filters = num, filter_size = size, stride =1, pad = 'same',
                                    nonlinearity = nonlin, name='conv1', W = init))

    if not AR:
        prob_nn = (ll.Conv1DLayer(rec_nn, num_filters = superres,filter_size = 11,stride =1, pad = 'same',
                          nonlinearity=lnl.sigmoid,name='conv2', b = lasagne.init.Constant(-3.)))
        prob_nn = ll.DimshuffleLayer(prob_nn,(0,2,1))
        prob_nn = ll.FlattenLayer(prob_nn)
    else:
        prob_nn = (ll.Conv1DLayer(rec_nn, num_filters = n_features,filter_size = 11,stride =1, pad = 'same',
                          nonlinearity=nonlin,name='conv2'))
        prob_nn = ll.DimshuffleLayer(prob_nn,(0,2,1))

    return {'network':prob_nn,'input':input_l, 'superres':superres, 'n_features':n_features, 'rnn_units':n_rnn_units}
