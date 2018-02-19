import sys
sys.path.append('../lib/')
sys.path.append('../funcs/')

import theano
import lasagne
from theano import tensor as T
import theano.tensor.nlinalg as Tla
import numpy as np
from theano import config

import layers as ll
from layers import recurrent as rec

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
srng = RandomStreams(seed=np.random.randint(10e6))

def clipped_binary_crossentropy(predictions, targets):

    targets = T.clip(targets,0,1)
    return theano.tensor.nnet.binary_crossentropy(predictions, targets)

class RecognitionModel(object):
    '''
    Recognition Model Interace Class

    Recognition model approximates the posterior given some observations

    Different forms of recognition models will have this interface

    The constructor must take the Input Theano variable and create the
    appropriate sampling expression.
    '''

    def __init__(self,Input, rng, batchSize):

        self.Input = Input
        self.batchSize = batchSize
        self.rng = rng
        self.superres = 1

class GRU_FB_BernoulliRecognition(RecognitionModel):

    def __init__(self,RecognitionParams,Input, rng, n_samples = 1):
        '''
        h = Q_phi(z|x), where phi are parameters, z is our latent class, and x are data
        '''
        super().__init__(Input, rng, n_samples)
        self.n_units = RecognitionParams['rnn_units']
        self.n_convfeatures = RecognitionParams['n_features']
        
        self.conv_back = RecognitionParams['network']
       
        conv_cell = RecognitionParams['network']
        conv_cell = ll.DimshuffleLayer(conv_cell, (1,0,2))
        self.conv_cell = ll.get_output(conv_cell, inputs = self.Input)

        inp_cell = RecognitionParams['input']
        inp_cell = ll.DimshuffleLayer(inp_cell, (1,0,'x'))
        self.inp_cell = ll.get_output(inp_cell, inputs = self.Input)

        inp_back = RecognitionParams['input']
        inp_back = ll.DimshuffleLayer(inp_back, (0,1,'x'))
        inp_back = ll.ConcatLayer([self.conv_back,inp_back],axis=2)
        
        cell_inp = ll.InputLayer((None, self.n_convfeatures + self.n_units + 1 + 1 + 1))
        self.cell = rec.GRUCell(cell_inp, self.n_units,grad_clipping=100.)
        self.p_out = ll.DenseLayer((None,self.n_units+self.n_convfeatures),1,nonlinearity=lasagne.nonlinearities.sigmoid,b = lasagne.init.Constant(-3.))

        hid_0 = T.zeros([self.Input.shape[0],self.n_units])
        samp_0 = T.zeros([self.Input.shape[0],1])

        self.back_nn = rec.GRULayer(inp_back,self.n_units,backwards = True)
        self.back_nn = ll.DimshuffleLayer(self.back_nn, (1,0,2))
        self.backward = ll.get_output(self.back_nn, inputs = self.Input)

        def sampleStep(conv_cell, inp_cell, back , hid_tm1, samp_tm1, prob_tm1):

            cell_in = T.concatenate([conv_cell,inp_cell,back,samp_tm1,prob_tm1],axis=1)
            rnn_t = self.cell.get_output_for({'input':cell_in, 'output':hid_tm1})
            prob_in = T.concatenate([conv_cell,rnn_t['output']],axis=1)
            prob_t = self.p_out.get_output_for(prob_in)
            samp_t = srng.binomial(prob_t.shape, n=1, p = prob_t, dtype=theano.config.floatX) 

            return rnn_t['output'], samp_t, prob_t

        ((rnn_temp,s_t, p_t), updates) =\
            theano.scan(fn=sampleStep,
                        sequences=[self.conv_cell,self.inp_cell, self.backward],
#                         outputs_info=[T.unbroadcast(hid_0,1), T.unbroadcast(samp_0,1), T.unbroadcast(samp_0,1)])
                        outputs_info=[hid_0, samp_0, samp_0])

        for k, v in updates.items():
            k.default_update = v
            
        self.recfunc = theano.function([self.Input], outputs= p_t[:,:,0].T, updates = updates)
        self.samplefunc = theano.function([self.Input], outputs= s_t[:,:,0].T, updates = updates)
        self.dualfunc = theano.function([self.Input], outputs=[p_t[:,:,0].T,s_t[:,:,0].T], updates = updates)
        self.detfunc = self.recfunc      

    def getParams(self):
        network_params = ll.get_all_params(self.conv_back, trainable = True)
        for p in ll.get_all_params(self.back_nn):
            network_params.append(p)
        for p in ll.get_all_params(self.cell):
            network_params.append(p)
        for p in ll.get_all_params(self.p_out):
            network_params.append(p)
        return network_params

    def getSample(self, X, n_samples = 1, rebin = False, det = False):

        X = np.repeat(X,n_samples, axis = 0)
        sample = self.samplefunc(X)
        return sample

    def evalLogDensity(self, hsamp, buffers = [0,1], forEval = False):

        for b in buffers: b *= self.superres
        batchSize = self.batchSize
        n_samples = hsamp.shape[0]//batchSize
        if forEval:
            batchSize = hsamp.shape[0]
            n_samples = 1
            
        conv_cell = T.repeat(self.conv_cell, n_samples, axis = 1)
        inp_cell = T.repeat(self.inp_cell, n_samples, axis = 1)
        backward = T.repeat(self.backward, n_samples, axis = 1)

        samp_inp = T.concatenate((T.zeros(hsamp[:,:1].shape),hsamp[:,:-1]),axis=1)
        samp_inp = T.transpose(samp_inp)
        samp_inp = T.reshape(samp_inp,[samp_inp.shape[0],samp_inp.shape[1],1])
        
        hid_0 = T.zeros([hsamp.shape[0],self.n_units])
        prob_0 = T.zeros([hsamp.shape[0],1])

        def evalStep(conv_cell, inp_cell, back, samp_tm1 , hid_tm1, prob_tm1):

            cell_in = T.concatenate([conv_cell,inp_cell,back,samp_tm1,prob_tm1],axis=1)
            rnn_t = self.cell.get_output_for({'input':cell_in, 'output':hid_tm1})
            prob_in = T.concatenate([conv_cell,rnn_t['output']],axis=1)
            prob_t = self.p_out.get_output_for(prob_in)            
            return rnn_t['output'], prob_t
        
        ((rnn_temp, prob_t), updates) =\
            theano.scan(fn=evalStep,
                        sequences=[conv_cell,inp_cell, backward, samp_inp],
#                         outputs_info=[T.unbroadcast(hid_0,1), T.unbroadcast(prob_0,1)])
                        outputs_info=[hid_0, prob_0])
            
        for k, v in updates.items():
            k.default_update = v

        prob = prob_t[:,:,0].T
        prob = T.clip(prob, 0.001, 0.999)
        prob = prob.reshape((batchSize, n_samples, -1))
        hsamp = hsamp.reshape((batchSize, n_samples, -1))
        return -clipped_binary_crossentropy(prob, hsamp)[:,:,buffers[0]:-buffers[1]].sum(axis=-1)

class BernoulliRecognition(RecognitionModel):

    def __init__(self,RecognitionParams,Input, rng, n_samples = 1):
        '''
        h = Q_phi(z|x), where phi are parameters, z is our latent class, and x are data
        '''
        super().__init__(Input, rng, n_samples)
        self.NN = RecognitionParams['network']

        self.p = ll.get_output(self.NN, inputs = self.Input)
        self.p_det = ll.get_output(self.NN, inputs = self.Input, deterministic = True)
        sample = srng.binomial(self.p.shape, n=1, p = self.p, dtype=theano.config.floatX)

        self.recfunc = theano.function([self.Input], self.p)
        self.detfunc = theano.function([self.Input], self.p_det)
        self.dualfunc = theano.function([self.Input], outputs=[self.p, sample])
        
    def getParams(self):
        network_params = ll.get_all_params(self.NN, trainable = True)
        return network_params

    def getSample(self, X, n_samples = 1, rebin = True, det = False):

        pi = self.recfunc(X)
        pi = np.repeat(pi, n_samples, axis = 0)
        sample = self.rng.binomial(n=1, p = pi).astype(config.floatX)
        return sample


    def evalLogDensity(self, hsamp, buffers = [0,1], forEval = False):

        for b in buffers: b *= self.superres
        batchSize = self.batchSize
        n_samples = hsamp.shape[0]//batchSize
        if forEval:
            batchSize = hsamp.shape[0]
            n_samples = 1
        
        prob = T.clip(self.p, 0.001, 0.999)
        prob = prob.dimshuffle(0, 'x', 1)
        hsamp = hsamp.reshape((batchSize, n_samples, -1))
        return -clipped_binary_crossentropy(prob, hsamp)[:,:,buffers[0]:-buffers[1]].sum(axis=-1)