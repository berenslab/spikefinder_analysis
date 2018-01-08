"""
Created on Fri Apr  1 13:04:14 2016

@author: artur
"""

from RecognitionModel import *
import collections
import pickle

import time
# import psutil
import os

from data_funcs import *
from perf_funcs import *


class TrainingAlgo(object):
    def __init__(self,
                 rec_params,  # dictionary of approximate posterior ("recognition model") parameters
                 REC_MODEL,  # class that inherits from RecognitionModel
                 batchSize,
                 n_samples,
                 filename,
                 rng,
                 use_patience):

        # ---------------------------------------------------------
        ## actual model parameters
        self.X = T.matrix('X')  # symbolic variable for the data
        self.Z = T.matrix('Z')  # symbolic variable for true spikes

        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.RandomState()

        self.batchSize = batchSize
        self.n_samples = n_samples
        self.buffers = [1, 1]
        self.lr_decay = 0.9995
        self._filename = filename
        self._use_patience = use_patience
        self.exp_params = None
        self.superres = 1
        self.resample = 60
        self.description = None
        
        if self._use_patience:
            self._patience = 100000
            self._patience_increase = 2
            self._improvement_threshold = 0.995

        # instantiate our recognition model
        self.mrec = REC_MODEL(rec_params, self.X, self.rng, self.batchSize)
        if 'superres' in rec_params.keys(): self.superres = rec_params['superres']; self.mrec.superres = rec_params['superres']

        self.facs = [1]
        self.eval_T = 1000
        self.eval_rep = 1
        self._iter_count = 0

        self.cost_hist = collections.OrderedDict([])

        self.corr_train = [collections.OrderedDict([]) for _ in range(len(self.facs))]
        self.rmse_train = [collections.OrderedDict([]) for _ in range(len(self.facs))]
        self.corr_ave_train = collections.OrderedDict([])
        self.rmse_ave_train = collections.OrderedDict([])
        self.corr_base = collections.OrderedDict([])

        self.corr_test = [collections.OrderedDict([]) for _ in range(len(self.facs))]
        self.rmse_test = [collections.OrderedDict([]) for _ in range(len(self.facs))]
        self.corr_ave_test = collections.OrderedDict([])
        self.rmse_ave_test = collections.OrderedDict([])

        self.factor = collections.OrderedDict([])

        self.mrec_params = []

        self.update_time = collections.OrderedDict([])
        self.eval_time = collections.OrderedDict([])

        self.best_validation_corr = -np.inf

    def save_object(self, filename):
        ''' Save object to file filename '''
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def update_patience(self, validation_corr):

        if validation_corr:
            if validation_corr > self.best_validation_corr:
                if validation_corr > self.best_validation_corr * self._improvement_threshold:
                    self._patience = max(self._patience, self._iter_count * self._patience_increase)
                self.best_validation_corr = validation_corr
                if self._filename:
                    self.save_object(self._filename + '.pkl')

    def getParams(self, cell=0):
        '''
        Return Recognition Model parameters that are currently being trained.
        '''
        return self.mrec.getParams()

    def set_dataset(self, train_traces, train_spikes, test_traces, test_spikes):

        self.train_spikes = train_spikes
        self.train_traces = train_traces
        self.test_spikes = test_spikes
        self.test_traces = test_traces

    def fit(self, trainf, trains, max_epochs=100, learning_rate=1e-3, print_output=True, print_freq=1,
            stream_traces=False):
        
        epoch = 0
        self.print_freq = print_freq

        self.lr = theano.shared(np.array(learning_rate, dtype=theano.config.floatX), name='lr')
        self.lr_decay = np.array(self.lr_decay, dtype=theano.config.floatX)
        self.sv_inds = []

        param_updaters = self.update_params()

        ''' TRAINING '''

        while epoch < max_epochs and (self._patience > self._iter_count):

            if epoch > 5 and np.abs(list(self.factor.values())[-1] - list(self.factor.values())[-2]) < 1e-5 and np.abs(
                            list(self.factor.values())[-2] - list(self.factor.values())[-3]) < 1e-5:
                print('Training Stuck')
                break

            t0 = time.time()
            self.lr.set_value(self.lr.get_value() * self.lr_decay)
            
            Iterator = DatasetMiniBatchIterator(self, trainf, trains)

            batches, cells, z_true, indices = zip(*Iterator)
            tot_cost = 0; bl_cost = 0

            for x, c, z, inds in zip(batches, cells, z_true, indices):

                self._iter_count += 1
                tot_cost += param_updaters(x, z, self.lr.get_value())

            self.cost_hist[self._iter_count] = tot_cost / self.print_freq
            updatetime = 1000 * (time.time() - t0) / (self.print_freq)

            ''' EVALUATION '''

            t0 = time.time()

            if self.test_traces is not None:

                pred_prob, pred_sample = self.mrec.dualfunc(
                    np.repeat(np.vstack(self.test_traces)[:,:self.eval_T], self.eval_rep, axis=0))
                pred_prob = np.mean(pred_prob.reshape([-1, self.eval_rep, self.eval_T*self.superres]), axis=1)

                RMSEs, Corrs, Factor = eval_all(pred_prob, np.vstack(self.test_spikes)[:,:self.eval_T*self.superres], self.buffers,self.facs)
                for i in range(len(self.facs)):
                    self.rmse_test[i][self._iter_count] = RMSEs[i]
                    self.corr_test[i][self._iter_count] = Corrs[i]

                self.rmse_ave_test[self._iter_count] = np.mean(RMSEs)
                self.corr_ave_test[self._iter_count] = np.mean(Corrs)
                if self.train_traces is None: self.factor[self._iter_count] = Factor

            if self.train_traces is not None:

                pred_prob, pred_sample = self.mrec.dualfunc(np.repeat(np.vstack(self.train_traces)[:,:self.eval_T], self.eval_rep, axis=0))
                    
                pred_prob = np.mean(pred_prob.reshape([-1, self.eval_rep, self.eval_T*self.superres]), axis=1)
                RMSEs, Corrs, Factor = eval_all(pred_prob, np.vstack(self.train_spikes)[:,:self.eval_T*self.superres], self.buffers,self.facs)

                for i in range(len(self.facs)):
                    self.rmse_train[i][self._iter_count] = RMSEs[i]
                    self.corr_train[i][self._iter_count] = Corrs[i]

                self.rmse_ave_train[self._iter_count] = np.mean(RMSEs)
                self.corr_ave_train[self._iter_count] = np.mean(Corrs)
                self.factor[self._iter_count] = Factor

                if self._use_patience:
                    self.update_patience(self.corr_ave_train[self._iter_count])

            evaltime = time.time() - t0

            if print_output:
                print('{}{:0.3f}{}{:0.3f} {}{}{:0.1f}{}{:0.1f}{} {}{}{}' \
                      .format('Corr. Val./Train: ', float(self.corr_train[0][self._iter_count]), '/',
                              float(self.corr_test[0][self._iter_count]),
                              ' || ', 'Time upd./Eval.:  ', float(updatetime), ' ms ', float(evaltime), ' s',
                              ' || ', 'BatchNr.:  ', self._iter_count))

            if self._filename:

                self.save_object(self._filename + '_curr.pkl')
                col_dict = {
                    'cost_hist': self.cost_hist,
                    'corr_test': self.corr_test,
                    'rmse_test': self.rmse_test,
                    'corr_ave_test': self.corr_ave_test,
                    'rmse_ave_test': self.rmse_ave_test,
                    'corr_train': self.corr_train,
                    'rmse_train': self.rmse_train,
                    'corr_ave_train': self.corr_ave_train,
                    'rmse_ave_train': self.rmse_ave_train,
                    'factor': self.factor,
                    'update_time': self.update_time,
                    'eval_time': self.eval_time,
                    'exp_params': self.exp_params,
                    'description': self.description
                }
                with open(self._filename + '_dicts.pkl', 'wb') as f:
                    pickle.dump(col_dict, f)

            evaltime = time.time() - t0
            self.update_time[self._iter_count] = updatetime
            self.eval_time[self._iter_count] = evaltime
            epoch += 1
