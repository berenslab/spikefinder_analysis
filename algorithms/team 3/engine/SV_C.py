
from TrainingAlgos import *
from data_funcs import *

class SV(TrainingAlgo):

    def __init__(self,
                rec_params, # dictionary of approximate posterior ("recognition model") parameters
                REC_MODEL, # class that inherits from RecognitionModel
                batchSize = 20,
                n_samples = 1,
                filename = None,
                rng = None,
                use_patience = True):

        super().__init__(rec_params,REC_MODEL,batchSize, n_samples, filename, rng, use_patience)
        self.algo = 'Supervised'

    def update_params(self):

        lr = T.scalar('lr')
        LL = T.mean(self.mrec.evalLogDensity(self.Z, self.buffers))

        grads = T.grad(-LL, wrt = self.mrec.getParams())
        updates = lasagne.updates.adam(grads, self.mrec.getParams(), lr)

        perform_updates_params = theano.function(
            inputs= [self.X,self.Z,lr],
            on_unused_input = 'ignore',
            updates=updates,
            outputs=LL)

        return perform_updates_params
