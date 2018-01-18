import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential, Model
from keras.layers.wrappers import Bidirectional
from keras.layers.core import Masking
from keras.layers.merge import Concatenate
from keras.layers import Dense, Activation, Dropout, Input, LSTM
from keras.layers.convolutional import Conv1D
from keras.callbacks import TensorBoard

from keras import backend as K
import tensorflow as tf



def pearson_corr(y_true, y_pred, pool=True):
    """
        Calculates Pearson correlation as a metric.
        This calculates Pearson correlation the way that 
        the competition calculates it (as integer values).
        y_true and y_pred have shape (batch_size, num_timesteps, 1).

    """
    if pool:
        y_true = pool1d(y_true, length=4)
        y_pred = pool1d(y_pred, length=4)

    mask = tf.to_float(y_true>=0.)
    samples = K.sum(mask,axis=1, keepdims=True)
    x_mean = y_true - K.sum(mask*y_true, axis=1, keepdims=True)/samples
    y_mean = y_pred - K.sum(mask*y_pred, axis=1, keepdims=True)/samples

    # Numerator and denominator.
    n = K.sum(x_mean * y_mean * mask, axis=1)
    d = (K.sum(K.square(x_mean) * mask, axis=1) *
         K.sum(K.square(y_mean) * mask, axis=1))

    return 1. - K.mean(n/(K.sqrt(d) + 1e-12))


def pool1d(x, length=4):
    """
    Adds groups of `length` over the time dimension in x.
    Args:
        x: 3D Tensor with shape (batch_size, time_dim, feature_dim).
        length: the pool length.
    Returns:
        3D Tensor with shape (batch_size, time_dim // length,
            feature_dim).

    """
    x = tf.expand_dims(x, -1)  # Add "channel" dimension.
    avg_pool = tf.nn.avg_pool(x,
        ksize=(1, length, 1, 1),
        strides=(1, length, 1, 1),
        padding='SAME')
    x = tf.squeeze(avg_pool, axis=-1)

    return x * length


def load_data(load_test=True, dataloc = '../../data/raw/truth/'):
    calcium_train = []
    spikes_train = []
    ids = []
    calcium_test = []
    ids_test = []
    for dataset in range(10):
        calcium_train.append(np.array(pd.read_csv(dataloc 
             + str(dataset+1) + 
            '.train.calcium.csv')))
        spikes_train.append(np.array(pd.read_csv(dataloc
            + str(dataset+1) + 
            '.train.spikes.csv')))
        ids.append(np.array([dataset]*calcium_train[-1].shape[1]))
        if load_test and dataset < 5:
            calcium_test.append(np.array(pd.read_csv(dataloc +
                str(dataset+1) +
                '.test.calcium.csv')))
            ids_test.append(np.array([dataset]*calcium_test[-1].shape[1]))

    maxlen = max([c.shape[0] for c in calcium_train])
    maxlen_test = max([c.shape[0] for c in calcium_test])
    calcium_train_padded = \
        np.hstack([np.pad(c, ((0, maxlen-c.shape[0]), (0, 0)),
            'constant', constant_values=np.nan) for c in calcium_train])
    spikes_train_padded = \
        np.hstack([np.pad(c, ((0, maxlen-c.shape[0]), (0, 0)),
            'constant', constant_values=np.nan) for c in spikes_train])
    calcium_test_padded = \
        np.hstack([np.pad(c, ((0, maxlen_test-c.shape[0]), (0, 0)),
        'constant', constant_values=np.nan) for c in calcium_test])
    ids_stacked = np.hstack(ids)
    if load_test:
        ids_test_stacked = np.hstack(ids_test)
    else:
        ids_test_stacked = []
    sample_weight = 1. + 1.5*(ids_stacked<5)
    sample_weight /= sample_weight.mean()
    calcium_train_padded[spikes_train_padded<-1] = np.nan
    spikes_train_padded[spikes_train_padded<-1] = np.nan

    calcium_train_padded[np.isnan(calcium_train_padded)] = 0.
    spikes_train_padded[np.isnan(spikes_train_padded)] = -1.

    calcium_train_padded = calcium_train_padded.T[:, :, np.newaxis]
    spikes_train_padded = spikes_train_padded.T[:, :, np.newaxis]
    calcium_test_padded = calcium_test_padded.T[:, :, np.newaxis]

    ids_oneshot = np.zeros((calcium_train_padded.shape[0],
        calcium_train_padded.shape[1], 10))
    ids_oneshot_test = np.zeros((calcium_test_padded.shape[0],
        calcium_test_padded.shape[1], 10))
    for n,i in enumerate(ids_stacked):
        ids_oneshot[n, :, i] = 1.
    for n,i in enumerate(ids_test_stacked):
        ids_oneshot_test[n, :, i] = 1.

    return calcium_train, calcium_train_padded, spikes_train_padded,\
            calcium_test, calcium_test_padded, ids_oneshot, ids_oneshot_test,\
            ids_stacked, ids_test_stacked, sample_weight
            

def create_model():
    '''
        Creates a multilayered convolutional network
        with a LSTM in between the layers.

    '''
    main_input = Input(shape=(None,1), name='main_input')
    dataset_input = Input(shape=(None,10), name='dataset_input')
    x = Conv1D(10, 300, padding='same', input_shape=(None,1))(main_input)
    x = Activation('tanh')(x)
    x = Dropout(0.3)(x)
    x = Conv1D(10, 10, padding='same')(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)
    x = Concatenate()([x, dataset_input])
    x = Conv1D(10, 5, padding='same')(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    z = Bidirectional(LSTM(10, return_sequences=True),
                merge_mode='concat', weights=None)(x)
    x = Concatenate()([x, z])

    x = Conv1D(8, 5, padding='same')(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)
    x = Conv1D(4, 5, padding='same')(x)
    x = Activation('relu')(x)
    x = Conv1D(2, 5, padding='same')(x)
    x = Activation('relu')(x)
    x = Conv1D(2, 5, padding='same')(x)
    x = Activation('relu')(x)
    x = Conv1D(1, 5, padding='same')(x)
    output = Activation('sigmoid')(x)

    model = Model(inputs=[main_input, dataset_input], outputs=output)
    model.compile(loss=pearson_corr, optimizer='adam')

    return model


def model_fit(model):
    tbCallBack = TensorBoard(log_dir='./logtest2', histogram_freq=0,
            write_graph=True, write_images=True)

    model.fit([calcium_train_padded, ids_oneshot], 
        spikes_train_padded, epochs=1,
        batch_size=5, validation_split=0.2, sample_weight=sample_weight,
        callbacks=[tbCallBack])
    model.save_weights('model_convi_6')
    return model


def model_test(model):
    #model.compile(loss=pearson_corr,optimizer='adam')
    model.load_weights('model_convnet_6')
    pred_train = model.predict([calcium_train_padded, ids_oneshot])
    pred_test = model.predict([calcium_test_padded, ids_oneshot_test])

    for dataset in range(10):
        pd.DataFrame(pred_train[ids_stacked == dataset,
            :calcium_train[dataset].shape[0]].squeeze().T).\
            to_csv(dataloc + 'predict_6/' + str(dataset+1)+\
            '.train.spikes.csv', sep=',', index=False)
        if dataset < 5:
            pd.DataFrame(pred_test[ids_test_stacked == dataset,
                :calcium_test[dataset].shape[0]].squeeze().T).\
                to_csv(dataloc + 'predict_6/' + str(dataset+1)+\
                '.test.spikes.csv', sep=',', index=False)


def plot_kernels(model, layer=0):
    srate = 100.
    weights = model.get_weights()[layer]
    t = np.arange(-weights.shape[0]/srate/2,
                weights.shape[0]/srate/2, 1./srate)
    for j in range(weights.shape[2]):
        plt.plot(t, weights[:, 0, j] + .3*j)

    plt.xlabel('Time [s]')
    plt.ylabel('Kernel amplitudes')
    plt.title('Convolutional kernels of the input layer')
    plt.show()
