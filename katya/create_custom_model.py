import os
import argparse
import json
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
    )
from tensorflow.keras.layers import BatchNormalization, Reshape
from sklearn.model_selection import train_test_split
from models import (
    conv_vae, conv_ae
    )
from sklearn.preprocessing import StandardScaler
import tensorflow_model_optimization as tfmot
import pickle

from qkeras import QConv2D, QDense, QActivation

from tensorflow_model_optimization.sparsity.keras import strip_pruning
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
from custom_layers import KLLoss, Radius, CustomMSE
from models import model_set_weights
import hls4ml

import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

def mse_loss(inputs, outputs):
    return tf.math.reduce_mean(tf.math.square(outputs-inputs), axis=-1)

def make_mse(inputs, outputs):
    # remove last dimension
    inputs = tf.squeeze(inputs, axis=-1)
    inputs = tf.cast(inputs, dtype=tf.float32)
    # trick with phi
    outputs_phi = math.pi*tf.math.tanh(outputs)
    # trick with phi
    outputs_eta_egamma = 3.0*tf.math.tanh(outputs)
    outputs_eta_muons = 2.1*tf.math.tanh(outputs)
    outputs_eta_jets = 4.0*tf.math.tanh(outputs)
    outputs_eta = tf.concat([outputs[:,0:1,:,:], outputs_eta_egamma[:,1:5,:,:], outputs_eta_muons[:,5:9,:,:], outputs_eta_jets[:,9:19,:,:]], axis=1)
    # use both tricks
    outputs = tf.concat([outputs[:,:,0,:], outputs_eta[:,:,1,:], outputs_phi[:,:,2,:]], axis=2)
    # mask zero features
    mask = tf.math.not_equal(inputs,0)
    mask = tf.cast(mask, tf.float32)
    outputs = mask * outputs

    loss = mse_loss(tf.reshape(inputs, [-1, 57]), tf.reshape(outputs, [-1, 57]))
    # loss = tf.math.reduce_mean(loss, axis=0) # average over batch
    return loss

def flatten(loaded_model, model_type):
    if model_type=='conv_vae':
        flat_model = strip_pruning(loaded_model)
    if model_type=='conv_ae':
        stripped_encoder = strip_pruning(loaded_model.layers[1])
        stripped_decoder = strip_pruning(loaded_model.layers[2])
        flat_model = tf.keras.Sequential()
        for layer in stripped_encoder.layers:
            layer_config = layer.get_config()
            new_layer = layer.from_config(layer_config)
            flat_model.add(new_layer)
            new_layer.set_weights(layer.get_weights())
        for layer in stripped_decoder.layers[1:]: # We skip the 'input' layer of the decoder if encoder is there
            layer_config = layer.get_config()
            new_layer = layer.from_config(layer_config)
            flat_model.add(new_layer)
            new_layer.set_weights(layer.get_weights())    # Looks fine
        flat_model.summary()
    return flat_model

def create_custom_model(model_type, quant_size, pruning, latent_dim, beta,
    output_file, ptq):
    # # magic trick to make sure that Lambda function works
    # tf.compat.v1.disable_eager_execution()

    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    if beta==0: beta='0'
    with open(f'output/model-{model_type}-{latent_dim}-b{beta}-q{quant_size}-{pruning}.json', 'r') as jsonfile:
        config = jsonfile.read()
    model = tf.keras.models.model_from_json(config,
        custom_objects={'PruneLowMagnitude': pruning_wrapper.PruneLowMagnitude,
            'QDense': QDense, 'QConv2D': QConv2D, 'QActivation': QActivation})

    if ptq=='ptq':
        model = model_set_weights(model, f'output/model-{model_type}-{latent_dim}-b{beta}-q0-{pruning}', quant_size)
    else:
        model.load_weights(f'output/model-{model_type}-{latent_dim}-b{beta}-q{quant_size}-{pruning}.h5')

    model.summary()

    original_model = model

    if model_type=='conv_vae':
        # take only encoder
        model = model.layers[1]
        # get mu and sigma from model
        z_mean = model.layers[-3].output
        z_log_var = model.layers[-2].output

        # calculate KL distance with the custom layer
        custom_output = KLLoss()([z_mean, z_log_var])

        # create new model
        model = Model(inputs=model.input, outputs=custom_output)
        model = flatten(model, model_type)

    elif model_type=='conv_ae':
        model = flatten(model, model_type)
        # get the input and model prediction
        true = model.input
        predicted = model.layers[-1].output

        flat_input = Reshape((57,), name='flat_input')(true)
        scaled_input = BatchNormalization(trainable=False, name='scaled_input')(flat_input)

        # calculate MSE between them
        custom_output = CustomMSE(reshape=(19,3,1))([true, scaled_input, predicted])

        # create new model
        model = Model(inputs=model.input, outputs=custom_output)
        """ Keras BatchNorm layer returns
            gamma * (batch - self.moving_mean) / sqrt(self.moving_var + epsilon) + beta
            epsilon=0.001
            momentum=0.99
            moving_mean = moving_mean * momentum + mean(batch) * (1 - momentum)
            moving_variance = moving_var * momentum + var(batch) * (1 - momentum)

            pt_scaler
            pt_scaler.mean_
            pt_scaler.var_
        """
        with open('output/data_-1.pickle', 'rb') as f:
            x_train, y_train, x_test, y_test, all_bsm_data, pt_scaler = pickle.load(f)

        mean_ = np.zeros((57,))
        var_ = np.ones((57,))
        for i in range(19):
            mean_[3*i] = pt_scaler.mean_[i]
            var_[3*i] = pt_scaler.var_[i]
        # order of weights is (gamma,beta,mean,std)
        model.get_layer('scaled_input').set_weights((np.ones((57,)),np.zeros((57,)),mean_,var_))

    # save custom model
    model.summary()
    model.save(output_file)


    test = False
    if test:
        model = tf.keras.models.load_model(output_file,
              custom_objects={'PruneLowMagnitude': pruning_wrapper.PruneLowMagnitude,
              'QDense': QDense, 'QConv2D': QConv2D, 'QActivation': QActivation,
              'KLLoss': KLLoss, 'CustomMSE': CustomMSE})

        model.compile()

        custom_mse_output = model.predict(x_test[:5])
        keras_trace = hls4ml.model.profiling.get_ymodel_keras(model, x_test[:5])

        standard_output = original_model.predict(x_test[:5])
        original_mse_output = make_mse(y_test[:5], standard_output)
        import pdb; pdb.set_trace()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', default='conv_vae',
        choices=['conv_vae', 'conv_ae'],
        help='Which model to use')
    parser.add_argument('--quant-size', default=0, type=int, help='Train quantized model with QKeras')
    parser.add_argument('--pruning', type=str, help='Train with pruning')
    parser.add_argument('--latent-dim', type=int, required=True, help='Latent space dimension')
    parser.add_argument('--beta', type=float, default=1.0, help='Fraction of KL loss')
    parser.add_argument('--ptq', type=str, default='qat', help='Which type of quantization is used')
    parser.add_argument('--output-file', type=str, help='Where to save the model')
    args = parser.parse_args()
    create_custom_model(**vars(args))
