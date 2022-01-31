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
    output_file):
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
    model = model_set_weights(model, f'output/model-{model_type}-{latent_dim}-b{beta}-q0-{pruning}')
    model.summary()
    model.save(output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', default='conv_vae',
        choices=['conv_vae', 'conv_ae'],
        help='Which model to use')
    parser.add_argument('--quant-size', default=0, type=int, help='Train quantized model with QKeras')
    parser.add_argument('--pruning', type=str, help='Train with pruning')
    parser.add_argument('--latent-dim', type=int, required=True, help='Latent space dimension')
    parser.add_argument('--beta', type=float, default=1.0, help='Fraction of KL loss')
    parser.add_argument('--output-file', type=str, help='Where to save the model')
    args = parser.parse_args()
    create_custom_model(**vars(args))
