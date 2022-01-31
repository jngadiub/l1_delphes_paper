import os
import pickle
import argparse
import numpy as np
import tensorflow as tf
import plotting
import hls4ml
from qkeras import QConv2D, QDense, QActivation

from custom_layers import KLLoss, CustomMSE
from tensorflow.keras.models import Model
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
from tensorflow_model_optimization.sparsity.keras import strip_pruning

from hls4ml.model.profiling import compare
from hls4ml.model.optimizer import optimize_model

from models import (
    conv_vae,
    conv_ae
    )
from create_custom_model import flatten

import setGPU

hls4ml.model.optimizer.OutputRoundingSaturationMode.layers = ['Activation']
hls4ml.model.optimizer.OutputRoundingSaturationMode.rounding_mode = 'AP_RND_CONV'
hls4ml.model.optimizer.OutputRoundingSaturationMode.saturation_mode = 'AP_SAT'

def create_hls_config(model_type, quant_size, pruning, hardware, latent_dim,
    beta, input_file, output_config, output_folder):

    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    if beta==0: beta='0'
    # load keras model
    model = tf.keras.models.load_model(input_file,
      custom_objects={'PruneLowMagnitude': pruning_wrapper.PruneLowMagnitude,
      'QDense': QDense, 'QConv2D': QConv2D, 'QActivation': QActivation,
      'KLLoss': KLLoss, 'CustomMSE': CustomMSE})
    model.summary()

    # load dataset
    with open('output/data_5000.pickle', 'rb') as handle:
        x_train, y_train, x_test, y_test, all_bsm_data, pt_scaler = pickle.load(handle)

    # read model config
    config = hls4ml.utils.config_from_keras_model(
        model,
        default_precision='ap_fixed<16,6,AP_RND_CONV,AP_SAT>',
        max_bits=20,
        data_type_mode='auto_accum', # auto_accum_only
        granularity='name',
        test_inputs=np.concatenate((x_test,all_bsm_data[0],all_bsm_data[1]),axis=0))
    print(config)

    config['Model']['Strategy'] = 'Resource'
    config['LayerName']['q_conv2d'].update({
            'Strategy': 'Resource',
            'ParallelizationFactor': 10
        })
    config['LayerName']['q_conv2d_1'].update({
            'Strategy': 'Resource',
            'ParallelizationFactor': 10
        })

    if '_vae' in model_type:
        config['LayerName']['batch_normalization'] = {
                'Precision': 'ap_fixed<16, 6, AP_RND_CONV, AP_SAT>'
            }
        config['LayerName']['q_conv2d_1']['Precision'].update({
                'result': 'ap_fixed<18,9,AP_RND,AP_SAT>',
                'accum': 'ap_fixed<18,9,AP_RND,AP_SAT>'
            })
        config['LayerName']['latent_mu'].update({
                'Strategy': 'Latency',
            })
        config['LayerName']['latent_sigma'].update({
                'Strategy': 'Latency',
            })
        config['LayerName']['latent_mu']['Precision'].update({
                'result': 'ap_fixed<32,6>'
            })
        config['LayerName']['latent_sigma']['Precision'].update({
                'result': 'ap_fixed<32,6>'
            })
        config['LayerName']['latent_mu_linear'] = {
                'Precision': 'ap_fixed<32, 6, AP_RND_CONV, AP_SAT>'
            }
        config['LayerName']['latent_sigma_linear'] = {
                'Precision': 'ap_fixed<32, 6, AP_RND_CONV, AP_SAT>'
            }
        config['LayerName']['kl_loss'].update({
                'Precision': {
                    'accum': 'ap_fixed<32,10,AP_RND,AP_SAT>',
                    'result': 'ap_fixed<32,10>'
                },
                'exp_range': 0.5,
                'exp_table_t': 'ap_fixed<32,10,AP_RND,AP_SAT>',
                'table_size': 1024*4
            })

    if '_ae' in model_type:
        config['LayerName']['batch_normalization'] = {
                'Precision': 'ap_fixed<16, 6, AP_RND_CONV, AP_SAT>'
            }
        config['LayerName']['q_activation_1'] = {
                'Precision': 'ap_fixed<18, 6, AP_RND_CONV, AP_SAT>'
            }
        config['LayerName']['average_pooling2d'] = {
                'Precision': 'ap_fixed<18, 6, AP_RND_CONV, AP_SAT>'
            }
        config['LayerName']['q_conv2d_1']['Precision'].update({
                'result': 'ap_fixed<18,9,AP_RND,AP_SAT>',
                'accum': 'ap_fixed<18,9,AP_RND,AP_SAT>'
            })
        config['LayerName']['q_conv2d_1'].update({
                'Strategy': 'Resource',
                'ParallelizationFactor': 100
            })
        config['LayerName']['q_conv2d_3'].update({
                'Strategy': 'Resource',
                'ParallelizationFactor': 100
            })
        config['LayerName']['q_conv2d_2'].update({
                'Strategy': 'Resource',
                'ParallelizationFactor': 100
            })
        config['LayerName']['q_conv2d_4'].update({
                'Strategy': 'Resource',
                'ParallelizationFactor': 100
            })
        config['LayerName']['up_sampling2d'] = {
        'Precision': {
            'result': config['LayerName']['q_activation_3']['Precision']['result'].replace('>', ', AP_RND_CONV, AP_SAT>') if \
            type(config['LayerName']['q_activation_3']['Precision'])==dict\
            else config['LayerName']['q_activation_3']['Precision'].replace('>', ', AP_RND_CONV, AP_SAT>')
        }}
        config['LayerName']['up_sampling2d_1'] = {
            'Precision': {
            'result': config['LayerName']['q_activation_4']['Precision']['result'].replace('>', ', AP_RND_CONV, AP_SAT>') if \
            type(config['LayerName']['q_activation_4']['Precision'])==dict\
            else config['LayerName']['q_activation_4']['Precision'].replace('>', ', AP_RND_CONV, AP_SAT>')
            }
        }
        config['LayerName']['custom_mse']['Precision'].update({
            'result': 'ap_fixed<32, 16, AP_RND_CONV, AP_SAT>',
            'accum': 'ap_fixed<32, 16, AP_RND_CONV, AP_SAT>'
        })
    print(config)

    # save config
    with open(output_config, 'wb') as handle:
        pickle.dump(config, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', default='conv_vae',
        choices=['conv_vae', 'conv_ae'],
        help='Which model to use')
    parser.add_argument('--quant-size', default=0, type=int, help='Train quantized model with QKeras')
    parser.add_argument('--pruning', type=str, help='Train with pruning')
    parser.add_argument('--hardware', default='xcvu9p-flgb2104-2-e', type=str, help='Train with pruning')
    parser.add_argument('--latent-dim', type=int, required=True, help='Latent space dimension')
    parser.add_argument('--beta', type=float, default=1.0, help='Fraction of KL loss')
    parser.add_argument('--input-file', type=str, help='Where to save the model')
    parser.add_argument('--output-config', type=str, help='Where to save the config')
    parser.add_argument('--output-folder', type=str, help='Where to save the model')
    args = parser.parse_args()
    create_hls_config(**vars(args))
