import h5py
import pickle
import numpy as np
import tensorflow as tf
import plotting
import hls4ml
from qkeras import QConv2D, QDense, QActivation
from tensorflow.keras.layers import BatchNormalization, Reshape

from custom_layers import KLLoss, CustomMSE
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
from tensorflow_model_optimization.sparsity.keras import strip_pruning

# from hls4ml.model.hls_model import Layer, FixedPrecisionType, register_layer
from hls4ml.converters.keras_to_hls import parse_default_keras_layer, register_keras_layer_handler
from hls4ml.templates import get_backend

from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

class Sampling(Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def dense_to_hls():

    hardware = 'xcvu9p-flgb2104-2-e'

    input_file = '/eos/user/e/epuljak/forDelphes/CorrectDataResults/AE_models/QAT/AE_qkeras8'

    with open(input_file+'.json', 'r') as jsonfile:
        config = jsonfile.read()
    model = tf.keras.models.model_from_json(config,
        custom_objects={'PruneLowMagnitude': pruning_wrapper.PruneLowMagnitude,
            'QDense': QDense, 'QConv2D': QConv2D, 'QActivation': QActivation})
    model.load_weights(input_file+'.h5')

    true = model.input
    predicted = model.layers[-1].output
    scaled_input = BatchNormalization(trainable=False, name='scaled_input')(true)
    custom_output = CustomMSE(reshape=(19,3,1))([true, scaled_input, predicted])
    # create new model
    custom_model = Model(inputs=model.input, outputs=custom_output)

    custom_model.save('output/custom-dense_ae-qkeras8.h5')

    with open('output/data_5000.pickle', 'rb') as f:
        x_train, y_train, x_test, y_test, all_bsm_data, pt_scaler = pickle.load(f)

    # read model config
    config = hls4ml.utils.config_from_keras_model(
        model,
        default_precision='ap_fixed<16,6,AP_RND_CONV,AP_SAT>',
        max_bits=20,
        data_type_mode='auto_accum', # auto_accum_only
        granularity='name',
        test_inputs=x_test.reshape((-1,57)))
    print(config)

    # save config
    with open('output/custom-dense_ae-qkeras8.pickle', 'wb') as handle:
        pickle.dump(config, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    dense_to_hls()
