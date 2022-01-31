import math
import tensorflow as tf
from tensorflow.python.keras.layers.merge import _Merge as Merge
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops

class Distance(Merge):
    def _check_inputs(self, inputs):
        if len(inputs) not in  [2,3]:
            raise ValueError('A `{}` layer should be called '
                             'on exactly 2 or 3 inputs'.format(self.__class__.__name__))

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        super(Distance, self).build(input_shape)
        self._check_inputs(input_shape)

class KLLoss(Distance):

    def _merge_function(self, inputs):
        self._check_inputs(inputs)

        mean = inputs[0]
        log_var = inputs[1]

        kl = 1. + log_var - math_ops.square(mean) - math_ops.exp(log_var)
        kl = -0.5 * math_ops.reduce_mean(kl, axis=-1, keepdims=True)

        return kl

class Radius(Distance):

    def _merge_function(self, inputs):
        self._check_inputs(inputs)

        mean = inputs[0]
        log_var = inputs[1]

        sigma = math_ops.exp(log_var)

        radius = math_ops.div_no_nan(math_ops.square(mean), sigma)
        radius = math_ops.reduce_sum(radius, axis=-1, keepdims=True)

        return radius

class CustomMSE(Distance):

    def __init__(self, reshape, **kwargs):
        super(CustomMSE, self).__init__(**kwargs)
        self.reshape = tuple(reshape)

    def _merge_function(self, inputs):
        self._check_inputs(inputs)

        true = array_ops.reshape(inputs[0], (array_ops.shape(inputs[0])[0],) + self.reshape)
        true_scaled = array_ops.reshape(inputs[1], (array_ops.shape(inputs[1])[0],) + self.reshape)
        predicted = array_ops.reshape(inputs[2], (array_ops.shape(inputs[2])[0],) + self.reshape)
        # remove last dimension
        true = tf.squeeze(true, axis=-1)
        # remove last dimension
        true_scaled = tf.squeeze(true_scaled, axis=-1)
        true_scaled = tf.cast(true_scaled, dtype=tf.float32)
        # trick with phi
        outputs_phi = math.pi*math_ops.tanh(predicted)
        # trick with phi
        outputs_eta_egamma = 3.0*math_ops.tanh(predicted)
        outputs_eta_muons = 2.1*math_ops.tanh(predicted)
        outputs_eta_jets = 4.0*math_ops.tanh(predicted)
        outputs_eta = tf.concat([predicted[:,0:1,:,:], outputs_eta_egamma[:,1:5,:,:], outputs_eta_muons[:,5:9,:,:], outputs_eta_jets[:,9:19,:,:]], axis=1)
        # use both tricks
        predicted = tf.concat([predicted[:,:,0,:], outputs_eta[:,:,1,:], outputs_phi[:,:,2,:]], axis=2)
        # mask zero features from original input
        mask = math_ops.not_equal(true,0)
        mask = tf.cast(mask, tf.float32)
        predicted = mask * predicted
        # mask true scaled as well
        true_scaled = mask * true_scaled

        true_scaled = tf.reshape(true_scaled, [-1, 57])
        predicted = tf.reshape(predicted, [-1, 57])

        return  math_ops.reduce_mean(math_ops.square(true_scaled-predicted), axis=-1, keepdims=True)

    def get_config(self):
        config = {
            "reshape": self.reshape
        }
        base_config = super(CustomMSE, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))