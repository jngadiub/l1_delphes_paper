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
    loss = tf.math.reduce_mean(loss, axis=0) # average over batch
    return loss

def make_kl(z_mean, z_log_var):
    @tf.function
    def kl_loss(inputs, outputs):
        kl =  - 0.5 * (1. + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl = tf.reduce_mean(kl, axis=-1) # average over the latent space
        kl = tf.reduce_mean(kl, axis=-1) # average over batch
        return kl
    return kl_loss

def make_mse_kl(z_mean, z_log_var, beta):
    kl_loss = make_kl(z_mean, z_log_var)
    # multiplying back by N because input is so sparse -> average error very small
    @tf.function
    def mse_kl_loss(inputs, outputs):
        return make_mse(inputs, outputs) + kl_loss(inputs, outputs) if beta==0 \
            else (1 - beta) * make_mse(inputs, outputs) + beta * kl_loss(inputs, outputs)
    return mse_kl_loss
