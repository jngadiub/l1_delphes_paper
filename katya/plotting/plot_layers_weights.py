import os
import argparse
import json
import numpy as np
import tensorflow as tf
from qkeras import QConv2D, QDense, QActivation
from tensorflow_model_optimization.sparsity.keras import strip_pruning
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper

import cv2
from matplotlib.patches import FancyBboxPatch
from matplotlib import pyplot as plt

from plotting import add_logo

import mplhep as hep
plt.style.use(hep.style.CMS)

import matplotlib as mpl
mpl.rcParams['yaxis.labellocation'] = 'center'
mpl.rcParams['xaxis.labellocation'] = 'center'
mpl.rcParams['lines.markersize'] = 10
mpl.rcParams['lines.markeredgewidth'] = 2.0
mpl.rcParams['xtick.minor.top'] = False    # draw x axis top minor ticks
mpl.rcParams['xtick.minor.bottom'] = False    # draw x axis bottom minor ticks
mpl.rcParams['ytick.minor.left'] = True    # draw x axis top minor ticks
mpl.rcParams['ytick.minor.right'] = True    # draw x axis bottom minor ticks

wcols = ['#7a5195','#ef5675','#3690c0','#ffa600','#67a9cf','#014636', '#016c59']

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

def plot_layers_weights(model_path, output_dir):

    if 'h5' in model_path:
        model = tf.keras.models.load_model(model_path,
            custom_objects={'PruneLowMagnitude': pruning_wrapper.PruneLowMagnitude,
            'QDense': QDense, 'QConv2D': QConv2D, 'QActivation': QActivation})
    else:
        with open(f'{model_path}.json', 'r') as jsonfile:
            config = jsonfile.read()
        model = tf.keras.models.model_from_json(config,
            custom_objects={'PruneLowMagnitude': pruning_wrapper.PruneLowMagnitude,
            'QDense': QDense, 'QConv2D': QConv2D, 'QActivation': QActivation})
        model.load_weights(f'{model_path}.h5')
        model = flatten(model, 'conv_ae') if 'conv_ae' in model_path else model

    allWeightsByLayer = {}
    for layer in model.layers:
        print(layer._name)
        if (layer._name).find("batch")!=-1 or len(layer.get_weights())<1: continue
        layername = layer._name.replace('prune_low_magnitude_','').replace('_',' ').capitalize()
        if layername.find("prune")!=-1:
            layername = layername + ' (Pruned)'
        weights=layer.weights[0].numpy().flatten()
        allWeightsByLayer[layername] = weights
        print('Layer {}: % of zeros = {}'.format(layername,np.sum(weights==0)/np.size(weights)))
    labelsW = []
    histosW = []

    for key in reversed(sorted(allWeightsByLayer.keys())):
        labelsW.append(key)
        histosW.append(allWeightsByLayer[key])

    fig = plt.figure()
    ax = fig.add_subplot()
    plt.legend(loc='upper left', fontsize=15, frameon=False)
    plt.grid(False)
    # add_logo(ax, fig, 0.3, position='upper right')
    bins = np.linspace(-1.5, 1.5, 50)
    ax.hist(histosW, bins, histtype='stepfilled', stacked=True, label=labelsW,
        color=wcols if 'conv_ae' in model_path else wcols[:-1], edgecolor='black')

    model_name = 'CNN AE' if 'conv_ae' in model_path else 'DNN AE'
    ax.legend(frameon=False, loc='upper left', title=model_name)
    axis = plt.gca()
    ymin, ymax = axis.get_ylim()
    plt.ylabel('Number of Weights')
    plt.xlabel('Weights')
    model_type = 'cnn_ae' if 'conv_ae' in model_path else 'dnn_ae'
    pruning = 'not_pruned' if 'not_pruned' in model_path or 'notpruned' in model_path else 'pruned'

    print('Saving ',model_path, 'in', f'{model_type}_weights_{pruning}.pdf')
    plt.savefig(os.path.join(output_dir,f'{model_type}_weights_{pruning}.pdf'))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str)
    parser.add_argument('--output-dir', type=str, default='figures/')
    args = parser.parse_args()
    plot_layers_weights(**vars(args))
