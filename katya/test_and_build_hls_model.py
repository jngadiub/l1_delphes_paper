import os
import pickle
import pickle5
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

from plotting.plotting import PLOTTING_LABELS, make_plot_roc_curves, mse_loss

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

def test_hls_model(hardware, input_file, output_folder, output_report,
    build, input_config, plots_dir):

    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    model_type = 'dnn' if 'epuljak' in input_file or 'dense' in input_file else 'cnn'

    if not ('.h5' in input_file):
        with open(input_file+'.json', 'r') as jsonfile:
            config = jsonfile.read()
        model = tf.keras.models.model_from_json(config,
            custom_objects={'PruneLowMagnitude': pruning_wrapper.PruneLowMagnitude,
                'QDense': QDense, 'QConv2D': QConv2D, 'QActivation': QActivation,
                'KLLoss': KLLoss, 'CustomMSE': CustomMSE})
        model.load_weights(input_file+'.h5')
    else:
        # load keras model
        model = tf.keras.models.load_model(input_file,
          custom_objects={'PruneLowMagnitude': pruning_wrapper.PruneLowMagnitude,
          'QDense': QDense, 'QConv2D': QConv2D, 'QActivation': QActivation,
          'KLLoss': KLLoss, 'CustomMSE': CustomMSE})

    model.summary()

    if '_vae' in input_file:
        # insert the cast layer
        x = model.layers[0].output
        for i in range(1,11):
            x = model.layers[i](x)
            if i == 5 or i == 8:
                x = QActivation('quantized_bits(16,6)', name='cast_{}'.format(i))(x)
        x0 = model.layers[11](x)
        x1 = model.layers[12](x)
        y = model.layers[13]([x0, x1])
        model = Model(inputs=model.layers[0].output, outputs=y)
        model.summary()

    # load config
    with open(input_config, 'rb') as handle:
        config = pickle5.load(handle) if 'epuljak' in input_config else pickle5.load(handle)
    for layer in config['LayerName'].keys():
         config['LayerName'][layer]['Trace'] = True
    print(config)

    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=output_folder.strip('/'),
        fpga_part=hardware)

    # remove linear nodes from the network
    for l in list(hls_model.get_layers()):
        if '_linear' in l.name: hls_model.remove_node(l)

    if model_type=='cnn':
        # load dataset
        with open('output/data_-1.pickle', 'rb') as handle:
            x_train, y_train, x_test, y_test, all_bsm_data, pt_scaler = pickle.load(handle)

        # Profiling weights
        from matplotlib import pyplot as plt
        print('Profiling hls model')
        wp, wph, ap, aph = hls4ml.model.profiling.numerical(
            model=model,
            hls_model=hls_model,
            X=np.concatenate((x_test[:10000],all_bsm_data[0],all_bsm_data[1]),axis=0))
        wp.savefig(os.path.join(plots_dir,'wp.pdf'))
        wph.savefig(os.path.join(plots_dir,'wph.pdf'))
        ap.savefig(os.path.join(plots_dir,'ap.pdf'))
        aph.savefig(os.path.join(plots_dir,'aph.pdf'))
        plt.clf()

        print('Plotting hls model')
        hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True,
            to_file=os.path.join(plots_dir,'hls-model.pdf'))

        from matplotlib import pyplot as plt
        model.summary()
        model.compile()
        print('Keras predictions')
        y_keras = model.predict(x_test)
        print('Compiling hls model')
        hls_model.compile()
        y_hls = hls_model.predict(x_test)

        fig1 = compare(model, hls_model, x_test[:1000], plot_type="dist_diff")
        fig1.savefig(os.path.join(plots_dir,'compare.pdf'))

        # Tracing the model output layer by layer
        print('Tracing hls model')
        hls4ml_pred, hls4ml_trace = hls_model.trace(x_test[:5])
        keras_trace = hls4ml.model.profiling.get_ymodel_keras(model, x_test[:5])
        for layer in hls4ml_trace.keys():
            if layer in keras_trace.keys():
                print(f'Keras layer {layer}, first sample:')
                print(config['LayerName'][layer])
                print(keras_trace[layer][:].flatten()[:])
                print(hls4ml_trace[layer][:].flatten()[:])
                print(keras_trace[layer][:].flatten()[:]-hls4ml_trace[layer][:].flatten()[:])

        if 'q0' in input_file:
            model_no_qkeras = tf.keras.models.load_model(input_file.replace('custom', 'custom-no_qkeras'),
          custom_objects={'PruneLowMagnitude': pruning_wrapper.PruneLowMagnitude,
          'QDense': QDense, 'QConv2D': QConv2D, 'QActivation': QActivation,
          'KLLoss': KLLoss, 'CustomMSE': CustomMSE})
            y_no_qkeras = model_no_qkeras.predict(x_test)

        ## Plotting ROCs
        print('Plotting ROCs...')
        fig = plt.figure(figsize=[18,10])
        for i, bsm in enumerate(all_bsm_data):
            bsm_keras = model.predict(bsm)
            bsm_hls = hls_model.predict(bsm)
            if 'q0' in input_file:
                bsm_no_qkeras = model_no_qkeras.predict(bsm)
            make_plot_roc_curves(plt, y_keras, bsm_keras, f'QKeras {PLOTTING_LABELS[i+1]}', i, alpha=0.5, line='--')
            make_plot_roc_curves(plt, y_hls, bsm_hls, f'hls {PLOTTING_LABELS[i+1]}', i)
            if 'q0' in input_file:
                make_plot_roc_curves(plt, y_no_qkeras, bsm_no_qkeras, f'Keras {PLOTTING_LABELS[i+1]}', i, line=':')
        plt.savefig(os.path.join(plots_dir,'rocs.pdf'))

    if model_type=='dnn':
        # load dataset
        with open('output/data_-1.pickle', 'rb') as handle:
            x_train, y_train, x_test, y_test, all_bsm_data, pt_scaler = pickle.load(handle)

        # Profiling weights
        from matplotlib import pyplot as plt
        print('Profiling hls model')
        wp, wph, ap, aph = hls4ml.model.profiling.numerical(
            model=model,
            hls_model=hls_model,
            X=x_test[:10000].reshape((-1,57))
            )
        wp.savefig(os.path.join(plots_dir,'wp.pdf'))
        wph.savefig(os.path.join(plots_dir,'wph.pdf'))
        ap.savefig(os.path.join(plots_dir,'ap.pdf'))
        aph.savefig(os.path.join(plots_dir,'aph.pdf'))
        plt.clf()

        print('Plotting hls model')
        hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True,
            to_file=os.path.join(plots_dir,'hls-model.pdf'))

        from matplotlib import pyplot as plt
        model.summary()
        model.compile()
        print('Keras predictions')
        y_keras = model.predict(x_test.reshape((-1,57)))
        print('Compiling hls model')
        hls_model.compile()
        y_hls = hls_model.predict(x_test.reshape((-1,57)))

        fig1 = compare(model, hls_model, x_test[:1000].reshape((-1,57)), plot_type="dist_diff")
        fig1.savefig(os.path.join(plots_dir,'compare.pdf'))

        ## Plotting ROCs
        print('Plotting ROCs...')
        fig = plt.figure(figsize=[18,10])
        for i, bsm in enumerate(all_bsm_data):
            bsm_keras = model.predict(bsm.reshape((-1,57)))
            bsm_hls = hls_model.predict(bsm.reshape((-1,57)))
            make_plot_roc_curves(plt, y_keras, bsm_keras, f'QKeras {PLOTTING_LABELS[i+1]}', i, alpha=0.5, line='--')
            make_plot_roc_curves(plt, y_hls, bsm_hls, f'hls {PLOTTING_LABELS[i+1]}', i)
        plt.savefig(os.path.join(plots_dir,'rocs.pdf'))


    if build:
        report = hls_model.build(reset=True, csim=True, cosim=True, synth=True, vsynth=True)
        print(report)
        # save report
        with open(output_report, 'wb') as handle:
            pickle.dump(report, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hardware', default='xcvu9p-flgb2104-2-e', type=str, help='Train with pruning')
    parser.add_argument('--input-file', type=str, help='Where to save the model')
    parser.add_argument('--output-folder', type=str, help='Where to save the model')
    parser.add_argument('--output-report', type=str, help='Where to save the model')
    parser.add_argument('--build', type=int, default=0, help='If to build model')
    parser.add_argument('--input-config', type=str, help='Where to save the config')
    parser.add_argument('--plots-dir', type=str, help='Where to save the plots')
    args = parser.parse_args()
    test_hls_model(**vars(args))
