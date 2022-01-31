import os
import argparse
import numpy as np
import math
import h5py
import tensorflow as tf
import joblib
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)
import matplotlib as mpl
mpl.rcParams["yaxis.labellocation"] = 'center'
mpl.rcParams["xaxis.labellocation"] = 'center'

def make_feature_plot(feature_index, predicted_qcd, X_test, particle_name,
    model_type, output_dir):

    fig = plt.figure(figsize=[14,10])
    input_featurenames = ['pT', 'eta', 'phi']

    true0 = X_test[:,:,feature_index].flatten() if len(X_test.shape)==4 else X_test[:,feature_index].flatten()
    predicted0 = predicted_qcd[:,:,feature_index].flatten() if len(predicted_qcd.shape)==4 else predicted_qcd[:,feature_index].flatten()

    zeroes = [i for i,v in enumerate(true0) if v==0]
    true0 = np.delete(true0, zeroes) if not(feature_index==1 and particle_name=='MET') else true0
    predicted0 = np.delete(predicted0, zeroes) if not(feature_index==1 and particle_name=='MET') else predicted0

    # trick on eta
    if feature_index==1:
        if particle_name=='Electrons':
            predicted0 = 3.0*np.tanh(predicted0)
        elif particle_name=='Muons':
            predicted0 = 2.1*np.tanh(predicted0)
        elif particle_name=='Jets':
            predicted0 = 4.0*np.tanh(predicted0)
    # trick on phi
    if feature_index==2:
        predicted0 = math.pi*np.tanh(predicted0)

    plt.hist(true0, 100, density=True, histtype='step', linewidth=1.5, label=f'{particle_name} True')
    plt.hist(predicted0, 100, density=True, histtype='step', linewidth=1.5, label=f'{particle_name} Predicted')
    if feature_index==0: plt.yscale('log', nonpositive='clip')
    plt.legend(frameon=False)
    plt.xlabel(str(input_featurenames[feature_index]))
    plt.ylabel('Prob. Density (a.u.)')
    plt.savefig(os.path.join(output_dir, f'reconstructions/{model_type}_{particle_name}_{input_featurenames[feature_index]}.pdf'))
    plt.clf()

    if not(feature_index==1 and particle_name=='MET'):
        delta = (true0-predicted0)/true0
        rmin = np.min(delta) if np.min(delta)>-1000 else -1000
        rmax = np.max(delta) if np.max(delta)<1000 else 1000
        n, bins, _ = plt.hist(delta, 100, range=(rmin, rmax), density=False,
            histtype='step', fill=False, linewidth=1.5, label=f'{particle_name}')
        # calculate mean and RMS of the pull
        mids = 0.5*(bins[1:] + bins[:-1])
        mean = np.average(mids, weights=n)
        var = np.average((mids - mean)**2, weights=n)
        sigma = np.sqrt(var)
        plt.yscale('log', nonpositive='clip')
        plt.legend(frameon=False, title=f'mean={mean:.2}\n RMS={sigma:.2}')
        plt.xlabel(str(input_featurenames[feature_index])+'pull')
        plt.ylabel('Prob. Density(a.u.)')
        plt.axvline(delta.mean(), color='k', linestyle='dashed', linewidth=1, label='mean = '+str(round(delta.mean(),2)))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,f'{model_type}_{particle_name}_{input_featurenames[feature_index]}_pull.pdf'))
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-file', type=str,
        default='output/result-conv_vae-8-b0.8-q0-pruned.h5',
        help='input file')
    parser.add_argument('--output-dir', type=str, default='figures/', help='output directory')
    args = parser.parse_args()

    model_type = 'cnn_vae' if '_vae' in args.results_file else 'cnn_ae'

    with h5py.File(args.results_file, 'r') as h5f:
        x_test = h5f['QCD'][:]
        predicted_qcd = h5f['predicted_QCD'][:]
        for i in range(3):
            make_feature_plot(i, predicted_qcd[:,0,:], x_test[:,0,:], 'MET', model_type, args.output_dir)
            make_feature_plot(i, predicted_qcd[:,1:5,:], x_test[:,1:5,:], 'Electrons', model_type, args.output_dir)
            make_feature_plot(i, predicted_qcd[:,5:9,:], x_test[:,5:9,:], 'Muons', model_type, args.output_dir)
            make_feature_plot(i, predicted_qcd[:,9:19,:], x_test[:,9:19,:], 'Jets', model_type, args.output_dir)
