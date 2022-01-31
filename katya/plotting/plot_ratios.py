import os
import h5py
import math
import numpy as np
import scipy as scipy
import argparse
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc, confusion_matrix

import matplotlib as mpl
mpl.rcParams['yaxis.labellocation'] = 'center'
mpl.rcParams['xaxis.labellocation'] = 'center'
mpl.rcParams['lines.markersize'] = 10
mpl.rcParams['lines.markeredgewidth'] = 2.0
mpl.rcParams['xtick.minor.top'] = False    # draw x axis top minor ticks
mpl.rcParams['xtick.minor.bottom'] = False    # draw x axis bottom minor ticks
mpl.rcParams['ytick.minor.left'] = True    # draw x axis top minor ticks
mpl.rcParams['ytick.minor.right'] = True    # draw x axis bottom minor ticks
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['legend.fontsize'] = 16
mpl.rcParams['font.size'] = 16

from plotting import (
    LABELS,
    reco_loss,
    radius,
    kl_loss
    )

SAMPLES = ['QCD', 'Leptoquark', 'A to 4 leptons', 'hChToTauNu', 'hToTauTau']


def plot(plt, data, model, label, prefix):
    model = 'cnn_ae' if model=='conv_ae' else 'cnn_vae'
    for sample in SAMPLES:
        if sample=='QCD': continue
        data_val = [i[0] for i in data[sample]]
        data_err = [i[1] for i in data[sample]]
        print(model, data_val)
        plt.errorbar(list(range(2,18,2)), data_val, yerr=data_err, label=LABELS[sample][0],
            linestyle='None', marker=LABELS[sample][1], capsize=3, color=LABELS[sample][2])
        if 'tpr' in label:
            plt.ylim(0, 2)
            plt.yticks([0.0,0.5,1.0,1.5,2.0])
        else:
            plt.ylim(0.6, 1.4)
            plt.yticks([0.6,0.8,1.0,1.2,1.4])
        plt.ylabel('AUC/AUC baseline' if 'auc' in label else 'TPR/TPR baseline', fontsize=16)
        plt.xlabel('Bit width', fontsize=16)
        plt.title('CNN AE' if model=='cnn_ae' else 'CNN VAE')
        plt.xticks(range(2,18,2))
        plt.hlines(1, 1, 17, linestyles='--', color='#ef5675', linewidth=1.5)
        if prefix: plt.legend(loc='best', frameon=False, fontsize=16)
        plt.tight_layout()
    plt.savefig(f'figures/{prefix}{model}_{label}.pdf')
    plt.clf()

def read_data(results_file, model, beta, baseline=True):

    loss_data = []
    kl_data = []
    r_data = []

    with h5py.File(results_file, 'r') as h5f:
        for sample in SAMPLES:
            if sample=='QCD':
                inval = h5f[sample][int(h5f[sample].shape[0]/2):] if baseline \
                   else h5f[sample][:int(h5f[sample].shape[0]/2)]
            else:
                inval = h5f[sample+'_scaled'][int(h5f[sample+'_scaled'].shape[0]/2):] if baseline \
                   else h5f[sample+'_scaled'][:int(h5f[sample+'_scaled'].shape[0]/2)]

            outval = h5f['predicted_'+sample][int(h5f['predicted_'+sample].shape[0]/2):] if baseline \
                else h5f['predicted_'+sample][:int(h5f['predicted_'+sample].shape[0]/2)]
            meanval = h5f['encoded_mean_'+sample][int(h5f['encoded_mean_'+sample].shape[0]/2):] if baseline \
                 else h5f['encoded_mean_'+sample][:int(h5f['encoded_mean_'+sample].shape[0]/2)]
            logvarval = h5f['encoded_logvar_'+sample][int(h5f['encoded_logvar_'+sample].shape[0]/2):] if baseline \
                   else h5f['encoded_logvar_'+sample][:int(h5f['encoded_logvar_'+sample].shape[0]/2)]
            if model=='conv_vae' or model=='dense_vae':
                mseval = reco_loss(inval, outval, dense=False)
                kl = kl_loss(meanval, logvarval)
                kl_data.append(kl)
                r_data.append(radius(meanval, logvarval))
            else:
                mseval = reco_loss(inval, outval, dense=False)
            loss_data.append(mseval)

    del inval, outval, meanval, logvarval

    return loss_data, kl_data, r_data

def get_metric(qcd, bsm):
    true_val = np.concatenate((np.ones(bsm.shape[0]), np.zeros(qcd.shape[0])))
    pred_val = np.concatenate((bsm, qcd))

    fpr_loss, tpr_loss, threshold_loss = roc_curve(true_val, pred_val)
    auc_data = auc(fpr_loss, tpr_loss)

    # AUC error
    n_n = qcd.shape[0]
    n_p = bsm.shape[0]
    D_p = (n_p - 1) * ((auc_data/(2 - auc_data)) - auc_data**2)
    D_n = (n_n - 1) * ((2 * auc_data**2)/(1 + auc_data) - auc_data**2)
    auc_error = np.sqrt((auc_data * (1 - auc_data) + D_p + D_n)/(n_p * n_n))

    # TPR and its error
    position = np.where(fpr_loss>=10**(-5))[0][0]
    threshold_data = threshold_loss[position]
    pred_data = [1 if i>= threshold_data else 0 for i in list(pred_val)]
    tn, fp, fn, tp = confusion_matrix(true_val, pred_data).ravel()
    tpr_data = tp / (tp + fn)
    tpr_error = np.sqrt( tpr_data * (1 - tpr_data) / (tp + fn) )

    return (auc_data, auc_error), (tpr_data, tpr_error)

def divide_error(numerator, denumerator):
    val = numerator[0]/denumerator[0]
    val_error = val * np.sqrt((numerator[1]/numerator[0])**2 + (denumerator[1]/denumerator[0])**2)

    return (val, val_error)

def plot_ratios(model, latent_dim, beta, pruning, prefix):
    import matplotlib.pyplot as plt

    if model=='conv_vae':
        auc_data_kl, auc_data_radius, auc_data_mse = dict(), dict(), dict()
        tpr_data_kl, tpr_data_radius, tpr_data_mse = dict(), dict(), dict()
        for sample in SAMPLES:
            auc_data_kl[sample] = list()
            auc_data_radius[sample] = list()
            auc_data_mse[sample] = list()
            tpr_data_kl[sample] = list()
            tpr_data_radius[sample] = list()
            tpr_data_mse[sample] = list()

        baseline_data_mse, baseline_data_kl, baseline_data_radius = read_data(f'output/result-{model}-{latent_dim}-b{beta}-q0-{pruning}.h5', model, beta, baseline=True)
        #
        for bit in list(range(2,18,2)):
            print('Evaluating bit', bit)
            loss_data_mse, loss_data_kl, loss_data_radius = read_data(f'output/{prefix}result-{model}-{latent_dim}-b{beta}-q{bit}-{pruning}.h5', model, beta, baseline=False)
            #
            for i, sample in enumerate(SAMPLES):
                if sample=='QCD': continue
                auc_data_kl[sample].append(divide_error(get_metric(loss_data_kl[0], loss_data_kl[i])[0], get_metric(baseline_data_kl[0], baseline_data_kl[i])[0]))
                auc_data_radius[sample].append(divide_error(get_metric(loss_data_radius[0], loss_data_radius[i])[0], get_metric(baseline_data_radius[0], baseline_data_radius[i])[0]))
                auc_data_mse[sample].append(divide_error(get_metric(loss_data_mse[0], loss_data_mse[i])[0], get_metric(baseline_data_mse[0], baseline_data_mse[i])[0]))
                tpr_data_kl[sample].append(divide_error(get_metric(loss_data_kl[0], loss_data_kl[i])[1], get_metric(baseline_data_kl[0], baseline_data_kl[i])[1]))
                tpr_data_radius[sample].append(divide_error(get_metric(loss_data_radius[0], loss_data_radius[i])[1], get_metric(baseline_data_radius[0], baseline_data_radius[i])[1]))
                tpr_data_mse[sample].append(divide_error(get_metric(loss_data_mse[0], loss_data_mse[i])[1], get_metric(baseline_data_mse[0], baseline_data_mse[i])[1]))

        # plot(plt, auc_data_kl, model, 'auc_kl', prefix)
        # plot(plt, auc_data_radius, model, 'auc_radius', prefix)
        # plot(plt, auc_data_mse, model, 'auc_mse', prefix)
        plot(plt, tpr_data_kl, model, 'tpr_kl', prefix)
        # plot(plt, tpr_data_radius, model, 'tpr_radius', prefix)
        plot(plt, tpr_data_mse, model, 'tpr_mse', prefix)

    else:
        auc_data = dict()
        tpr_data = dict()
        for sample in SAMPLES:
            auc_data[sample] = list()
            tpr_data[sample] = list()

        baseline_data, _, _ = read_data(f'output/result-{model}-{latent_dim}-b0-q0-{pruning}.h5', model, beta, baseline=True)

        for bit in list(range(2,18,2)):
            print('Evaluating bit', bit)
            loss_data, _, _ = read_data(f'output/{prefix}result-{model}-{latent_dim}-b0-q{bit}-{pruning}.h5', model, beta, baseline=False)
            for i, sample in enumerate(SAMPLES):
                if sample=='QCD': continue
                auc_data[sample].append(divide_error(get_metric(loss_data[0], loss_data[i])[0], get_metric(baseline_data[0], baseline_data[i])[0]))
                tpr_data[sample].append(divide_error(get_metric(loss_data[0], loss_data[i])[1], get_metric(baseline_data[0], baseline_data[i])[1]))

        # plot(plt, auc_data, model, 'auc', prefix)
        plot(plt, tpr_data, model, 'tpr', prefix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='vae',
        choices=['conv_vae', 'conv_ae'], help='Use either VAE or AE')
    parser.add_argument('--latent-dim', default=8)
    parser.add_argument('--beta', type=float, default=0.8)
    parser.add_argument('--pruning', type=str, default='pruned')
    parser.add_argument('--prefix', type=str, default='')
    args = parser.parse_args()
    plot_ratios(**vars(args))