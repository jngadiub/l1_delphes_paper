import h5py
import os
import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt
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
mpl.rcParams['xtick.labelsize'] = 26
mpl.rcParams['ytick.labelsize'] = 26
mpl.rcParams['legend.fontsize'] = 26
mpl.rcParams['font.size'] = 26

from plotting import (
    add_logo,
    BSM_SAMPLES,
    kl_loss,
    mse_loss,
    reco_loss,
    PLOTTING_LABELS
    )

def read_loss_data(results_file, beta=None):

    with h5py.File(results_file, 'r') as data:

        vae = True if ('vae' in results_file) or ('VAE' in results_file) else False
        dense = True if 'epuljak' in results_file else False

        total_loss = []
        kl_data = []
        mse_loss=[]

        if vae:
            qcd_mean = data['encoded_mean_QCD'][:]
            qcd_logvar = data['encoded_logvar_QCD'][:]
            kl_data.append(kl_loss(qcd_mean.astype(np.float32), qcd_logvar.astype(np.float32)))
            for bsm in BSM_SAMPLES:
                bsm_mean = data['encoded_mean_'+bsm][:]
                bsm_logvar = data['encoded_logvar_'+bsm][:]
                kl_data.append(kl_loss(bsm_mean.astype(np.float32), bsm_logvar.astype(np.float32)))

        else:
            X_test_scaled = data['QCD'][:]
            qcd_prediction = data['predicted_QCD'][:]
            #compute loss
            mse_loss.append(reco_loss(X_test_scaled, qcd_prediction.astype(np.float32), dense=dense))
            #BSM
            for bsm in BSM_SAMPLES:
                bsm_target = data[bsm+'_scaled'][:]
                bsm_prediction = data['predicted_'+ bsm][:]
                mse_loss.append(reco_loss(bsm_target, bsm_prediction.astype(np.float32), dense=dense))

    if vae: del qcd_mean, qcd_logvar, bsm_mean, bsm_logvar
    else: del X_test_scaled, qcd_prediction, bsm_target, bsm_prediction

    return mse_loss, kl_data

def get_metric(qcd, bsm):
    true_val = np.concatenate((np.ones(bsm.shape[0]), np.zeros(qcd.shape[0])))
    pred_val = np.concatenate((bsm, qcd))

    fpr_loss, tpr_loss, threshold_loss = roc_curve(true_val, pred_val)
    auc_data = auc(fpr_loss, tpr_loss)
    return fpr_loss, tpr_loss, auc_data

def return_label(anomaly):
    if (anomaly == 'Leptoquark'):
        marker = 'o'; sample_label=r'LQ $\rightarrow$ b$\tau$'
    elif (anomaly == 'A to 4 leptons'):
        marker='X'; sample_label=r'A $\rightarrow$ 4$\ell$'
    elif anomaly == 'hToTauTau':
        marker = 'd'; sample_label=r'$h^{0} \rightarrow \tau\tau$'
    else:
        marker = 'v'; sample_label=r'$h^{\pm} \rightarrow \tau\nu$'
    return marker, sample_label

def plot_rocs(vae_file, ae_file, anomaly, output_dir):

    model_type = 'cnn' if 'conv_' in vae_file else 'dnn'
    _, baseline_data_kl = read_loss_data(vae_file, 0.8)
    # load AE model
    baseline_total_loss, _ = read_loss_data(ae_file, 0.8)

    baseline_lq_kl = get_metric(baseline_data_kl[0], baseline_data_kl[1])
    baseline_ato4l_kl = get_metric(baseline_data_kl[0], baseline_data_kl[2])
    baseline_hChToTauNu_kl = get_metric(baseline_data_kl[0], baseline_data_kl[3])
    baseline_hToTauTau_kl = get_metric(baseline_data_kl[0], baseline_data_kl[4])
    #
    labels = [ r'VAE $D_{KL}$', 'IO AE']

    marker, sample_label = return_label(anomaly)

    if sample_label == r'LQ $\rightarrow$ b$\tau$':
        figures_of_merit = [baseline_lq_kl]
        figure_of_merit = get_metric(baseline_total_loss[0], baseline_total_loss[1])
    elif sample_label == r'A $\rightarrow$ 4$\ell$':
        figures_of_merit = [baseline_ato4l_kl]
        figure_of_merit = get_metric(baseline_total_loss[0], baseline_total_loss[2])
    elif sample_label == r'$h^{\pm} \rightarrow \tau\nu$':
        figures_of_merit = [baseline_hChToTauNu_kl]
        figure_of_merit = get_metric(baseline_total_loss[0], baseline_total_loss[3])
    elif sample_label == r'$h^{0} \rightarrow \tau\tau$':
        figures_of_merit = [baseline_hToTauTau_kl]
        figure_of_merit = get_metric(baseline_total_loss[0], baseline_total_loss[4])

    for i, base in enumerate(figures_of_merit):
        plt.plot(base[0], base[1], "-",
            label=f'{labels[i]} (AUC = {base[2]*100:.0f}%)',
            linewidth=3, color=colors[i])

    plt.plot(figure_of_merit[0], figure_of_merit[1], "-",
        label=f'{labels[-1]} (AUC = {figure_of_merit[2]*100:.0f}%)',
        linewidth=3, color=colors[-1])

    plt.xlim(10**(-6),1)
    plt.ylim(10**(-6),1.2)
    plt.semilogx()
    plt.semilogy()
    plt.ylabel('True Positive Rate', fontsize=26)
    plt.xlabel('False Positive Rate', fontsize=26)
    plt.plot(np.linspace(0, 1),np.linspace(0, 1), ':', color='0.75', linewidth=3)
    plt.vlines(1e-5, 0, 1, linestyles='--', color='#ef5675', linewidth=3)
    plt.legend(loc='lower right', frameon=False, title=f'{model_type} '.upper()+f'ROC {sample_label}', fontsize=26)
    # add_logo(ax, fig, 0.3, position='lower left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_type}_quantized_rocs_{bsm}.pdf'))
    plt.clf()

def plot_loss(vae_file, ae_file, loss_type, output_dir):

    model_type = 'cnn' if 'conv_' in vae_file else 'dnn'
    _, baseline_data_kl = read_loss_data(vae_file, 0.8)

    labels = {'mse': 'IO VAE', 'mse_ae': 'IO AE', 'kl': r'VAE $D_{KL}$', 'radius': r'VAE $R_z$'}

    if loss_type=='kl':
        base = baseline_data_kl
        hrange = np.linspace(0,40,100)
    elif loss_type=='mse_ae':
        base, _ = read_loss_data(ae_file, 0)
        hrange = np.linspace(0,5000,100)

    for i, base_bsm in enumerate(base):
        plt.hist(base_bsm, hrange,
            label=PLOTTING_LABELS[i],
            linewidth=3,
            color=colors[i],
            histtype='step',
            density=True)

    plt.semilogy()
    plt.ylabel('A.U.', fontsize=26)
    plt.xlabel(f'{model_type} '.upper()+f'{labels[loss_type]}', fontsize=26)
    if loss_type=='kl': plt.legend(loc='upper right', frameon=False, fontsize=26)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_type}_quantized_loss_{loss_type}.pdf'))
    plt.clf()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ae', type=str)
    parser.add_argument('--vae', type=str)
    parser.add_argument('--output-dir', type=str, default='figures/')
    args = parser.parse_args()

    colors = ['#7a5195', '#67a9cf']

    for bsm in BSM_SAMPLES:
        plot_rocs(args.vae, args.ae, bsm, args.output_dir)
