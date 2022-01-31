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
    read_loss_data,
    BSM_SAMPLES,
    PLOTTING_LABELS
    )

def get_metric(qcd, bsm):
    true_val = np.concatenate((np.ones(bsm.shape[0]), np.zeros(qcd.shape[0])))
    pred_val = np.concatenate((bsm, qcd))

    fpr_loss, tpr_loss, threshold_loss = roc_curve(true_val, pred_val)
    auc_data = auc(fpr_loss, tpr_loss)

    return fpr_loss, tpr_loss, auc_data

def get_threshold(qcd, loss_type):

    qcd[::-1].sort()
    threshold = qcd[int(len(qcd)*10**-5)]
    print(loss_type, threshold)

    return threshold

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
    _, baseline_data_mse, baseline_data_kl, baseline_data_radius = read_loss_data(vae_file, 0.8)
    # load AE model
    baseline_total_loss, _, _, _ = read_loss_data(ae_file, 0.8)

    #correct radius
    baseline_radius=list()
    for rad in baseline_data_radius:
        r = np.nan_to_num(rad);r[r==-np.inf] = 0;r[r==np.inf] = 0;r[r>=1E308] = 0;baseline_radius.append(r)

    mse_tr = get_threshold(baseline_data_mse[0], 'mse vae')
    baseline_lq_mse = get_metric(baseline_data_mse[0], baseline_data_mse[1])
    baseline_ato4l_mse = get_metric(baseline_data_mse[0], baseline_data_mse[2])
    baseline_hChToTauNu_mse = get_metric(baseline_data_mse[0], baseline_data_mse[3])
    baseline_hToTauTau_mse = get_metric(baseline_data_mse[0], baseline_data_mse[4])

    kl_tr = get_threshold(baseline_data_kl[0], 'kl')
    baseline_lq_kl = get_metric(baseline_data_kl[0], baseline_data_kl[1])
    baseline_ato4l_kl = get_metric(baseline_data_kl[0], baseline_data_kl[2])
    baseline_hChToTauNu_kl = get_metric(baseline_data_kl[0], baseline_data_kl[3])
    baseline_hToTauTau_kl = get_metric(baseline_data_kl[0], baseline_data_kl[4])
    #
    radius_tr = get_threshold(baseline_radius[0], 'radius')
    baseline_lq_radius = get_metric(baseline_radius[0], baseline_radius[1])
    baseline_ato4l_radius = get_metric(baseline_radius[0], baseline_radius[2])
    baseline_hChToTauNu_radius = get_metric(baseline_radius[0], baseline_radius[3])
    baseline_hToTauTau_radius = get_metric(baseline_radius[0], baseline_radius[4])

    labels = ['IO VAE', r'VAE $D_{KL}$', r'VAE $R_z$', 'IO AE']

    marker, sample_label = return_label(anomaly)


    mse_ae_tr = get_threshold(baseline_total_loss[0], 'mse ae')
    if sample_label == r'LQ $\rightarrow$ b$\tau$':
        figures_of_merit = [baseline_lq_mse, baseline_lq_kl, baseline_lq_radius]
        figure_of_merit = get_metric(baseline_total_loss[0], baseline_total_loss[1])
    elif sample_label == r'A $\rightarrow$ 4$\ell$':
        figures_of_merit = [baseline_ato4l_mse, baseline_ato4l_kl, baseline_ato4l_radius]
        figure_of_merit = get_metric(baseline_total_loss[0], baseline_total_loss[2])
    elif sample_label == r'$h^{\pm} \rightarrow \tau\nu$':
        figures_of_merit = [baseline_hChToTauNu_mse, baseline_hChToTauNu_kl, baseline_hChToTauNu_radius]
        figure_of_merit = get_metric(baseline_total_loss[0], baseline_total_loss[3])
    elif sample_label == r'$h^{0} \rightarrow \tau\tau$':
        figures_of_merit = [baseline_hToTauTau_mse, baseline_hToTauTau_kl, baseline_hToTauTau_radius]
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
    plt.savefig(os.path.join(output_dir, f'{model_type}_rocs_{bsm}.pdf'))
    plt.clf()

    return kl_tr, radius_tr, mse_tr, mse_ae_tr

def plot_loss(vae_file, ae_file, loss_type, threshold, output_dir):

    model_type = 'cnn' if 'conv_' in vae_file else 'dnn'
    _, baseline_data_mse, baseline_data_kl, baseline_data_radius = read_loss_data(vae_file, 0.8)

    labels = {'mse': 'IO VAE', 'mse_ae': 'IO AE', 'kl': r'VAE $D_{KL}$', 'radius': r'VAE $R_z$'}

    if loss_type=='mse':
        base = baseline_data_mse
        hrange = np.linspace(0,5000,100) if model_type=='cnn' else np.linspace(0,50000,100)
    elif loss_type=='kl':
        base = baseline_data_kl
        hrange = np.linspace(0,10,100) if model_type=='cnn' else np.linspace(0,250,100)
    elif loss_type=='radius':
        base = baseline_data_radius
        hrange = np.linspace(0,5000,100) if model_type=='cnn' else np.linspace(0,5*10**10,100)
    elif loss_type=='mse_ae':
        base, _, _, _ = read_loss_data(ae_file, 0)
        hrange = np.linspace(0,5000,100) if model_type=='cnn' else np.linspace(0,50000,100)

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
    plt.vlines(threshold, 0, plt.gca().get_ylim()[1], linestyles='--', color='#ef5675', linewidth=3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_type}_loss_{loss_type}.pdf'))
    plt.clf()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ae', type=str)
    parser.add_argument('--vae', type=str)
    parser.add_argument('--output-dir', type=str, default='figures/')
    args = parser.parse_args()

    colors = ['#016c59', '#7a5195', '#ef5675', '#ffa600', '#67a9cf']

    for bsm in BSM_SAMPLES:
        kl_tr, radius_tr, mse_tr, mse_ae_tr = plot_rocs(args.vae, args.ae, bsm, args.output_dir)

    plot_loss(args.vae, args.ae, 'kl', kl_tr, args.output_dir)
    plot_loss(args.vae, args.ae, 'radius', radius_tr, args.output_dir)
    plot_loss(args.vae, args.ae, 'mse', mse_tr, args.output_dir)
    plot_loss(args.vae, args.ae, 'mse_ae', mse_ae_tr, args.output_dir)
