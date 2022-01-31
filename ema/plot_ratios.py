import os
import h5py
import math
import numpy as np
import scipy as scipy
import argparse
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
from losses import mse_split_loss, radius, kl_loss, make_mse_loss_numpy

plt.rcParams['yaxis.labellocation'] = 'center'
plt.rcParams['xaxis.labellocation'] = 'center'
plt.rcParams['lines.markersize'] = 10
plt.rcParams['lines.markeredgewidth'] = 2.0
plt.rcParams['xtick.minor.top'] = False    # draw x axis top minor ticks
plt.rcParams['xtick.minor.bottom'] = False    # draw x axis bottom minor ticks
plt.rcParams['ytick.minor.left'] = True    # draw x axis top minor ticks
plt.rcParams['ytick.minor.right'] = True    # draw x axis bottom minor ticks
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['font.size'] = 16

bsm_labels = ['Leptoquark','A to 4 leptons','hChToTauNu', 'hToTauTau']
SAMPLES = ['QCD', 'Leptoquarks', 'Ato4l', 'hChToTauNu', 'hToTauTau']
LABELS = {'Leptoquarks': (r'LQ $\rightarrow$ b$\tau$', 'o', '#016c59'),
    'Ato4l': (r'A $\rightarrow$ 4$\ell$, 'X', '#7a5195'),
    'hChToTauNu': (r'$h_{\pm} \rightarrow \tau\nu$', 'v', '#67a9cf'),
    'hToTauTau': (r'$h_{0} \rightarrow \tau\tau$', 'd', '#ffa600')
         }

def plot(plt, data, model, label):
    #model = 'cnn_ae' if model=='conv_ae' else 'cnn_vae'
    for sample in SAMPLES:
        if sample=='QCD': continue
        data_val = [i[0] for i in data[sample]]
        data_err = [i[1] for i in data[sample]]
        plt.errorbar(list(range(2,18,2)), data_val, yerr=data_err, label=LABELS[sample][0],
            linestyle='None', marker=LABELS[sample][1], capsize=3, color=LABELS[sample][2])
        if 'tpr_kl' in label:
            plt.ylim(0, 2.0)
        elif 'tpr_radius' in label:
            plt.ylim(0, 2.0)
        elif 'auc' in label:
            plt.ylim(0.6, 1.4)
        #elif 'tpr' in label:
            #plt.ylim(0, 3)
        else:
            if model=='AE':
                plt.ylim(0, 2.0)
   
        plt.title('DNN AE' if model=='AE' else 'DNN VAE')
        if 'auc' in label: plt.legend(loc='best', frameon=False, fontsize=16)
        plt.ylabel('AUC/AUC baseline' if 'auc' in label else 'TPR/TPR baseline')
        plt.xlabel('Bit width')
        plt.xticks(range(2,18,2))
        plt.hlines(1, 1, 17, linestyles='--', color='#ef5675', linewidth=1.5)
        #plt.axhline(1, color='red', linestyle='dashed', linewidth=1)
#         if model == 'AE':
#             if 'tpr' in label:
#                 plt.legend(bbox_to_anchor=[1.0, 1], loc='upper right', frameon=False)
#             else:
#                 plt.legend(bbox_to_anchor=[1.0, 1], loc='lower right', frameon=False)
#         else: 
#             if 'tpr_radius' in label or 'tpr_kl' in label:
#                 plt.legend(loc='upper center', frameon=False)
#             else:
#                 plt.legend(bbox_to_anchor=[1.0, 1], loc='upper right', frameon=False)
        plt.tight_layout()
    plt.savefig(f'{model}_{label}_PTQ.pdf')
    plt.clf()

def read_data(results_file, model, beta=None):
    data = h5py.File(results_file, 'r')

    total_loss = []
    kl_data = []
    r_data = []
    mse_loss=[]
    
    X_test_scaled = data['QCD'][:]
    qcd_prediction = data['predicted_QCD'][:]
    #compute loss
    mse_loss.append(make_mse_loss_numpy(X_test_scaled, qcd_prediction.astype(np.float32), beta=beta))
    if model=='VAE':
        qcd_mean = data['encoded_mean_QCD'][:]
        qcd_logvar = data['encoded_logvar_QCD'][:]
        qcd_z = data['encoded_z_QCD'][:]
        kl_data.append(kl_loss(qcd_mean.astype(np.float32), qcd_logvar.astype(np.float32), beta=beta))
        r_data.append(radius(qcd_mean.astype(np.float32), qcd_logvar.astype(np.float32)))
        
    #BSM
    for bsm in bsm_labels:
        bsm_target=data[bsm+'_scaled'][:]
        bsm_prediction=data['predicted_'+ bsm][:]
        mse_loss.append(make_mse_loss_numpy(bsm_target, bsm_prediction.astype(np.float32), beta=beta))
        if model=='VAE':
            bsm_mean=data['encoded_mean_'+bsm][:]
            bsm_logvar=data['encoded_logvar_'+bsm][:]
            bsm_z=data['encoded_z_'+bsm][:]
            kl_data.append(kl_loss(bsm_mean.astype(np.float32), bsm_logvar.astype(np.float32), beta=beta))
            r_data.append(radius(bsm_mean.astype(np.float32), bsm_logvar.astype(np.float32)))
    
    if model=='VAE':
        total_loss=[]
        for mse, kl in zip(mse_loss, kl_data):
            total_loss.append(np.add(mse, kl))
    else:
        total_loss = mse_loss.copy()
    
    data.close()
    if model == 'VAE': del X_test_scaled, qcd_prediction, qcd_mean, qcd_logvar, qcd_z, bsm_target, bsm_prediction,\
                            bsm_mean, bsm_logvar, bsm_z
    else: del X_test_scaled, qcd_prediction, bsm_target, bsm_prediction

    return total_loss, kl_data, r_data

def read_data_PTQ(results_file, model, beta=None):
    data = h5py.File(results_file, 'r')
    
    #change to the one loss you need
    total_loss = []
    kl_data = []
    r_data = []
    mse_loss=[]
    print(data['QCD_Qkeras'][:].shape)
    kl_data.append(data['QCD_Qkeras'][:])
    
    #BSM
    for bsm in bsm_labels:
        kl_data.append(data['%s_Qkeras'%bsm][:])
        #kl_data.append(data['%s_BP'%bsm][:])
        
    data.close()

    return total_loss, kl_data, r_data

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
    #print(position)
    threshold_data = threshold_loss[position]
    pred_data = [1 if i>= threshold_data else 0 for i in list(pred_val)]
    tn, fp, fn, tp = confusion_matrix(true_val, pred_data).ravel()
    tpr_data = tp / (tp + fn)
    tpr_error = np.sqrt( tpr_data * (1 - tpr_data) / (tp + fn) )
    #print(tpr_error)
    return (auc_data, auc_error), (tpr_data, tpr_error)

def divide_error(numerator, denumerator):
    #print('Numerator: '+str(numerator)+', denumerator:'+str(denumerator))
    val = numerator[0]/denumerator[0]
    val_error = val * np.sqrt((numerator[1]/numerator[0])**2 + (denumerator[1]/denumerator[0])**2)
    #print(val_error)
    return (val, val_error)

def plot_metric(model, beta, pruning, directory_vae_pruned, directory_vae_qkeras, directory_ae_pruned,directory_ae_qkeras):
    
    if model=='VAE':
        auc_data_kl, auc_data_radius = dict(), dict()
        tpr_data_kl, tpr_data_radius = dict(), dict()
        for sample in SAMPLES:
            auc_data_kl[sample] = list()
            auc_data_radius[sample] = list()
            tpr_data_kl[sample] = list()
            tpr_data_radius[sample] = list()

        _, baseline_data_kl, _ = read_data(f'{directory_vae_pruned}/VAE_result_pruned.h5', model, beta=beta)
        #
        #correct radius
#         baseline_radius=list()
#         for rad in baseline_data_radius:
#             r = np.nan_to_num(rad);r[r==-np.inf] = 0;r[r==np.inf] = 0;r[r>=1E308] = 0;baseline_radius.append(r)
        integers=[1,2,2,3,3,4,4,6]
        for i, bit in enumerate(list(range(2,18,2))):
            print('Evaluating bit', bit)
            _, loss_data_kl, _ = read_data_PTQ(f'{directory_vae_qkeras}/PTQ_{model}_result_qkeras{bit}_1610.h5', model, beta=beta)
            #loss_data_radius
            #correct radius
#             loss_radius=list()
#             for rad in loss_data_radius:
#                 r = np.nan_to_num(rad);r[r==-np.inf] = 0;r[r==np.inf] = 0;r[r>=1E308] = 0;loss_radius.append(r)
            
            for i, sample in enumerate(SAMPLES):
                if sample=='QCD': continue
                auc_data_kl[sample].append(divide_error(get_metric(loss_data_kl[0], loss_data_kl[i])[0], get_metric(baseline_data_kl[0], baseline_data_kl[i])[0]))
#                 auc_data_radius[sample].append(divide_error(get_metric(loss_radius[0], loss_radius[i])[0], get_metric(baseline_radius[0], baseline_radius[i])[0]))
                tpr_data_kl[sample].append(divide_error(get_metric(loss_data_kl[0], loss_data_kl[i])[1], get_metric(baseline_data_kl[0], baseline_data_kl[i])[1]))
#                 tpr_data_radius[sample].append(divide_error(get_metric(loss_radius[0], loss_radius[i])[1], get_metric(baseline_radius[0], baseline_radius[i])[1]))

        plot(plt, auc_data_kl, model, 'auc_kl')
#         plot(plt, auc_data_radius, model, 'auc_radius')
        plot(plt, tpr_data_kl, model, 'tpr_kl')
#         plot(plt, tpr_data_radius, model, 'tpr_radius')

    else:
        auc_data = dict()
        tpr_data = dict()
        for sample in SAMPLES:
            auc_data[sample] = list()
            tpr_data[sample] = list()

        baseline_data, _, _ = read_data(f'{directory_ae_pruned}/{model}_result_pruned.h5', model)

        for bit in list(range(2,18,2)):
            print('Evaluating bit', bit)
            loss_data, _, _ = read_data_PTQ(f'{directory_ae_qkeras}/PTQ_{model}_result_qkeras{bit}.h5', model)
            for i, sample in enumerate(SAMPLES):
                if sample=='QCD': continue
                auc_data[sample].append(divide_error(get_metric(loss_data[0], loss_data[i])[0], get_metric(baseline_data[0], baseline_data[i])[0]))
                tpr_data[sample].append(divide_error(get_metric(loss_data[0], loss_data[i])[1], get_metric(baseline_data[0], baseline_data[i])[1]))

        plot(plt, auc_data, model, 'auc')
        plot(plt, tpr_data, model, 'tpr')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='VAE',
        choices=['VAE', 'AE'],
        help='Use either VAE or AE')
    #parser.add_argument('--latent-dim', default=3)
    parser.add_argument('--beta', type=float, default=0.8)
    parser.add_argument('--pruning', type=str, default='pruned')
    parser.add_argument('--directory_ae_pruned', type=str, default='AE_results')
    parser.add_argument('--directory_ae_qkeras', type=str, default='AE_results')
    parser.add_argument('--directory_vae_pruned', type=str, default='VAE_results')
    parser.add_argument('--directory_vae_qkeras', type=str, default='VAE_results')
    args = parser.parse_args()
    plot_metric(**vars(args))
