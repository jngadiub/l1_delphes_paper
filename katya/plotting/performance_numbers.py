import os
import h5py
import math
import numpy as np
import scipy as scipy
import argparse
import tensorflow as tf
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc
from pickle import load

from plotting import (
    radius,
    reco_loss,
    kl_loss
    )

def performance_numbers(ae, vae, quantized=False):

    labels = ['Leptoquark','A to 4 leptons', 'hChToTauNu', 'hToTauTau']
    dense = False
    if 'epuljak' in ae:
        dense = True

    #read in data
    with h5py.File(ae, 'r') as h5f_ae:

        ae_mse_tpr = []
        ae_mse_auc = []
        ae_mse_qcd = reco_loss(np.array(h5f_ae['QCD']), np.array(h5f_ae['predicted_QCD']), dense=dense)

        for sample in labels:
            inval = np.array(h5f_ae[sample+'_scaled'])
            outval = np.array(h5f_ae['predicted_'+sample])
            ae_mse_bsm = reco_loss(inval, outval, dense=dense)

            ae_mse_true_val = np.concatenate((np.ones(ae_mse_bsm.shape[0]), np.zeros(ae_mse_qcd.shape[0])))
            ae_mse_pred_val = np.nan_to_num(np.concatenate((ae_mse_bsm, ae_mse_qcd)))

            ae_mse_fpr_loss, ae_mse_tpr_loss, _ = roc_curve(ae_mse_true_val, ae_mse_pred_val)
            ae_mse_tpr.append(np.interp(10**(-5), ae_mse_fpr_loss, ae_mse_tpr_loss)*100)
            ae_mse_auc.append(auc(ae_mse_fpr_loss, ae_mse_tpr_loss)*100)

    #read in data
    with h5py.File(vae, 'r') as h5f_vae:

        kl_tpr = []
        kl_auc = []
        r_tpr = []
        r_auc = []
        mse_tpr = []
        mse_auc = []
        kl_qcd = kl_loss(np.array(h5f_vae['encoded_mean_QCD']), np.array(h5f_vae['encoded_logvar_QCD']))
        if not quantized: mse_qcd = reco_loss(np.array(h5f_vae['QCD']), np.array(h5f_vae['predicted_QCD']), dense)
        r_qcd = radius(np.array(h5f_vae['encoded_mean_QCD']), np.array(h5f_vae['encoded_logvar_QCD']))

        for sample in labels:
            if not quantized:
                inval = np.array(h5f_vae[sample+'_scaled'])
                outval = np.array(h5f_vae['predicted_'+sample])
                mse_bsm = reco_loss(inval, outval, dense)
            meanval = np.array(h5f_vae['encoded_mean_'+sample])
            logvarval = np.array(h5f_vae['encoded_logvar_'+sample])
            kl_bsm = kl_loss(meanval, logvarval)
            r_bsm = radius(meanval, logvarval)

            kl_true_val = np.concatenate((np.ones(kl_bsm.shape[0]), np.zeros(kl_qcd.shape[0])))
            kl_pred_val = np.nan_to_num(np.concatenate((kl_bsm, kl_qcd)))

            kl_fpr_loss, kl_tpr_loss, kl_threshold_loss = roc_curve(kl_true_val, kl_pred_val)
            kl_tpr.append(np.interp(10**(-5), kl_fpr_loss, kl_tpr_loss)*100)
            kl_auc.append(auc(kl_fpr_loss, kl_tpr_loss)*100)

            r_true_val = np.concatenate((np.ones(r_bsm.shape[0]), np.zeros(r_qcd.shape[0])))
            r_pred_val = np.nan_to_num(np.concatenate((r_bsm, r_qcd)))

            r_fpr_loss, r_tpr_loss, r_threshold_loss = roc_curve(r_true_val, r_pred_val)
            r_tpr.append(np.interp(10**(-5), r_fpr_loss, r_tpr_loss)*100)
            r_auc.append(auc(r_fpr_loss, r_tpr_loss)*100)

            if not quantized:
                mse_true_val = np.concatenate((np.ones(mse_bsm.shape[0]), np.zeros(mse_qcd.shape[0])))
                mse_pred_val = np.nan_to_num(np.concatenate((mse_bsm, mse_qcd)))

                mse_fpr_loss, mse_tpr_loss, mse_threshold_loss = roc_curve(mse_true_val, mse_pred_val)
                mse_tpr.append(np.interp(10**(-5), mse_fpr_loss, mse_tpr_loss)*100)
                mse_auc.append(auc(mse_fpr_loss, mse_tpr_loss)*100)

    print('Results for', ae)

    mod = 'CNN' if 'conv_' in ae else 'DNN'
    print(r'\hline')
    print(f'{mod} AE  & IO & {ae_mse_tpr[0]:.2f}&{ae_mse_tpr[1]:.2f}&{ae_mse_tpr[2]:.2f}&{ae_mse_tpr[3]:.2f}'\
                         f'& {ae_mse_auc[0]:.0f}&{ae_mse_auc[1]:.0f}&{ae_mse_auc[2]:.0f}&{ae_mse_auc[3]:.0f}' + r'\\')
    print(r'\hline')
    if not quantized:
        print(f'\\multirow{{3}}{{*}}{{{mod} VAE}} & IO & {mse_tpr[0]:.2f}&{mse_tpr[1]:.2f}&{mse_tpr[2]:.2f}&{mse_tpr[3]:.2f}'\
                                                 f'& {mse_auc[0]:.0f}&{mse_auc[1]:.0f}&{mse_auc[2]:.0f}&{mse_auc[3]:.0f}'  +r'\\')
        print(r'\cline{2-10}')
        print(f'{mod} VAE & \\Rz  & {r_tpr[0]:.2f}&{r_tpr[1]:.2f}&{r_tpr[2]:.2f}&{r_tpr[3]:.2f}'\
                                f'& {r_auc[0]:.0f}&{r_auc[1]:.0f}&{r_auc[2]:.0f}&{r_auc[3]:.0f}' + r'\\')
    print(r'\cline{2-10}')
    print(f'          & \\Dkl & {kl_tpr[0]:.2f}&{kl_tpr[1]:.2f}&{kl_tpr[2]:.2f}&{kl_tpr[3]:.2f}'\
                            f'& {kl_auc[0]:.0f}&{kl_auc[1]:.0f}&{kl_auc[2]:.0f}&{kl_auc[3]:.0f}' + r'\\')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ae', type=str)
    parser.add_argument('--vae', type=str)
    parser.add_argument('--quantized', action='store_true')
    args = parser.parse_args()
    performance_numbers(**vars(args))
