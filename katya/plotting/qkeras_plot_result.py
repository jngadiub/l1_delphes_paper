import os
import h5py
import math
import numpy as np
import scipy as scipy
import argparse
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc
from pickle import load

def radius(mean, logvar):
    sigma = np.sqrt(np.exp(logvar))
    radius = mean*mean/sigma/sigma
    radius = np.sum(radius, axis=-1)

    radius = np.nan_to_num(radius)
    radius[radius==-np.inf] = 0
    radius[radius==np.inf] = 0
    radius[radius>=1E308] = 0
    return radius

def make_plot_training_history(plt, h5f, output_dir):
    loss = h5f['loss'][:]
    val_loss = h5f['val_loss'][:]

    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('Training History')

    plt.semilogy()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['training', 'validation'], loc='upper right')
    plt.tight_layout()
    plt.savefig(output_dir+'training_hist.pdf')
    return plt

def make_plot_loss_distribution(plt, mse_data, loss_type, labels, output_dir):
    bin_size = 500
    plt.figure()
    for i, label in enumerate(labels):
        plt.hist(mse_data[i], bins=bin_size, label=label, histtype='step', fill=False, linewidth=1.5)
    plt.semilogy()
    plt.semilogx()
    plt.title(loss_type)
    plt.xlabel('Autoencoder Loss')
    plt.ylabel('Probability (a.u.)')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{loss_type}_dist.pdf'))
    return plt

def make_plot_roc_curves(plt, qcd, bsm, label, output_dir, loss_type, color_id):
    colors = ['#4eb3d3','#7a5195','#ef5675','#3690c0','#ffa600','#67a9cf','#014636','#016c59']

    true_val = np.concatenate((np.ones(bsm.shape[0]), np.zeros(qcd.shape[0])))
    pred_val = np.concatenate((bsm, qcd))

    fpr_loss, tpr_loss, threshold_loss = roc_curve(true_val, pred_val)

    auc_loss = auc(fpr_loss, tpr_loss)

    plt.plot(fpr_loss, tpr_loss, '-', label=f'{label} {loss_type} (auc = %.1f%%)'%(auc_loss*100.),
        linewidth=1.5) #, color=colors[color_id])
    plt.plot(np.linspace(0, 1),np.linspace(0, 1), '--', color='0.75')
    plt.vlines(1e-5, 0, 1, linestyles='--', color='lightcoral')

    plt.semilogx()
    plt.semilogy()
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    return plt

def make_feature_plot(plt, feature_index, predicted_qcd, X_test, particle_name,
    output_dir, sample_name='QCD'):

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

    fig, ax = plt.subplots(2)

    ax[0].hist(true0, 100, density=True, histtype='step', linewidth=1.5, label=f'{particle_name} True')
    ax[0].hist(predicted0, 100, density=True, histtype='step', linewidth=1.5, label=f'{particle_name} Predicted')
    if feature_index==0: ax[0].set_yscale('log', nonpositive='clip')
    ax[0].legend(fontsize=12, frameon=False)
    ax[0].set_xlabel(str(input_featurenames[feature_index]), fontsize=12)
    ax[0].set_ylabel('Prob. Density (a.u.)', fontsize=12)

    if not(feature_index==1 and particle_name=='MET'):
        pull = (true0-predicted0)/true0
        rmin = np.min(pull) if np.min(pull)>-1000 else -1000
        rmax = np.max(pull) if np.max(pull)<1000 else 1000
        n, bins, _ = ax[1].hist(pull, 100, range=(rmin, rmax), density=False, histtype='step', fill=False, linewidth=1.5, label=f'{particle_name} Pull')
        # calculate mean and RMS of the pull
        mids = 0.5*(bins[1:] + bins[:-1])
        mean = np.average(mids, weights=n)
        var = np.average((mids - mean)**2, weights=n)
        sigma = np.sqrt(var)
        ax[1].set_yscale('log', nonpositive='clip')
        ax[1].legend(fontsize=12, frameon=False, title=f'mean={mean:.2} RMS={sigma:.2}')
        ax[1].set_xlabel(str(input_featurenames[feature_index]), fontsize=12)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{particle_name}_{input_featurenames[feature_index]}.pdf'))

def make_plot_features(plt, h5f, output_dir):
    X_test = h5f['QCD'][:]
    predicted_qcd = h5f['predicted_QCD'][:]
    for i in range(3):
        make_feature_plot(plt, i, predicted_qcd[:,0,:], X_test[:,0,:], 'MET', output_dir)
        make_feature_plot(plt, i, predicted_qcd[:,1:5,:], X_test[:,1:5,:], 'Electrons', output_dir)
        make_feature_plot(plt, i, predicted_qcd[:,5:9,:], X_test[:,5:9,:], 'Muons', output_dir)
        make_feature_plot(plt, i, predicted_qcd[:,9:19,:], X_test[:,9:19,:], 'Jets', output_dir)
    return plt

def mse_loss(inputs, outputs):
    return np.mean((inputs-outputs)*(inputs-outputs), axis=-1)

def kl_loss(z_mean, z_log_var):
    kl = 1 + z_log_var - np.square(z_mean) - np.exp(z_log_var)
    kl = - 0.5 * np.mean(kl, axis=-1) # multiplying mse by N -> using sum (instead of mean) in kl loss (todo: try with averages)
    return kl

def compute_loss(inputs, outputs, z_mean=None, z_log_var=None, beta=None):
    # trick on phi
    outputs_phi = math.pi*np.tanh(outputs)
    # trick on eta
    outputs_eta_egamma = 3.0*np.tanh(outputs)
    outputs_eta_muons = 2.1*np.tanh(outputs)
    outputs_eta_jets = 4.0*np.tanh(outputs)
    outputs_eta = np.concatenate([outputs[:,0:1,:,:], outputs_eta_egamma[:,1:5,:,:], outputs_eta_muons[:,5:9,:,:], outputs_eta_jets[:,9:19,:,:]], axis=1)
    outputs = np.concatenate([outputs[:,:,0,:], outputs_eta[:,:,1,:], outputs_phi[:,:,2,:]], axis=2)
    # change input shape
    inputs = np.squeeze(inputs, -1)
    # calculate and apply mask
    mask = np.not_equal(inputs, 0)
    outputs = np.multiply(outputs, mask)

    loss = mse_loss(inputs.reshape(inputs.shape[0],57), outputs.reshape(outputs.shape[0],57))

    reco_loss = np.copy(loss)
    kl = None
    if z_mean is not None:
        kl = kl_loss(z_mean, z_log_var)
        loss = reco_loss + kl if beta==0 else (1-beta)*reco_loss + beta*kl

    return loss, kl, reco_loss

def plot_result(model, latent_dim, results_file, beta, output_dir):

    from matplotlib import pyplot as plt
    labels = ['QCD', 'Leptoquark', 'A to 4 leptons', 'hChToTauNu', 'hToTauTau']

    for i, sample in enumerate(labels):
        if i==0: continue
        for i_bit, bit in enumerate(range(0,18,2)):
            results_file = f'output/result-conv_vae-8-b0.8-q{bit}-pruned.h5'
            with h5py.File(results_file, 'r') as h5f:
                loss_qcd, kl_qcd, reco_loss_qcd = compute_loss(np.array(h5f['QCD']), np.array(h5f['predicted_QCD']), \
                    np.array(h5f['encoded_mean_QCD']), np.array(h5f['encoded_logvar_QCD']), 0.8)
                loss_bsm, kl_bsm, reco_loss_bsm = compute_loss(np.array(h5f[f'{sample}_scaled']), np.array(h5f[f'predicted_{sample}']),\
                    np.array(h5f[f'encoded_mean_{sample}']), np.array(h5f[f'encoded_logvar_{sample}']), 0.8)
                plt = make_plot_roc_curves(plt, kl_qcd, kl_bsm, sample, output_dir, f'KL {bit} bits', i_bit)
        plt.savefig(os.path.join(output_dir, f'rocs_kl_{sample}.pdf'))
        plt.clf()

    # for i, sample in enumerate(labels):
    #     if i==0: continue
    #     for i_bit, bit in enumerate(range(0,18,2)):
    #         results_file = f'output/result-conv_ae-8-b0-q{bit}-pruned.h5'
    #         with h5py.File(results_file, 'r') as h5f:
    #             loss_qcd, kl_qcd, reco_loss_qcd = compute_loss(np.array(h5f['QCD']), np.array(h5f['predicted_QCD']))
    #             loss_bsm, kl_bsm, reco_loss_bsm = compute_loss(np.array(h5f[f'{sample}_scaled']), np.array(h5f[f'predicted_{sample}']))
    #             plt = make_plot_roc_curves(plt, loss_qcd, loss_bsm, sample, output_dir, f'KL {bit} bits', i_bit)
    #     plt.savefig(os.path.join(output_dir, f'rocs_ae_{sample}.pdf'))
    #     plt.clf()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='conv_vae',
        help='Use either VAE or AE')
    parser.add_argument('--latent-dim', default=8)
    parser.add_argument('--results-file', type=str)
    parser.add_argument('--beta', type=float, default=0.8)
    parser.add_argument('--output-dir', type=str, default='rocs', help='output directory')
    args = parser.parse_args()
    plot_result(**vars(args))
