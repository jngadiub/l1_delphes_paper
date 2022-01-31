import os
import h5py
import math
import numpy as np
import argparse
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
mpl.rcParams['xtick.labelsize'] = 40
mpl.rcParams['ytick.labelsize'] = 40
mpl.rcParams['legend.fontsize'] = 40
mpl.rcParams['font.size'] = 40

def make_feature_plot(feature_index, feature_name, file, dataset_name, particle_name, prange):
    colors = {'qcd':'#016c59', 'lq':'#7a5195', 'ato4l':'#ef5675', 'hChToTauNu':'#ffa600', 'hToTauTau':'#67a9cf'}

    with h5py.File(file, 'r') as h5f:
        data = h5f['Particles'][:,:,:3]
        data[:,9:19,0] = np.where(data[:,9:19,1]>4,0,data[:,9:19,0])
        data[:,9:19,0] = np.where(data[:,9:19,1]<-4,0,data[:,9:19,0])
        data[:,9:19,1] = np.where(data[:,9:19,1]>4,0,data[:,9:19,1])
        data[:,9:19,1] = np.where(data[:,9:19,1]<-4,0,data[:,9:19,1])
        data[:,9:19,2] = np.where(data[:,9:19,1]>4,0,data[:,9:19,2])
        data[:,9:19,2] = np.where(data[:,9:19,1]<-4,0,data[:,9:19,2])
        is_ele = data[:,1,0] > 23
        is_mu = data[:,5,0] > 23
        is_lep = (is_ele+is_mu) > 0
        data_filtered = data[is_lep]
        particle = data_filtered[:,prange[0]:prange[1],feature_index].flatten()

    particle_flat = [i for i in particle if i!=0]
    del particle

    hist_range = {'MET': [(0,1000), (-4.6,4.6), (-3.2,3.2)],
        r'$e/\gamma$': [(0,1000), (-3.1,3.1), (-3.2,3.2)],
        r'$\mu$': [(0,1000), (-2.2,2.2), (-3.2,3.2)],
        'jet': [(0,1000), (-4.15,4.15), (-3.2,3.2)],
        }

    labels = {
        'qcd': 'Background',
        'lq': r'LQ $\rightarrow$ b$\tau$',
        'ato4l': r'A $\rightarrow$ 4$\ell$',
        'hChToTauNu': r'$h^{0} \rightarrow \tau\tau$',
        'hToTauTau': r'$h^{\pm} \rightarrow \tau\nu$'
        }

    y, bin_edges = np.histogram(particle_flat, bins=100, range=hist_range[particle_name][feature_index])
    bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])

    plt.errorbar(bin_centers, y,
        yerr=y**0.5,
        color=colors[dataset_name],
        drawstyle='steps-mid',
        linewidth=3,
        label=labels[dataset_name])

    plt.hist(particle_flat, 100,
        range=hist_range[particle_name][feature_index],
        linewidth=3,
        histtype='step',
        linestyle='-',
        color=colors[dataset_name])


    plt.yscale('log', nonpositive='clip')
    # if feature_index==0 and particle_name=='MET': legend = plt.legend(frameon=False, loc='best')
    plt.xlabel(f'{particle_name} {feature_name}')
    plt.ylabel('Simulated events')
    plt.tight_layout()

def plot(files, output_dir):

    features = [('pT', r'$p_{T}$'), ('eta', r'$\eta$'), ('phi', r'$\phi$')]


    for particle, prange, particle_file_name in [('MET', [0,1], 'met'), (r'$e/\gamma$', [1,5], 'electrons'), (r'$\mu$', [5,9], 'muons'), ('jet', [9,19], 'jets')]:
        for i, feature in enumerate(features):
            fig = plt.figure()
            ax = fig.add_subplot()
            make_feature_plot(i, feature[1], files[0], 'qcd', particle, prange)
            make_feature_plot(i, feature[1], files[1], 'lq', particle, prange)
            make_feature_plot(i, feature[1], files[2], 'ato4l', particle, prange)
            make_feature_plot(i, feature[1], files[3], 'hChToTauNu', particle, prange)
            make_feature_plot(i, feature[1], files[4], 'hToTauTau', particle, prange)
            plt.savefig(f'{output_dir}/dataset_{feature[0]}_{particle_file_name}.pdf')
            plt.clf()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', type=str, nargs='+')
    parser.add_argument('--output-dir', default='figures/', type=str)
    args = parser.parse_args()
    plot(**vars(args))
