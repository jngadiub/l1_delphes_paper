import argparse
import h5py
import numpy as np


def create_data(inputs, out):
    data = []
    for inp in inputs:
        with h5py.File('/eos/project/d/dshep/TOPCLASS/DijetAnomaly/VAE_data/events/'+inp, 'r') as f:

            j1Eta = f['eventFeatures'][:,2]
            j1Phi = f['eventFeatures'][:,3]

            j2Eta = f['eventFeatures'][:,9]-f['eventFeatures'][:,2]
            j2Phi = f['eventFeatures'][:,10]-f['eventFeatures'][:,3]

            # put absolute phi and eta for the first 16 particles
            new_jet1 = np.empty((f['jetConstituentsList'].shape[0],16,3))
            new_jet2 = np.empty((f['jetConstituentsList'].shape[0],16,3))
            for i in range(16):
                new_jet1[:,i,0] = f['jetConstituentsList'][:,0,i,0] - j1Eta
                new_jet1[:,i,1] = f['jetConstituentsList'][:,0,i,1] - j1Phi
                # fit phi in +/-pi
                new_jet1[:,i,1] = np.where(new_jet1[:,i,1]>np.pi, new_jet1[:,i,1]-2*np.pi, new_jet1[:,i,1])
                new_jet1[:,i,1] = np.where(new_jet1[:,i,1]<-np.pi, new_jet1[:,i,1]+2*np.pi, new_jet1[:,i,1])

                new_jet2[:,i,0] = f['jetConstituentsList'][:,1,i,0] - j2Eta
                new_jet2[:,i,1] = f['jetConstituentsList'][:,1,i,1] - j2Phi
                # fit phi in +/-pi
                new_jet2[:,i,1] = np.where(new_jet2[:,i,1]>np.pi, new_jet2[:,i,1]-2*np.pi, new_jet2[:,i,1])
                new_jet2[:,i,1] = np.where(new_jet2[:,i,1]<-np.pi, new_jet2[:,i,1]+2*np.pi, new_jet2[:,i,1])

            new_jet1[:,:16,2] = f['jetConstituentsList'][:,0,:16,2]
            new_jet2[:,:16,2] = f['jetConstituentsList'][:,0,:16,2]
            # take first 16 particles
            data.append(np.concatenate((new_jet1, new_jet2), axis=0))
            particle_feature_names = np.array(f['particleFeatureNames'])
    data = np.concatenate(data, axis=0)
    print(out.strip('.h5'), data.shape)

    with h5py.File('/afs/cern.ch/work/e/egovorko/public/for_Hassan/'+out, 'w') as handle:
        handle.create_dataset('jetConstituentsList', data=data)
        handle.create_dataset('particleFeatureNames', data=particle_feature_names)


def dataset_for_hassan():
    # read QCD data
    create_data(['qcd_sqrtshatTeV_13TeV_PU40_NEW_EXT_sideband_parts/qcd_sqrtshatTeV_13TeV_PU40_NEW_EXT_sideband_000.h5',
                 'qcd_sqrtshatTeV_13TeV_PU40_NEW_EXT_signalregion_parts/qcd_sqrtshatTeV_13TeV_PU40_NEW_EXT_signalregion_000.h5'],
                 'bkg_3mln.h5')

    create_data(['RSGraviton_WW_BROAD_13TeV_PU40_1.5TeV_NEW_parts/RSGraviton_WW_BROAD_13TeV_PU40_1.5TeV_NEW_concat_000.h5',
                 'RSGraviton_WW_BROAD_13TeV_PU40_1.5TeV_NEW_parts/RSGraviton_WW_BROAD_13TeV_PU40_1.5TeV_NEW_concat_001.h5'],
                'sig_broad_1.5TeV.h5')

    create_data(['RSGraviton_WW_BROAD_13TeV_PU40_3.5TeV_NEW_parts/RSGraviton_WW_BROAD_13TeV_PU40_3.5TeV_NEW_concat_000.h5',
                 'RSGraviton_WW_BROAD_13TeV_PU40_3.5TeV_NEW_parts/RSGraviton_WW_BROAD_13TeV_PU40_3.5TeV_NEW_concat_001.h5'],
                 'sig_broad_3.5TeV.h5')

    create_data(['RSGraviton_WW_NARROW_13TeV_PU40_3.5TeV_NEW_parts/RSGraviton_WW_NARROW_13TeV_PU40_3.5TeV_NEW_concat_000.h5',
                 'RSGraviton_WW_NARROW_13TeV_PU40_3.5TeV_NEW_parts/RSGraviton_WW_NARROW_13TeV_PU40_3.5TeV_NEW_concat_001.h5'],
                 'sig_narrow_3.5TeV.h5')

    create_data(['RSGraviton_WW_NARROW_13TeV_PU40_1.5TeV_NEW_parts/RSGraviton_WW_NARROW_13TeV_PU40_1.5TeV_NEW_concat_000.h5',
                 'RSGraviton_WW_NARROW_13TeV_PU40_1.5TeV_NEW_parts/RSGraviton_WW_NARROW_13TeV_PU40_1.5TeV_NEW_concat_001.h5'],
                 'sig_narrow_1.5TeV.h5')


if __name__ == '__main__':
    dataset_for_hassan()