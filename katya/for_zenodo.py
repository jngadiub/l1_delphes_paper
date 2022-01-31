import argparse
import h5py
import numpy as np

def for_zenodo():
    input_file = 'BlackBox_13TeV_PU20.h5'
    # read QCD data
    h5f = h5py.File(input_file, 'r')
    data = h5f['Particles'][:,:,:]
    np.random.shuffle(data)
    data = data[:,:,:]
    print('before cuts', data.shape[0])
    # remove jets eta >4 or <-4
    data[:,9:19,0] = np.where(data[:,9:19,1]>4,0,data[:,9:19,0])
    data[:,9:19,0] = np.where(data[:,9:19,1]<-4,0,data[:,9:19,0])
    data[:,9:19,1] = np.where(data[:,9:19,1]>4,0,data[:,9:19,1])
    data[:,9:19,1] = np.where(data[:,9:19,1]<-4,0,data[:,9:19,1])
    data[:,9:19,2] = np.where(data[:,9:19,1]>4,0,data[:,9:19,2])
    data[:,9:19,2] = np.where(data[:,9:19,1]<-4,0,data[:,9:19,2])

    # filter out events that don't have first lepton pt > 23 GeV/c
    is_ele = data[:,1,0] > 23
    is_mu = data[:,5,0] > 23
    is_lep = (is_ele+is_mu) > 0
    data = data[is_lep]
    print('after cuts', data.shape[0])

    with h5py.File('/eos/project/d/dshep/L1anomaly_DELPHES/for_Zenodo/BlackBox_13TeV_PU20_1.h5', 'w') as f:
        f.create_dataset('Particles',
            data=np.array(data), compression='gzip')
        f.create_dataset('Particles_Names',
            data=h5f['Particles_Names'], compression='gzip')
        f.create_dataset('Particles_Classes',
            data=h5f['Particles_Classes'],
            compression='gzip')

    # with h5py.File('/eos/project/d/dshep/L1anomaly_DELPHES/for_Zenodo/background_for_testing.h5', 'w') as h5f:
    #     print(data[4000000:].shape[0])
    #     h5f.create_dataset('Particles',
    #         data=np.array(data[4000000:]), compression='gzip')
    #     h5f.create_dataset('Particles_Names',
    #         data=np.array(['Pt', 'Eta', 'Phi', 'Class']), compression='gzip')
    #     h5f.create_dataset('Particles_Classes',
    #         data=np.array(['MET_class_1', 'Four_Ele_class_2', 'Four_Mu_class_3', 'Ten_Jet_class_4']),
    #         compression='gzip')


    h5f.close()

if __name__ == '__main__':
    for_zenodo()