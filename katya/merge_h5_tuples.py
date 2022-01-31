#!/usr/bin/python
from __future__ import print_function, division
import os
import h5py
import numpy as np
import argparse

def merge_h5_tuples(output_file, input_files):

        keys = ['Particles', 'Particles_Classes', 'Particles_Names']
        data = {feature: np.array for feature in keys}

        input_files = [os.path.join(input_files[0], f) for f in os.listdir(input_files[0]) if os.path.isfile(os.path.join(input_files[0], f))] \
            if len(input_files)==1 else input_files
        for k in keys:
            data[k] = np.concatenate([h5py.File(input_file, 'r')[k] for input_file in input_files], axis=0)

        #write in data
        h5f = h5py.File(output_file, 'w')
        for feature in keys:
            h5f.create_dataset(feature, data=data[feature])
        h5f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-file', type=str, help='output file', required=True)
    parser.add_argument('--input-files', type=str, nargs='+', help='input files', required=True)
    args = parser.parse_args()
    merge_h5_tuples(**vars(args))