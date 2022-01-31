import pickle
import pickle5
import argparse

def read_report(file):
    with open(file, 'rb') as handle:
        report = pickle.load(handle)

    print(file)
    print('DSP LUT FF BRAM Latency II')

    dsp = float(report['DSP48E'])/float(report['AvailableDSP48E']) * 100
    lut = float(report['LUT'])/float(report['AvailableLUT']) * 100
    ff = float(report['FF'])/float(report['AvailableFF']) * 100
    bram = float(report['BRAM_18K'])/float(report['AvailableBRAM_18K']) * 100
    latency = 5 * int(report['BestLatency'])
    ii = 5 * int(report['IntervalMin'])
    print(f' & {dsp:.1f} & {lut:.1f} & {ff:.2f} & {bram:.2f} & {latency} & {ii} \\\\')


    print(report['CosimIntervalMin'])

    # Xilinx V7-690 xc7vx690tffg1927-2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str)
    args = parser.parse_args()
    read_report(args.file)