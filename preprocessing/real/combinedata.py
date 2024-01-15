import argparse
import os
import h5py
import numpy as np
import pandas as pd
from collections import OrderedDict

parser = argparse.ArgumentParser(
    description='Merge h5 files with negative and positive labels into 1 file and add meta csv containing labels corresponding to indices')
parser.add_argument("--data_dir", help="data_dir")
parser.add_argument("--output_dir", help="output_dir")
parser.add_argument("--cellnames_file", help="File containing cellnames")
args = parser.parse_args()

def get_cellnames_from_txt(cellnames_file):
    with open(cellnames_file, 'r') as f:
        cellnames = [line.strip() for line in f]
    return cellnames

def merge_hdf5_files(data_dir, output_dir, cellnames):
    datasets = ["channel_1/images", "channel_6/images", "channel_9/images"]
    with h5py.File(os.path.join(output_dir, 'combined.h5'), 'w') as f_out:
        for dataset in datasets:
            data = []
            for cellname in cellnames:
                filename = cellname + '_90x90.h5'
                if filename in os.listdir(data_dir):
                    with h5py.File(os.path.join(data_dir, filename), 'r') as f_in:
                        data.append(f_in[dataset][()])
                        print(f"Properties for {filename} {dataset}:")
                        print("dtype:", f_in[dataset].dtype)
                        print("shape:", f_in[dataset].shape)
                        print("compression:", f_in[dataset].compression)

            data = np.concatenate(data, axis=0)
            f_out.create_dataset(dataset, data=data)
            # Print the properties of a dataset in the combined file
            print(f"Properties for combined.h5 {dataset}:")
            print("dtype:", f_out[dataset].dtype)
            print("shape:", f_out[dataset].shape)
            print("compression:", f_out[dataset].compression)

def merge_csv_files(data_dir, output_dir, cellnames):
    df_all = pd.DataFrame()
    for cellname in cellnames:
        filename = cellname + '_meta.csv'
        if filename in os.listdir(data_dir):
            df = pd.read_csv(os.path.join(data_dir, filename))
            df_all = df_all.append(df, ignore_index=True)
    df_all.to_csv(os.path.join(output_dir, 'combined.csv'), index=False)

cellnames = get_cellnames_from_txt(args.cellnames_file)
merge_hdf5_files(args.data_dir, args.output_dir, cellnames)
merge_csv_files(args.data_dir, args.output_dir, cellnames)

