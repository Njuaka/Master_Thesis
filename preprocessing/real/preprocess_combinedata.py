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
args = parser.parse_args()

# Function definition

def preprocessing_hdf5(run_dir, plate, path_pos, path_neg, merged_h5):
    with h5py.File(path_neg, 'r') as f_neg, h5py.File(path_pos, 'r') as f_pos:
        datasets = ["channel_1/images", "channel_6/images", "channel_9/images"]
        for d in datasets:
            merged_h5[d].append(np.append(f_pos[d], f_neg[d], axis=0))

def generate_meta_csv(run_dir, cell_names, pos_path, neg_path, merged_csv):
    for i in range(len(cell_names)):
        with h5py.File(pos_path[i], 'r') as f_pos, h5py.File(neg_path[i], 'r') as f_neg:
            labels = np.append(np.repeat(1, f_pos["channel_1/images"].shape[0]),
                               np.repeat(0, f_neg["channel_1/images"].shape[0]))

            df_dict = {}
            for j in range(len(cell_names)):
                if j == i:
                    df_dict[cell_names[j]] = labels
                else:
                    df_dict[cell_names[j]] = np.repeat(2, len(labels))

            for key, value in df_dict.items():
                merged_csv[key].append(value.tolist())

# Step 1: get list of the plates

# This script assumes the following file structure:
plates = os.listdir(args.data_dir)

# Loop over plates (pick either pos or neg directory) and add name of plate + results ls in dir to list
pos_path = []
neg_path = []
cell_names = []

for plate in plates:
    cells = os.listdir(args.data_dir + '/' + plate + '/PeNeg/')
    cell_names.extend([plate + '_' + cell[:-3] for cell in cells])
    pos_path.extend([args.data_dir + '/' + plate + '/PePos/' + cell for cell in cells])
    neg_path.extend([args.data_dir + '/' + plate + '/PeNeg/' + cell for cell in cells])

# Zip the lists together and sort them
sorted_tuples = sorted(zip(pos_path, neg_path, cell_names))
    
# Unzip the sorted list of tuples back into individual lists
pos_path, neg_path, cell_names = zip(*sorted_tuples)

# Combine all HDF5 files into one dataset HDF5 file
merged_h5_path = args.output_dir + '/merged_dataset.h5'
merged_h5 = h5py.File(merged_h5_path, 'w')

for i in range(len(cell_names)):
    preprocessing_hdf5(args.output_dir, cell_names[i], pos_path[i], neg_path[i], merged_h5)

merged_h5.close()

# Combine all CSV files into one CSV file
merged_csv_path = args.output_dir + '/merged_meta.csv'
merged_csv = {}

for cell_name in cell_names:
    merged_csv[cell_name] = []

generate_meta_csv(args.output_dir, cell_names, pos_path, neg_path, merged_csv)

df = pd.DataFrame(merged_csv)
df.to_csv(merged_csv_path, index=False)
