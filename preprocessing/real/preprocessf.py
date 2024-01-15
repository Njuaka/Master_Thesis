import argparse
import os
import h5py
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(
    description='Merge h5 files with negative and positive labels into 1 file and add meta csv containing labels corresponding to indices')
parser.add_argument("--data_dir", help="data_dir")
parser.add_argument("--output_dir", help="output_dir")
args = parser.parse_args()


# Function definition

def preprocessing_hdf5(run_dir, plate, path_pos, path_neg):
    prefix = run_dir + "/" + plate
    path = prefix + "_90x90.h5"
    with h5py.File(path_neg, 'r') as f_neg, h5py.File(path_pos, 'r') as f_pos, h5py.File(path, 'w') as f:
        datasets = ["channel_1/images", "channel_6/images", "channel_9/images"]
        for d in datasets:
            f.create_dataset(d, data=np.append(f_pos[d], f_neg[d], axis=0))
        
def generate_meta_csv(run_dir, cell_names, pos_path, neg_path):
    for i in range(len(cell_names)):
        prefix = run_dir + "/" + cell_names[i]
        with h5py.File(pos_path[i], 'r') as f_pos, h5py.File(neg_path[i], 'r') as f_neg:
            labels = np.append(np.repeat(1, f_pos["channel_1/images"].shape[0]),
                               np.repeat(-1, f_neg["channel_1/images"].shape[0]))
            other_labels = np.repeat(0, len(labels))
            df_dict = {cell_names[i]: labels}
            for j in range(len(cell_names)):
                if j != i:
                    df_dict[cell_names[j]] = other_labels
            df = pd.DataFrame(df_dict)
            df.to_csv(prefix + "_meta.csv", index=False)

# Step 1: get list of the plates

# This script assumes the following file structure:

plates = os.listdir(args.data_dir)

# Loop over plates  ( pick either pos or neg directory) and add name of plate +results ls in dir to list
pos_path = []
neg_path = []
cell_names = []

for plate in plates:
    cells = os.listdir(args.data_dir + '/' + plate + '/PeNeg/')
    cell_names.extend([plate + '_' + cell[:-3] for cell in cells])
    pos_path.extend([args.data_dir + '/' + plate + '/PePos/' + cell for cell in cells])
    neg_path.extend([args.data_dir + '/' + plate + '/PeNeg/' + cell for cell in cells])

# now merge both negative an positive cells into one hdf5 file and generate meta.csv
for i in range(len(cell_names)):
    preprocessing_hdf5(args.output_dir, cell_names[i], pos_path[i], neg_path[i])

generate_meta_csv(args.output_dir, cell_names, pos_path, neg_path)
