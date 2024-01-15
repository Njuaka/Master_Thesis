import argparse
import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

random.seed(1234)

parser = argparse.ArgumentParser(
    description='Combine all data and split in a train validation set based on 60-20-20')
parser.add_argument("--data_dir", help="data_dir")
parser.add_argument("--cellnames_file", help="File containing cellnames to train")
args = parser.parse_args()


def sample_and_split(cellnames_file, run_dir):
    df_all = pd.DataFrame()  # Initialize an empty DataFrame
    df_cell = pd.read_csv(cellnames_file, header=None).squeeze()
    for _, cellname in df_cell.items():
        filename = cellname + "_meta.csv"
        df = pd.read_csv(run_dir + filename)
        df_all = df_all.append(df)  # Append to the combined DataFrame
    split_train_validation_test(df_all, run_dir)  # Pass the combined DataFrame to the split function


def split_train_validation_test(df, run_dir):
    df.reset_index(inplace=True)
    X = df['index'].to_numpy().reshape(-1, 1)
    y = df.drop(columns='index').to_numpy()

    X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.6, random_state=1234)
    X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5)
    folder = run_dir + "Split/"

    if not os.path.exists(folder):
        os.makedirs(folder)
        print("The new directory is created!")

    np.savetxt(folder + "/train.txt", X_train, fmt='%d')
    np.savetxt(folder + "/val.txt", X_valid, fmt='%d')
    np.savetxt(folder + "/test.txt", X_test, fmt='%d')


sample_and_split(args.cellnames_file, args.data_dir)
