import argparse
import os
import random
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


random.seed(1234)

parser = argparse.ArgumentParser(
    description='Split model in a train validation set based on 60-20-20 and oversample minority class')
parser.add_argument("--data_dir", help="data_dir")
parser.add_argument("--cellnames_file", help="File containing cellnames to train")
args = parser.parse_args()


def plot_labels_per_image(split_name, y, folder):
    # sum the labels for each image to get the number of labels per image
    labels_per_image = y.sum(axis=1)

    plt.figure()
    plt.hist(labels_per_image, bins=range(y.shape[1] + 2), align='left', edgecolor='black')
    plt.title(f'{split_name} set: Number of Labels per Image')
    plt.xlabel('Number of labels')
    plt.ylabel('Number of images')
    plt.savefig(folder + f"/{split_name}_labels_per_image.png")

    # print summary statistics
    print(f'\n{split_name} set summary statistics:')
    print(f'Min number of labels: {labels_per_image.min()}')
    print(f'Max number of labels: {labels_per_image.max()}')
    print(f'Mean number of labels: {labels_per_image.mean():.2f}')
    print(f'Median number of labels: {np.median(labels_per_image)}')
    print(f'Standard deviation of number of labels: {labels_per_image.std():.2f}\n')

def sample_and_split(cellnames_file, run_dir):
    df_cell = pd.read_csv(cellnames_file, header=None).squeeze()
    for _, cellname in df_cell.items():
        split_train_validation_test(cellname, run_dir)


def split_train_validation_test(plate, run_dir):
    filename = plate + "_meta.csv"
    df = pd.read_csv(run_dir + filename)
    df.reset_index(inplace=True)
    X = df['index'].to_numpy().reshape(-1, 1)
    y = df.drop(columns='index').to_numpy()

    X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.6, random_state=1234)
    X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5)

    folder = run_dir + "Split/" + plate
    if not os.path.exists(folder):
        os.makedirs(folder)
        print("The new directory is created!")

    np.savetxt(folder + "/train.txt", X_train, fmt='%d')
    np.savetxt(folder + "/val.txt", X_valid, fmt='%d')
    np.savetxt(folder + "/test.txt", X_test, fmt='%d')

    # Plot the distribution of the number of labels per image for each split and print summary statistics
    for split_name, y_split in zip(["Train", "Validation", "Test"], [y_train, y_valid, y_test]):
        plot_labels_per_image(split_name, y_split, folder)


sample_and_split(args.cellnames_file, args.data_dir)

