import argparse
import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import Counter
from imblearn.over_sampling import RandomOverSampler

random.seed(1234)

parser = argparse.ArgumentParser(
    description='Split model in a train validation set based on 60-20-20 and oversample minority class')
parser.add_argument("--data_dir", help="data_dir")
args = parser.parse_args()


def split_train_validation_test(run_dir):
    filename = "combined.csv"  # Updated file name
    df = pd.read_csv(os.path.join(run_dir, filename))  # Using os.path.join to handle file paths correctly
    df.reset_index(inplace=True)
    X = df['index'].to_numpy().reshape(-1, 1)
    y = df.drop(columns='index').to_numpy()

    X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.6, random_state=1234,stratify=y)
    X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5,stratify=y_rem)
    
    ros = RandomOverSampler()

    # Convert y_train into composite labels (strings)
    y_train_composite = [''.join(map(str, row)) for row in y_train]

    # Oversample based on composite labels
    X_train, y_train_over_composite = ros.fit_resample(X_train, y_train_composite)

    # Convert the composite labels back to multi-label format
    y_train = np.array([[int(digit) for digit in label] for label in y_train_over_composite])
 
    folder = os.path.join(run_dir, "Split", "combined")  # Updated folder path

    if not os.path.exists(folder):
        os.makedirs(folder)
        print("The new directory is created!")

    np.savetxt(os.path.join(folder, "train.txt"), X_train, fmt='%d')  # Using os.path.join for file paths
    np.savetxt(os.path.join(folder, "val.txt"), X_valid, fmt='%d')
    np.savetxt(os.path.join(folder, "test.txt"), X_test, fmt='%d')  # Script to split data into train, validation and test set


    # Create label distributions and plot them
    for split_name, y_split in zip(["Train", "Validation", "Test"], [y_train, y_valid, y_test]):
        plot_label_distribution(split_name, y_split, folder)


def plot_label_distribution(split_name, y, folder):
    # flatten y to a 1D array
    y_flattened = y.flatten()
    label_counts = Counter(y_flattened)
    labels, counts = zip(*sorted(label_counts.items(), key=lambda x: x[1]))  # Sort labels by count in descending order

    total_counts = sum(counts)  # calculate total count of all labels

    plt.figure()
    #plt.bar(labels, counts)
    for label, count in zip(labels, counts):
        percentage = count / total_counts * 100  # calculate percentage for each label
        plt.bar(label, count, label=f'Label {label}: {percentage:.2f}%')  # include percentage in the label
        
    plt.legend(loc='upper left')
    plt.title(f'{split_name} set label distribution')
    plt.xlabel('Labels')
    plt.ylabel('Number of instances')
    plt.tight_layout()
    plt.savefig(folder + f"/{split_name}_label_distribution.png")


split_train_validation_test(args.data_dir)
