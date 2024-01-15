import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Calculates cell per well')
parser.add_argument("--data_dir", help="data_dir")
args = parser.parse_args()


def count_cells(combined_csv_file):
    """
    This function calculates the number of cells,
    the output is written to the Split directory in the run folder.
    """
    df = pd.read_csv(combined_csv_file)
    dict_length = {}
    dict_perc = {0: [], 1: [], 2: []}
    dict_count = {0: 0, 1: 0, 2: 0}
    dict_stats = {'min': [], 'max': [], 'mean': [], 'median': [], 'std': [], 'missing_perc': []}

    # Iterate through all columns in the DataFrame
    for column in df.columns:
        # Calculate the percentage of each label (0, 1, 2) in the current column and store in the corresponding list
        for label in [0, 1, 2]:
            dict_perc[label].append((df[column] == label).mean())
            dict_count[label] += (df[column] == label).sum()

                # Add statistics for each column
        dict_stats['min'].append(df[column].min())
        dict_stats['max'].append(df[column].max())
        dict_stats['mean'].append(df[column].mean())
        dict_stats['median'].append(df[column].median())
        dict_stats['std'].append(df[column].std())

         # Calculate percentage of missing values for each column (label 2 represents missing values)
        dict_stats['missing_perc'].append((df[column] == 2).mean())

    #for label in [0, 1, 2]:
        # Calculate the percentage of each label (0, 1, 2) and store in the corresponding list
        #dict_perc[label].append((df == label).mean())

    dict_length['total'] = len(df)
    
    folder = args.data_dir + "/Split/"
    with open(folder + 'count_cells.csv', 'w') as f:
        for key in dict_length.keys():
            f.write("%s, %s\n" % (key, dict_length[key]))

    with open(folder + 'perc_cells.csv', 'w') as f:
        for key in dict_perc.keys():
            f.write("%s, %s\n" % (key, dict_perc[key]))

    with open(folder + 'count_per_label.csv', 'w') as f:
        for key in dict_count.keys():
            f.write("%s, %s\n" % (key, dict_count[key]))

    with open(folder + 'stats_per_label.csv', 'w') as f:
        for key in dict_stats.keys():
            f.write("%s, %s\n" % (key, dict_stats[key]))

    # plot the histograms
    plt.figure()
    plt.hist(dict_length.values(), bins=20, color='b', alpha=0.5) 
    plt.title('Histogram of cell counts')
    plt.xlabel('Cell counts')
    plt.ylabel('Frequency')
    plt.savefig(folder + 'histogram_cell_counts.png')

    plt.figure(figsize=(10, 6))
    for label in [0, 1, 2]:
        plt.hist(dict_perc[label], bins=20, alpha=0.5, label=f'Label {label}')
    plt.title('Histogram of cell percentages')
    plt.xlabel('Cell percentages')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.savefig(folder + 'histogram_cell_percentages.png')

    # Plot count per label
    labels = ['Negative', 'Positive', 'Missing']
    counts = [dict_count[label] for label in [0, 1, 2]]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, counts, color=['red', 'green', 'blue'])
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.title('Count per label')
    plt.savefig(folder + 'count_per_label.png')

combined_csv_file = args.data_dir + "/combined.csv"

# Call the function with the combined CSV file path
count_cells(combined_csv_file)