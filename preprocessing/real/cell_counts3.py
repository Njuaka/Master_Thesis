import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

parser = argparse.ArgumentParser(description='Calculates cell per well')
parser.add_argument("--data_dir", help="data_dir")
parser.add_argument("--cellnames_file", help="File containing cellnames to train")
args = parser.parse_args()

def count_cells(cellnames_file, run_dir):
    df_cell = pd.read_csv(cellnames_file, header=None).squeeze()
    dict_length = {}
    dict_perc = {}

    for _, cellname in df_cell.items():
        filename = cellname + "_meta.csv"
        df = pd.read_csv(run_dir + filename)

        for well in df.columns:
            well_data = df[well].to_numpy()
            dict_length[well] = len(well_data)  # Count all cells
            dict_perc[well] = (well_data == 1).mean()  # Calculate the percentage of '1' cells

    folder = run_dir + "Split/"
    with open(folder + 'count_cells.csv', 'w') as f:
        for key, value in dict_length.items():
            f.write("%s, %s\n" % (key, value))
    with open(folder + 'perc_cells.csv', 'w') as f:
        for key, value in dict_perc.items():
            f.write("%s, %s\n" % (key, value))

    count_groups = defaultdict(list)
    for well, count in dict_length.items():
        count_groups[count].append(well)
    perc_groups = defaultdict(list)
    for well, perc in dict_perc.items():
        perc_groups[perc].append(well)

    # Plot histograms
    plt.figure(figsize=(10, 6))
    for count, wells in count_groups.items():
        if len(wells) > 1:
            label = 'others'
        else:
            label = wells[0]
        plt.hist([count]*len(wells), bins=20, label=label, alpha=0.5)
    plt.title('Histogram of cell counts')
    plt.xlabel('Count')
    plt.ylabel('Frequency')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
    plt.savefig(folder + 'count_cells.png')

    plt.figure(figsize=(10, 6))
    for perc, wells in perc_groups.items():
        if len(wells) > 1:
            label = 'others'
        else:
            label = wells[0]
        plt.hist([perc]*len(wells), bins=20, label=label, alpha=0.5)
    plt.title('Histogram of cell percentages')
    plt.xlabel('Percentage')
    plt.ylabel('Frequency')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
    plt.savefig(folder + 'perc_cells.png')

count_cells(args.cellnames_file, args.data_dir)
