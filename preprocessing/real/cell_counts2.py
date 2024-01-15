import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
            if well not in dict_length:
                dict_length[well] = []
            if well not in dict_perc:
                dict_perc[well] = []
            dict_length[well].append(len(well_data))  # Count all cells
            dict_perc[well].append((well_data == 1).mean())  # Calculate the percentage of '1' cells

    folder = run_dir + "Split/"
    with open(folder + 'count_cells.csv', 'w') as f:
        for key, values in dict_length.items():
            f.write("%s, %s\n" % (key, np.mean(values)))
    with open(folder + 'perc_cells.csv', 'w') as f:
        for key, values in dict_perc.items():
            f.write("%s, %s\n" % (key, np.mean(values)))

    # plot histogram for counts
    for well, values in dict_length.items():
        plt.figure(figsize=(10, 5))
        plt.hist(values, bins=20, color='blue', alpha=0.5)
        plt.title(f"Cell counts histogram for well {well}")
        plt.xlabel("Cell count")
        plt.ylabel("Frequency")
        plt.savefig(folder + f'count_cells_{well}.png')

    # plot histogram for percentages
    for well, values in dict_perc.items():
        plt.figure(figsize=(10, 5))
        plt.hist(values, bins=20, color='red', alpha=0.5)
        plt.title(f"Cell percentages histogram for well {well}")
        plt.xlabel("Cell percentage")
        plt.ylabel("Frequency")
        plt.savefig(folder + f'perc_cells_{well}.png')

count_cells(args.cellnames_file, args.data_dir)
