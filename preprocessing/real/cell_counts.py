import argparse
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Calculates cell per well')
parser.add_argument("--data_dir", help="data_dir")
parser.add_argument("--cellnames_file", help="File containing cellnames to train")
args = parser.parse_args()

def count_cells(cellnames_file, run_dir):
    """
    This function calculates the number of cells,
    the output is written to the Split directory in the run folder.
    """
    df_cell = pd.read_csv(cellnames_file, header=None).squeeze()
    dict_length = {}
    dict_perc = {}
    for _, cellname in df_cell.items():
        filename = cellname + "_meta.csv"
        df = pd.read_csv(run_dir + filename)

        wells = df.columns
        for well in wells:
            well_data = df[well].to_numpy()
            dict_length[well] = len(well_data)  # Count all cells
            dict_perc[well] = (well_data == 1).mean()  # Calculate the percentage of '1' cells

    folder = run_dir + "Split/"
    with open(folder + 'count_cells.csv', 'w') as f:
        for key in dict_length.keys():
            f.write("%s, %s\n" % (key, dict_length[key]))
    with open(folder + 'perc_cells.csv', 'w') as f:
        for key in dict_perc.keys():
            f.write("%s, %s\n" % (key, dict_perc[key]))
            
    # Plotting
    plt.figure(figsize=(10, 5))
    plt.bar(dict_length.keys(), dict_length.values())
    plt.title('Count of cells in each well')
    plt.ylabel('Count')
    plt.savefig(folder + 'count_cells.png')

    plt.figure(figsize=(10, 5))
    plt.bar(dict_perc.keys(), dict_perc.values())
    plt.title('Percentage of cells labeled as 1 in each well')
    plt.ylabel('Percentage')
    plt.savefig(folder + 'perc_cells.png')

count_cells(args.cellnames_file, args.data_dir)
