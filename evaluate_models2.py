import os
import argparse 
import pandas as pd
import numpy as np
from pathlib import Path
parser = argparse.ArgumentParser(description='calculates accuracy for models')

parser.add_argument("--run_dir", help="run_dir")
parser.add_argument("--data_dir", help="data_dir")
parser.add_argument("--type", help="wbc or pbmc") 
parser.add_argument("--cellnames_file", help="filepath to list")  
args = parser.parse_args()

from sklearn.metrics import accuracy_score, f1_score, hamming_loss

def calculate_metrics(run_dir, data_dir, cell_name):
    predictions = np.load(run_dir + cell_name + '/predictions.npy')
    print("Shape of predictions: ", predictions.shape)

    split = data_dir + 'Split/' + cell_name
    pred_indices = np.loadtxt(Path(split, "test.txt"), dtype=int)
    print("Shape of pred_indices: ", pred_indices.shape)

    y_pred = (predictions.squeeze() > 0.5).astype(int)  # apply threshold and convert to int
    y_pred_filtered = y_pred[y_pred != 2]  # Filter out '2' labels in predicted labels
    print("Shape of y_pred_filtered: ", y_pred_filtered.shape)

    meta = pd.read_csv(data_dir + cell_name + '_meta.csv')
    labels = meta.values
    y_true = labels[pred_indices]
    y_true_filtered = y_true[y_true != 2]  # Filter out '2' labels in true labels
    print("Shape of y_true_filtered: ", y_true_filtered.shape)

    subset_acc = accuracy_score(y_true_filtered, y_pred_filtered)
    micro_f1 = f1_score(y_true_filtered, y_pred_filtered, average='micro')
    h_loss = hamming_loss(y_true_filtered, y_pred_filtered)

    print(cell_name, subset_acc)
    return subset_acc, micro_f1, h_loss

def evaluation(run_dir, data_dir, type_cell, cellnames_file):
    df = pd.DataFrame(columns=['Name', 'Subset Accuracy', 'Micro F1 Score', 'Hamming Loss'])
    result_path = run_dir+'results_'+type_cell+'.csv'
    df_cell = pd.read_csv(cellnames_file,header=None).squeeze()
    
    for _, cellname in df_cell.items():
        if os.path.exists(run_dir+cellname+'/predictions.npy'):
            subset_acc, micro_f1, h_loss = calculate_metrics(run_dir, data_dir, cellname)
            dict = {'Name': cellname , 'Subset Accuracy': subset_acc, 'Micro F1 Score': micro_f1, 'Hamming Loss': h_loss}
            df = df.append(dict, ignore_index = True)
        else:
            print('no predictions for ', cellname)
            
    df.to_csv(result_path,index=False)
    print('written to', result_path)

evaluation(args.run_dir, args.data_dir, args.type, args.cellnames_file)
