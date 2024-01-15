import os
import argparse 
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, hamming_loss
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser(description='calculates accuracy for models')

parser.add_argument("--run_dir", help="run_dir")
parser.add_argument("--data_dir", help="data_dir")
parser.add_argument("--type", help="wbc or pbmc") 
parser.add_argument("--cellnames_file", help="filepath to list")  
args = parser.parse_args()

def plot_metrics(y_true, y_pred, cell_name, run_dir):
    fig, ax = plt.subplots(1, 3, figsize=(20, 6))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', ax=ax[0])
    ax[0].set_title(f'Confusion Matrix: {cell_name}')
    ax[0].set_xlabel('Predicted')
    ax[0].set_ylabel('True')
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    ax[1].plot(fpr, tpr, color='darkorange', label=f'ROC curve (area = {roc_auc:.2f})')
    ax[1].plot([0, 1], [0, 1], color='navy', linestyle='--')
    ax[1].set_xlim([0.0, 1.0])
    ax[1].set_ylim([0.0, 1.05])
    ax[1].set_xlabel('False Positive Rate')
    ax[1].set_ylabel('True Positive Rate')
    ax[1].set_title(f'Receiver Operating Characteristic: {cell_name}')
    ax[1].legend(loc="lower right")

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    ax[2].plot(recall, precision, color='blue', label=f'Precision-Recall curve')
    ax[2].set_xlim([0.0, 1.0])
    ax[2].set_ylim([0.0, 1.05])
    ax[2].set_xlabel('Recall')
    ax[2].set_ylabel('Precision')
    ax[2].set_title(f'Precision-Recall curve: {cell_name}')
    ax[2].legend(loc="lower right")
    
    plt.tight_layout()

    # Save the figure
    fig.savefig(os.path.join(run_dir, f'{cell_name}_metrics_plots.png'))

def calculate_metrics(run_dir, data_dir, cell_name):
    predictions = np.load(run_dir + cell_name + '/predictions.npy')
    split = data_dir + 'Split/' + cell_name
    pred_indices = np.loadtxt(Path(split, "test.txt"), dtype=int)

    y_pred = (predictions.squeeze() > 0.5).astype(int)  # apply threshold and convert to int

    meta = pd.read_csv(data_dir + cell_name + '_meta.csv')
    labels = meta.values
    y_true = labels[pred_indices]

    # Find indices where y_true is not '2'
    not_2_indices = y_true != 2

    # Filter y_true and y_pred using these indices
    y_true_filtered = y_true[not_2_indices]
    y_pred_filtered = y_pred[not_2_indices]

    subset_acc = accuracy_score(y_true_filtered, y_pred_filtered)
    micro_f1 = f1_score(y_true_filtered, y_pred_filtered, average='micro')
    h_loss = hamming_loss(y_true_filtered, y_pred_filtered)

    # Plot metrics
    plot_metrics(y_true_filtered, y_pred_filtered, cell_name, run_dir)

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