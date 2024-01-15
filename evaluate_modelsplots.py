import os
import argparse 
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='calculates accuracy for models')

parser.add_argument("--run_dir", help="run_dir")
parser.add_argument("--data_dir", help="data_dir")
parser.add_argument("--type", help="wbc or pbmc")   
args = parser.parse_args()

from sklearn.metrics import accuracy_score, f1_score, hamming_loss, precision_score,recall_score


def plot_metrics(y_true, y_pred, run_dir):
    n_classes = len(np.unique(y_true))
    for i in range(n_classes):
        if i == 2:  # Skip missing label
            continue
            
        plt.figure(figsize=(10, 6))
        
        # ROC curve
        fpr, tpr, _ = roc_curve(y_true == i, y_pred == i)
        roc_auc = auc(fpr, tpr)
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic: wbc, label {i}')
        plt.legend(loc="lower right")

        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_true == i, y_pred == i)
        average_precision = average_precision_score(y_true == i, y_pred == i)
        plt.subplot(1, 2, 2)
        plt.step(recall, precision, where='post')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title(f'Precision-Recall curve: wbc, label {i}\nAP={average_precision:.2f}')
        
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, f'wbc_metrics_plots_{i}.png'))
        plt.close()

def calculate_metrics(run_dir, data_dir):
    predictions = np.load(run_dir + 'combined/predictions.npy')
    split = data_dir + 'Split/combined'
    pred_indices = np.loadtxt(Path(split, "test.txt"), dtype=int)

    y_pred = (predictions.squeeze() > 0.5).astype(int)  # apply threshold and convert to int

    meta = pd.read_csv(data_dir + 'combined.csv')
    labels = meta.values
    y_true = labels[pred_indices]

    # Find indices where y_true is not '2'
    not_2_indices = y_true != 2

    # Filter y_true and y_pred using these indices
    y_true_filtered = y_true[not_2_indices]
    y_pred_filtered = y_pred[not_2_indices]

    subset_acc = accuracy_score(y_true_filtered, y_pred_filtered)
    micro_f1 = f1_score(y_true_filtered, y_pred_filtered, average='micro')
    macro_f1 = f1_score(y_true_filtered, y_pred_filtered, average='macro') 
    h_loss = hamming_loss(y_true_filtered, y_pred_filtered)
    precision = precision_score(y_true_filtered, y_pred_filtered)
    recall = recall_score(y_true_filtered, y_pred_filtered)


    # Plot metrics
    plot_metrics(y_true_filtered, y_pred_filtered, run_dir)

    
    return subset_acc, micro_f1, macro_f1, h_loss,precision, recall

def evaluation(run_dir, data_dir, type_cell):
    df = pd.DataFrame(columns=['Name', 'Subset Accuracy', 'Micro F1 Score','Macro F1 Score', 'Hamming Loss','Precision','Recall'])
    result_path = run_dir+'results_'+type_cell+'.csv'

    if os.path.exists(run_dir+'combined/predictions.npy'):
        subset_acc, micro_f1, macro_f1, h_loss, precision, recall = calculate_metrics(run_dir, data_dir)
        dict = {'Name': type_cell , 'Subset Accuracy': subset_acc, 'Micro F1 Score': micro_f1,'Macro F1 Score': macro_f1, 'Hamming Loss': h_loss,'Precision': precision, 'Recall': recall}
        df = df.append(dict, ignore_index = True)
    else:
        print('no predictions for ', type_cell)
            
    df.to_csv(result_path,index=False)
    print('written to', result_path)

evaluation(args.run_dir, args.data_dir, args.type)
