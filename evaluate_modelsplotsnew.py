import os
import argparse 
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, accuracy_score, f1_score, hamming_loss, precision_score, recall_score
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='calculates accuracy for models')

parser.add_argument("--run_dir", help="run_dir")
parser.add_argument("--data_dir", help="data_dir")
parser.add_argument("--type", help="wbc or pbmc")   
args = parser.parse_args()

def plot_metrics(y_true, y_pred, run_dir, label):
    plt.figure(figsize=(10, 6))
        
    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic: wbc, label {label}')
    plt.legend(loc="lower right")

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    average_precision = average_precision_score(y_true, y_pred)
    plt.subplot(1, 2, 2)
    plt.step(recall, precision, where='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f'Precision-Recall curve: wbc, label {label}\nAP={average_precision:.2f}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, f'wbc_metrics_plots_{label}.png'))
    plt.close()

def calculate_metrics(run_dir, data_dir):
    predictions = np.load(run_dir + 'combined/predictions.npy')
    split = data_dir + 'Split/combined'
    pred_indices = np.loadtxt(Path(split, "test.txt"), dtype=int)

    y_pred = (predictions.squeeze() > 0.5).astype(int)

    meta = pd.read_csv(data_dir + 'combined.csv')
    column_names = meta.columns
    labels = meta.values
    y_true = labels[pred_indices]

    overall_subset_acc = []
    overall_micro_f1 = []
    overall_macro_f1 = []
    overall_h_loss = []
    overall_precision = []
    overall_recall = []

    for idx, label in enumerate(column_names):
        y_true_label = y_true[:, idx]
        y_pred_label = y_pred[:, idx]

        valid_indices = y_true_label != 2
        y_true_label = y_true_label[valid_indices]
        y_pred_label = y_pred_label[valid_indices]

        subset_acc = accuracy_score(y_true_label, y_pred_label)
        micro_f1 = f1_score(y_true_label, y_pred_label, average='micro')
        macro_f1 = f1_score(y_true_label, y_pred_label, average='macro') 
        h_loss = hamming_loss(y_true_label, y_pred_label)
        precision = precision_score(y_true_label, y_pred_label)
        recall = recall_score(y_true_label, y_pred_label)

        overall_subset_acc.append(subset_acc)
        overall_micro_f1.append(micro_f1)
        overall_macro_f1.append(macro_f1)
        overall_h_loss.append(h_loss)
        overall_precision.append(precision)
        overall_recall.append(recall)

        plot_metrics(y_true_label, y_pred_label, run_dir, label)

    avg_subset_acc = np.mean(overall_subset_acc)
    avg_micro_f1 = np.mean(overall_micro_f1)
    avg_macro_f1 = np.mean(overall_macro_f1)
    avg_h_loss = np.mean(overall_h_loss)
    avg_precision = np.mean(overall_precision)
    avg_recall = np.mean(overall_recall)

    return avg_subset_acc, avg_micro_f1, avg_macro_f1, avg_h_loss, avg_precision, avg_recall

def evaluation(run_dir, data_dir, type_cell):
    df = pd.DataFrame(columns=['Name', 'Subset Accuracy', 'Micro F1 Score','Macro F1 Score', 'Hamming Loss','Precision','Recall'])
    result_path = run_dir+'results_'+type_cell+'.csv'

    if os.path.exists(run_dir+'combined/predictions.npy'):
        avg_subset_acc, avg_micro_f1, avg_macro_f1, avg_h_loss, avg_precision, avg_recall = calculate_metrics(run_dir, data_dir)
        dict = {
            'Name': type_cell , 
            'Subset Accuracy': avg_subset_acc, 
            'Micro F1 Score': avg_micro_f1,
            'Macro F1 Score': avg_macro_f1, 
            'Hamming Loss': avg_h_loss,
            'Precision': avg_precision, 
            'Recall': avg_recall
        }
        df = df.append(dict, ignore_index = True)
    else:
        print('no predictions for ', type_cell)
            
    df.to_csv(result_path,index=False)
    print('written to', result_path)

evaluation(args.run_dir, args.data_dir, args.type)
