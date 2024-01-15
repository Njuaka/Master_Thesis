import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import tensorflow as tf

parser = argparse.ArgumentParser(description='calculates accuracy for models')

parser.add_argument("--run_dir", help="run_dir")
parser.add_argument("--data_dir", help="data_dir")
parser.add_argument("--type", help="wbc or pbmc")   
args = parser.parse_args()

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

    # Convert numpy arrays to tensors
    y_true_tensor = tf.convert_to_tensor(y_true_filtered, dtype=tf.int32)
    y_pred_tensor = tf.convert_to_tensor(y_pred_filtered, dtype=tf.int32)
    
    # Use TensorFlow metrics
    subset_acc = tf.keras.metrics.Accuracy()(y_true_tensor, y_pred_tensor).numpy()
    micro_f1 = tf.keras.metrics.Mean()(tf.keras.metrics.Precision()(y_true_tensor, y_pred_tensor) * tf.keras.metrics.Recall()(y_true_tensor, y_pred_tensor) / (tf.keras.metrics.Precision()(y_true_tensor, y_pred_tensor) + tf.keras.metrics.Recall()(y_true_tensor, y_pred_tensor))).numpy()
    macro_f1 = tf.reduce_mean(tf.keras.metrics.F1Score(num_classes=len(np.unique(y_true_filtered)), average="macro")(y_true_tensor, tf.one_hot(y_pred_tensor, depth=len(np.unique(y_true_filtered))))).numpy()
    h_loss = 1 - subset_acc # Direct computation as hamming_loss = 1 - accuracy for binary classification
    precision = tf.keras.metrics.Precision()(y_true_tensor, y_pred_tensor).numpy()
    recall = tf.keras.metrics.Recall()(y_true_tensor, y_pred_tensor).numpy()

    return subset_acc, micro_f1, macro_f1, h_loss, precision, recall

def evaluation(run_dir, data_dir, type_cell):
    df = pd.DataFrame(columns=['Name', 'Subset Accuracy', 'Micro F1 Score','Macro F1 Score', 'Hamming Loss'])
    result_path = run_dir+'results_'+type_cell+'.csv'

    if os.path.exists(run_dir+'combined/predictions.npy'):
        subset_acc, micro_f1, macro_f1, h_loss, precision, recall = calculate_metrics(run_dir, data_dir)
        dict = {'Name': type_cell, 'Subset Accuracy': subset_acc, 'Micro F1 Score': micro_f1,'Macro F1 Score': macro_f1, 'Hamming Loss': h_loss, 'Precision': precision, 'Recall': recall}
        df = df.append(dict, ignore_index=True)
    else:
        print('no predictions for ', type_cell)
            
    df.to_csv(result_path, index=False)
    print('written to', result_path)

evaluation(args.run_dir, args.data_dir, args.type)
