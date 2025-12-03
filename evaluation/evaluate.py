import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import pandas as pd
import seaborn as sns
import os

def measure_latency(model, device, img_size=224, n=100):
    """
    Measure average inference time per image in milliseconds.
    """
    model.eval()
    x = torch.randn(1, 3, img_size, img_size).to(device)
    # Warmup
    for _ in range(10):
        _ = model(x)
    times = []
    with torch.no_grad():
        for _ in range(n):
            start = time.time()
            _ = model(x)
            end = time.time()
            times.append((end - start) * 1000)  # ms
    return float(np.mean(times)), float(np.std(times))

def per_generator_report(y_true, y_pred, generator_types):
    """
    Generate classification report per generator type.
    generator_types: list of generator labels aligned with y_true/y_pred
    """
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'generator': generator_types})
    reports = {}
    for gen, sub in df.groupby('generator'):
        reports[gen] = classification_report(sub['y_true'], sub['y_pred'], output_dict=True)
    return reports

def plot_roc(y_true, y_scores, save_path=None):
    """
    Plot ROC curve and compute AUC
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0,1],[0,1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()
    return roc_auc

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Plot and save confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()
    return cm
