#!/usr/bin/env python3
"""
Evaluate CyAnno predictions against ground truth
"""

import sys
sys.path.append('/home/javier/scina_module')

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from pathlib import Path

def main():
    # Get inputs and outputs from snakemake
    predictions_file = snakemake.input.predictions
    ground_truth_file = snakemake.input.ground_truth
    metrics_output = snakemake.output.metrics
    confusion_output = snakemake.output.confusion_matrix
    
    # Get parameters
    truth_column = snakemake.params.ground_truth_column
    
    print(f"Evaluating predictions...")
    print(f"Predictions: {predictions_file}")
    print(f"Ground truth: {ground_truth_file}")
    
    # Load data
    predictions_data = pd.read_csv(predictions_file)
    ground_truth_data = pd.read_csv(ground_truth_file)
    
    # Extract predictions and ground truth
    predicted_labels = predictions_data['predicted_cell_type'].values
    true_labels = ground_truth_data[truth_column].values
    
    # Calculate metrics
    f1_weighted = f1_score(true_labels, predicted_labels, average='weighted')
    f1_macro = f1_score(true_labels, predicted_labels, average='macro')
    f1_per_class = f1_score(true_labels, predicted_labels, average=None)
    
    # Get unique cell types
    cell_types = sorted(list(set(true_labels)))
    
    # Classification report
    class_report = classification_report(true_labels, predicted_labels, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=cell_types)
    
    # Calculate accuracy per cell type
    accuracy_per_class = {}
    for i, cell_type in enumerate(cell_types):
        mask = true_labels == cell_type
        if np.sum(mask) > 0:
            accuracy_per_class[cell_type] = np.mean(predicted_labels[mask] == true_labels[mask])
    
    # Prepare metrics dictionary
    metrics = {
        'f1_weighted': float(f1_weighted),
        'f1_macro': float(f1_macro),
        'f1_per_class': {cell_types[i]: float(f1_per_class[i]) for i in range(len(cell_types))},
        'accuracy_per_class': accuracy_per_class,
        'classification_report': class_report,
        'confusion_matrix': cm.tolist(),
        'cell_types': cell_types,
        'n_samples': len(true_labels),
        'overall_accuracy': float(np.mean(predicted_labels == true_labels))
    }
    
    # Save metrics
    with open(metrics_output, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Create confusion matrix plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=cell_types, yticklabels=cell_types)
    plt.title('CyAnno Confusion Matrix')
    plt.xlabel('Predicted Cell Type')
    plt.ylabel('True Cell Type')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(confusion_output, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Evaluation completed!")
    print(f"Overall F1 Score (weighted): {f1_weighted:.3f}")
    print(f"Overall F1 Score (macro): {f1_macro:.3f}")
    print(f"Overall Accuracy: {metrics['overall_accuracy']:.3f}")
    
    print("\nPer-class F1 scores:")
    for cell_type, f1 in metrics['f1_per_class'].items():
        print(f"  {cell_type}: {f1:.3f}")
    
    print(f"\nMetrics saved: {metrics_output}")
    print(f"Confusion matrix saved: {confusion_output}")

if __name__ == "__main__":
    main()
