#!/usr/bin/env python3
"""
Standalone runner for CyAnno pipeline
Test the complete pipeline without Snakemake
"""

import sys
import os
from pathlib import Path
import json

# Add the current directory to Python path
sys.path.append('/home/javier/scina_module')

from cyanno_pipeline.data_generator import generate_mock_data
from cyanno_pipeline.cyanno import CyAnnoClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, classification_report, confusion_matrix

def run_complete_pipeline():
    """Run the complete CyAnno pipeline"""
    
    print("=" * 60)
    print("CyAnno Cell Type Annotation Pipeline")
    print("=" * 60)
    
    # Configuration
    config = {
        'markers': ['FSC-A', 'SSC-A', 'CD3', 'CD4', 'CD8', 'CD14', 'CD19', 'CD45', 'CD56', 'CD20'],
        'normalize': True,
        'random_state': 42,
        'data_dir': '/home/javier/scina_module/data',
        'results_dir': '/home/javier/scina_module/results'
    }
    
    # Create directories
    os.makedirs(config['data_dir'], exist_ok=True)
    os.makedirs(config['results_dir'], exist_ok=True)
    
    # Step 1: Generate mock data (skip if real data exists)
    training_file = Path(config['data_dir']) / 'training_data.csv'
    test_file = Path(config['data_dir']) / 'test_data.csv'
    
    if training_file.exists() and test_file.exists():
        print("\n1. Using existing data files...")
        print(f"   Found training data: {training_file}")
        print(f"   Found test data: {test_file}")
        files = {
            'training_data': str(training_file),
            'test_data': str(test_file)
        }
    else:
        print("\n1. Generating mock cytometry data...")
        files = generate_mock_data(config['data_dir'])
        print(f"   Generated training data: {files['training_data']}")
        print(f"   Generated test data: {files['test_data']}")
    
    # Step 2: Train model
    print("\n2. Training CyAnno model...")
    training_data = pd.read_csv(files['training_data'])
    print(f"   Training samples: {len(training_data)}")
    print(f"   Cell types: {sorted(training_data['cell_type'].unique())}")
    
    classifier = CyAnnoClassifier(
        markers=config['markers'],
        normalize=config['normalize'],
        random_state=config['random_state']
    )
    
    training_metrics = classifier.train(training_data)
    print(f"   Training F1 Score: {training_metrics['f1_score']:.3f}")
    
    # Save model
    model_path = Path(config['results_dir']) / 'cyanno_model.pkl'
    classifier.save_model(model_path)
    
    # Save training metrics
    with open(Path(config['results_dir']) / 'training_metrics.json', 'w') as f:
        json.dump(training_metrics, f, indent=2)
    
    # Step 3: Make predictions
    print("\n3. Making predictions on test data...")
    test_data = pd.read_csv(files['test_data'])
    print(f"   Test samples: {len(test_data)}")
    
    predictions, probabilities = classifier.predict(test_data)
    
    # Prepare output
    output_data = test_data.copy()
    output_data['predicted_cell_type'] = predictions
    output_data['prediction_confidence'] = np.max(probabilities, axis=1)
    
    # Add class probabilities
    for i, cell_type in enumerate(classifier.cell_types):
        output_data[f'prob_{cell_type}'] = probabilities[:, i]
    
    # Save predictions
    predictions_path = Path(config['results_dir']) / 'cyanno_predictions.csv'
    output_data.to_csv(predictions_path, index=False)
    print(f"   Predictions saved: {predictions_path}")
    
    # Step 4: Evaluate results
    print("\n4. Evaluating predictions...")
    true_labels = test_data['cell_type'].values
    
    # Calculate metrics
    f1_weighted = f1_score(true_labels, predictions, average='weighted')
    f1_macro = f1_score(true_labels, predictions, average='macro')
    overall_accuracy = np.mean(predictions == true_labels)
    
    print(f"   Overall Accuracy: {overall_accuracy:.3f}")
    print(f"   F1 Score (weighted): {f1_weighted:.3f}")
    print(f"   F1 Score (macro): {f1_macro:.3f}")
    
    # Per-class metrics
    cell_types = sorted(list(set(true_labels)))
    f1_per_class = f1_score(true_labels, predictions, average=None)
    
    print("\n   Per-class F1 scores:")
    for i, cell_type in enumerate(cell_types):
        print(f"     {cell_type}: {f1_per_class[i]:.3f}")
    
    # Create confusion matrix
    cm = confusion_matrix(true_labels, predictions, labels=cell_types)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=cell_types, yticklabels=cell_types)
    plt.title('CyAnno Confusion Matrix')
    plt.xlabel('Predicted Cell Type')
    plt.ylabel('True Cell Type')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    cm_path = Path(config['results_dir']) / 'confusion_matrix.png'
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Confusion matrix saved: {cm_path}")
    
    # Save evaluation metrics
    evaluation_metrics = {
        'f1_weighted': float(f1_weighted),
        'f1_macro': float(f1_macro),
        'f1_per_class': {cell_types[i]: float(f1_per_class[i]) for i in range(len(cell_types))},
        'overall_accuracy': float(overall_accuracy),
        'confusion_matrix': cm.tolist(),
        'cell_types': cell_types,
        'n_samples': len(true_labels),
        'classification_report': classification_report(true_labels, predictions, output_dict=True)
    }
    
    with open(Path(config['results_dir']) / 'evaluation_metrics.json', 'w') as f:
        json.dump(evaluation_metrics, f, indent=2)
    
    # Step 5: Generate summary report
    print("\n5. Generating summary report...")
    
    print("\n" + "=" * 60)
    print("CYANNO PIPELINE RESULTS SUMMARY")
    print("=" * 60)
    
    print(f"\nDataset Information:")
    print(f"  Training samples: {len(training_data):,}")
    print(f"  Test samples: {len(test_data):,}")
    print(f"  Cell types: {len(cell_types)}")
    print(f"  Markers used: {len(config['markers'])}")
    
    print(f"\nModel Performance:")
    print(f"  Training F1 Score: {training_metrics['f1_score']:.3f}")
    print(f"  Test Accuracy: {overall_accuracy:.3f}")
    print(f"  Test F1 (weighted): {f1_weighted:.3f}")
    print(f"  Test F1 (macro): {f1_macro:.3f}")
    
    print(f"\nPer-Cell Type Performance:")
    for i, cell_type in enumerate(cell_types):
        accuracy_mask = true_labels == cell_type
        if np.sum(accuracy_mask) > 0:
            cell_accuracy = np.mean(predictions[accuracy_mask] == true_labels[accuracy_mask])
            print(f"  {cell_type:15}: F1={f1_per_class[i]:.3f}, Acc={cell_accuracy:.3f}")
    
    print(f"\nOutput Files:")
    print(f"  Model: {model_path}")
    print(f"  Predictions: {predictions_path}")
    print(f"  Confusion Matrix: {cm_path}")
    print(f"  Metrics: {Path(config['results_dir']) / 'evaluation_metrics.json'}")
    
    print(f"\n✅ Pipeline completed successfully!")
    print("=" * 60)
    
    return {
        'training_metrics': training_metrics,
        'evaluation_metrics': evaluation_metrics,
        'files': files,
        'config': config
    }


if __name__ == "__main__":
    # Install required packages if not available
    try:
        import pandas as pd
        import numpy as np
        import sklearn
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("Please install with: pip install pandas numpy scikit-learn matplotlib seaborn")
        sys.exit(1)
    
    # Run the pipeline
    results = run_complete_pipeline()
