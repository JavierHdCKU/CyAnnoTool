#!/usr/bin/env python3
"""
Standalone script to train CyAnno model
"""

import sys
sys.path.append('/home/javier/scina_module')

import json
from pathlib import Path
import pandas as pd
import yaml
from cyanno_pipeline.cyanno import CyAnnoClassifier

def main():
    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Set paths
    training_file = "data/training_data.csv"
    model_output = "results/cyanno_model.pkl"
    metrics_output = "results/training_metrics.json"
    
    # Get parameters from config
    markers = config['markers']
    normalize = config['normalize']
    random_state = config['random_state']
    
    print(f"Training CyAnno model with {len(markers)} markers...")
    print(f"Training data: {training_file}")
    
    # Load training data
    training_data = pd.read_csv(training_file)
    print(f"Loaded {len(training_data)} training samples")
    print(f"Cell types: {sorted(training_data['cell_type'].unique())}")
    
    # Check data distribution
    print("\nData distribution:")
    print(training_data['cell_type'].value_counts())
    
    # Initialize and train classifier
    classifier = CyAnnoClassifier(
        markers=markers,
        normalize=normalize,
        random_state=random_state
    )
    
    # Train model
    training_metrics = classifier.train(training_data)
    
    # Save model
    classifier.save_model(Path(model_output))
    
    # Save training metrics
    with open(metrics_output, 'w') as f:
        json.dump(training_metrics, f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"F1 Score: {training_metrics['f1_score']:.3f}")
    print(f"Model saved: {model_output}")

if __name__ == "__main__":
    main()
