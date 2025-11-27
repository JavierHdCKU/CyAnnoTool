#!/usr/bin/env python3
"""
Train CyAnno model
"""

import sys
sys.path.append('/home/javier/scina_module')

import json
from pathlib import Path
import pandas as pd
from cyanno_pipeline.cyanno import CyAnnoClassifier

def main():
    # Get inputs and outputs from snakemake
    training_file = snakemake.input.training
    model_output = snakemake.output.model
    metrics_output = snakemake.output.training_metrics
    
    # Get parameters
    markers = snakemake.params.markers
    normalize = snakemake.params.normalize
    random_state = snakemake.params.random_state
    
    print(f"Training CyAnno model with {len(markers)} markers...")
    print(f"Training data: {training_file}")
    
    # Load training data
    training_data = pd.read_csv(training_file)
    print(f"Loaded {len(training_data)} training samples")
    print(f"Cell types: {sorted(training_data['cell_type'].unique())}")
    
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
    
    print(f"Training completed!")
    print(f"F1 Score: {training_metrics['f1_score']:.3f}")
    print(f"Model saved: {model_output}")

if __name__ == "__main__":
    main()
