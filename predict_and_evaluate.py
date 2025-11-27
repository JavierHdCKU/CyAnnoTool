#!/usr/bin/env python3
"""
Standalone script to run predictions and evaluation
"""

import sys
sys.path.append('/home/javier/scina_module')

import json
import pandas as pd
import yaml
from pathlib import Path
from cyanno_pipeline.cyanno import CyAnnoClassifier

def main():
    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Set paths
    test_file = "data/test_data.csv"
    model_file = "results/cyanno_model.pkl"
    predictions_output = "results/cyanno_predictions.csv"
    metrics_output = "results/evaluation_metrics.json"
    
    print("Running predictions...")
    
    # Load test data
    test_data = pd.read_csv(test_file)
    print(f"Loaded {len(test_data)} test samples")
    
    # Load model
    classifier = CyAnnoClassifier()
    classifier.load_model(Path(model_file))
    
    # Make predictions
    predictions = classifier.predict(test_data)
    
    # Save predictions
    predictions.to_csv(predictions_output, index=False)
    print(f"Predictions saved: {predictions_output}")
    
    # Evaluate
    evaluation_metrics = classifier.evaluate(test_data)
    
    # Save evaluation metrics
    with open(metrics_output, 'w') as f:
        json.dump(evaluation_metrics, f, indent=2)
    
    print(f"\nEvaluation completed!")
    print(f"F1 Score: {evaluation_metrics['f1_score']:.3f}")
    print(f"Accuracy: {evaluation_metrics['accuracy']:.3f}")
    print(f"Metrics saved: {metrics_output}")

if __name__ == "__main__":
    main()
