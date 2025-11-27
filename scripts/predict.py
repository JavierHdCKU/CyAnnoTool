#!/usr/bin/env python3
"""
Make predictions using trained CyAnno model
"""

import sys
sys.path.append('/home/javier/scina_module')

import numpy as np
import pandas as pd
from pathlib import Path
from cyanno_pipeline.cyanno import CyAnnoClassifier

def main():
    # Get inputs and outputs from snakemake
    model_file = snakemake.input.model
    test_file = snakemake.input.test_data
    output_file = snakemake.output.predictions
    
    print(f"Loading model from: {model_file}")
    print(f"Test data: {test_file}")
    
    # Load model
    classifier = CyAnnoClassifier(markers=[])  # markers will be loaded from file
    classifier.load_model(Path(model_file))
    
    # Load test data
    test_data = pd.read_csv(test_file)
    print(f"Loaded {len(test_data)} test samples")
    
    # Make predictions
    predictions, probabilities = classifier.predict(test_data)
    
    # Prepare output data
    output_data = test_data.copy()
    output_data['predicted_cell_type'] = predictions
    output_data['prediction_confidence'] = np.max(probabilities, axis=1)
    
    # Add individual class probabilities
    for i, cell_type in enumerate(classifier.cell_types):
        output_data[f'prob_{cell_type}'] = probabilities[:, i]
    
    # Save predictions
    output_data.to_csv(output_file, index=False)
    
    print(f"Predictions saved to: {output_file}")
    print(f"Predicted {len(predictions)} samples")
    
    # Show prediction summary
    prediction_counts = pd.Series(predictions).value_counts()
    print("\nPrediction summary:")
    for cell_type, count in prediction_counts.items():
        print(f"  {cell_type}: {count} cells ({count/len(predictions)*100:.1f}%)")

if __name__ == "__main__":
    main()
