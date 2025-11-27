
import argparse
import pandas as pd
from pathlib import Path
import sys
import os

# Add the script's directory to the Python path to find 'cyanno_pipeline'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cyanno_pipeline.cyanno import CyAnnoClassifier

def main():
    """
    Wrapper script to run CyAnno within the OmniBenchmark framework.
    """
    # 1. Set up argument parser
    parser = argparse.ArgumentParser(description="CyAnno wrapper for OmniBenchmark")
    parser.add_argument("--data.matrix", required=True, help="Path to the input data matrix (.matrix.gz)")
    parser.add_argument("--data.true_labels", required=True, help="Path to the true labels file (.true_labels.gz)")
    parser.add_argument("--output_dir", required=True, help="Directory to save the output predictions")
    parser.add_argument("--name", required=True, help="Name of the module run (provided by OmniBenchmark)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    print("--- Running CyAnno Benchmark Wrapper ---")
    print(f"Data matrix: {getattr(args, 'data.matrix')}")
    print(f"True labels: {getattr(args, 'data.true_labels')}")
    print(f"Output directory: {args.output_dir}")
    
    # Ensure output directory exists
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 2. Load and prepare data
    print("\nLoading and preparing data...")
    try:
        # Load matrix data: gzipped, space-separated, no header
        matrix = pd.read_csv(getattr(args, 'data.matrix'), sep=' ', header=None, compression='gzip', low_memory=False)
        
        # Load labels data
        labels = pd.read_csv(getattr(args, 'data.true_labels'), header=None, compression='gzip')
        labels.columns = ['cell_type']
        
        # Generate generic marker names as CyAnno expects named columns
        num_markers = matrix.shape[1]
        marker_names = [f'Marker_{i+1}' for i in range(num_markers)]
        matrix.columns = marker_names
        
        # Combine into a single DataFrame for training
        training_data = pd.concat([matrix, labels], axis=1)
        print(f"Successfully loaded {len(training_data)} cells and {num_markers} markers.")

    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    # 3. Train model and predict
    print("\nTraining model and making predictions...")
    classifier = CyAnnoClassifier(
        markers=marker_names,
        normalize=True,
        random_state=args.seed
    )
    
    # The benchmark contract is to predict on the input data
    # We train and predict on the same data here.
    classifier.train(training_data)
    predictions, _ = classifier.predict(training_data)
    
    # 4. Save predictions
    output_file = output_path / "predicted_labels.txt"
    print(f"\nSaving predictions to {output_file}...")
    pd.DataFrame(predictions).to_csv(output_file, index=False, header=False)
    
    print("--- CyAnno run finished successfully! ---")

if __name__ == "__main__":
    main()
