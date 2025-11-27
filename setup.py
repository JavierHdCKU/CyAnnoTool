import argparse
import pandas as pd
import anndata as ad
import numpy as np
import os
import gzip
from cyanno_pipeline.cyanno import CyAnnoClassifier

def main():
    """
    Wrapper script to run the CyAnno tool within an OmniBenchmark pipeline.
    This script handles input/output contracts, data validation, and logging.
    """
    parser = argparse.ArgumentParser(description='Run CyAnno classifier on CyTOF data.')
    parser.add_argument('--matrix', required=True, help='Path to the input matrix file (gzipped TSV).')
    parser.add_argument('--labels', required=True, help='Path to the input labels file (gzipped TSV).')
    parser.add_argument('--output', required=True, help='Path to the output prediction file (.tsv).')
    parser.add_argument('--embedding_key', required=False, default='X_umap', help='Key for embeddings in adata.obsm.')
    parser.add_argument('--annotation_key', required=False, default='cell_type', help='Key for annotations in adata.obs.')
    parser.add_argument('--model_name', required=False, default='cyanno', help='Name of the model to use.')

    args = parser.parse_args()

    print("--- Starting CyAnno Wrapper Script ---")
    print(f"Input matrix file: {args.matrix}")
    print(f"Input labels file: {args.labels}")
    print(f"Output prediction file: {args.output}")

    # QA Step 1: Validate input files
    print("\n[QA] Validating input files...")
    for f_path in [args.matrix, args.labels]:
        if not os.path.exists(f_path):
            print(f"ERROR: Input file not found at '{f_path}'")
            exit(1)
        if os.path.getsize(f_path) == 0:
            print(f"ERROR: Input file '{f_path}' is empty.")
            exit(1)
    print("Input files exist and are not empty.")

    # QA Step 2: Load data and create AnnData object
    try:
        print("\n[QA] Loading data and creating AnnData object...")
        with gzip.open(args.matrix, 'rt') as f:
            matrix = pd.read_csv(f, sep='\t', index_col=0)
        print(f"Loaded matrix with shape: {matrix.shape}")

        with gzip.open(args.labels, 'rt') as f:
            labels = pd.read_csv(f, sep='\t', index_col=0)
        print(f"Loaded labels with shape: {labels.shape}")

        if not matrix.index.equals(labels.index):
             print("ERROR: Matrix and labels indices do not match.")
             exit(1)

        adata = ad.AnnData(X=matrix.values, obs=labels)
        adata.var_names = matrix.columns
        print("AnnData object created successfully.")
        print("AnnData object info:")
        print(adata)

    except Exception as e:
        print(f"ERROR: Failed to load data or create AnnData object.")
        print(f"Exception: {e}")
        exit(1)

    # QA Step 3: Run UMAP for embeddings as they are not pre-calculated
    print("\n[QA] Running UMAP for embeddings...")
    try:
        import umap
        reducer = umap.UMAP()
        embedding = reducer.fit_transform(adata.X)
        adata.obsm[args.embedding_key] = embedding
        print(f"UMAP embedding created with shape: {embedding.shape}")
    except Exception as e:
        print(f"ERROR: Failed to run UMAP.")
        print(f"Exception: {e}")
        exit(1)


    # Prepare data for the classifier
    print("\n--- Preparing data for CyAnnoClassifier ---")
    # No extra preparation needed as we pass the full adata object

    # Run the CyAnno classifier
    print("\n--- Running CyAnnoClassifier ---")
    try:
        model = CyAnnoClassifier(
            model_name=args.model_name,
            annotation_key=args.annotation_key,
            embedding_key=args.embedding_key
        )
        
        print("Model initialized. Starting prediction...")
        predictions = model.predict(adata)
        print("Prediction complete.")

    except Exception as e:
        print(f"ERROR: An error occurred during CyAnno model prediction.")
        print(f"Exception: {e}")
        exit(1)

    # QA Step 4: Validate and save predictions
    print("\n[QA] Validating and saving predictions...")
    if predictions is None:
        print("ERROR: Prediction function returned None.")
        exit(1)
    
    if not isinstance(predictions, pd.DataFrame) and not isinstance(predictions, pd.Series):
        print(f"ERROR: Predictions are not a pandas DataFrame or Series (got {type(predictions)}).")
        exit(1)

    print(f"Predictions have shape: {predictions.shape}")
    
    try:
        # Ensure output is a DataFrame with the correct index
        if isinstance(predictions, pd.Series):
            predictions = predictions.to_frame(name='prediction')
        
        predictions.to_csv(args.output, sep='\t', index=True)
        print(f"Predictions successfully saved to '{args.output}'")
    except Exception as e:
        print(f"ERROR: Failed to save predictions to '{args.output}'.")
        print(f"Exception: {e}")
        exit(1)

    # Final QA: Check if output file was created and is not empty
    if not os.path.exists(args.output) or os.path.getsize(args.output) == 0:
        print(f"ERROR: Output file '{args.output}' was not created or is empty.")
        exit(1)

    print("\n--- CyAnno Wrapper Script Finished Successfully ---")

if __name__ == '__main__':
    main()
