import argparse
import pandas as pd
import anndata as ad
import numpy as np
import os
from cyanno_pipeline.cyanno import CyAnnoClassifier

def main():
    """
    Wrapper script to run the CyAnno tool within an OmniBenchmark pipeline.
    This script handles input/output contracts, data validation, and logging.
    """
    parser = argparse.ArgumentParser(description='Run CyAnno classifier on CyTOF data.')
    parser.add_argument('--data', required=True, help='Path to the input AnnData (.h5ad) file.')
    parser.add_argument('--output', required=True, help='Path to the output prediction file (.csv).')
    parser.add_argument('--embedding_key', required=False, default='X_umap', help='Key for embeddings in adata.obsm.')
    parser.add_argument('--annotation_key', required=False, default='cell_type', help='Key for annotations in adata.obs.')
    parser.add_argument('--model_name', required=False, default='cyanno', help='Name of the model to use.')

    args = parser.parse_args()

    print("--- Starting CyAnno Wrapper Script ---")
    print(f"Input data file: {args.data}")
    print(f"Output prediction file: {args.output}")

    # QA Step 1: Validate input file existence and format
    print("\n[QA] Validating input file...")
    if not os.path.exists(args.data):
        print(f"ERROR: Input data file not found at '{args.data}'")
        exit(1)
    if not args.data.endswith('.h5ad'):
        print(f"ERROR: Input file '{args.data}' is not a .h5ad file.")
        exit(1)
    if os.path.getsize(args.data) == 0:
        print(f"ERROR: Input file '{args.data}' is empty.")
        exit(1)
    print("Input file exists and is not empty.")

    # QA Step 2: Load AnnData and validate its contents
    try:
        print("\n[QA] Loading AnnData file...")
        adata = ad.read_h5ad(args.data)
        print("AnnData file loaded successfully.")
        print("AnnData object info:")
        print(adata)
    except Exception as e:
        print(f"ERROR: Failed to load AnnData file '{args.data}'.")
        print(f"Exception: {e}")
        exit(1)

    # QA Step 3: Validate necessary keys and data shapes
    print("\n[QA] Validating AnnData contents...")
    if args.embedding_key not in adata.obsm:
        print(f"ERROR: Embedding key '{args.embedding_key}' not found in adata.obsm.")
        print(f"Available keys in .obsm: {list(adata.obsm.keys())}")
        exit(1)
    
    embeddings = adata.obsm[args.embedding_key]
    print(f"Found embedding key '{args.embedding_key}' with shape: {embeddings.shape}")

    if embeddings.shape[0] != adata.n_obs:
        print(f"ERROR: Shape mismatch! Embeddings shape ({embeddings.shape[0]}) does not match number of observations ({adata.n_obs}).")
        exit(1)

    print("AnnData contents are valid.")

    # Prepare data for the classifier
    print("\n--- Preparing data for CyAnnoClassifier ---")
    try:
        # CyAnnoClassifier expects a DataFrame with embedding columns
        df = pd.DataFrame(embeddings, index=adata.obs.index)
        print(f"Created DataFrame from embeddings. Shape: {df.shape}")
    except Exception as e:
        print(f"ERROR: Failed to create DataFrame from embeddings.")
        print(f"Exception: {e}")
        exit(1)

    # Run the CyAnno classifier
    print("\n--- Running CyAnnoClassifier ---")
    try:
        # Assuming CyAnnoClassifier is deterministic and doesn't need pre-training for this benchmark
        model = CyAnnoClassifier(
            model_name=args.model_name,
            annotation_key=args.annotation_key,
            embedding_key=args.embedding_key
        )
        
        # The `predict` method in the original code seems to expect `adata` directly
        # Let's align with that if it simplifies things. Re-checking cyanno.py...
        # `predict` uses `adata.obsm[self.embedding_key]`. So passing `adata` is correct.
        print("Model initialized. Starting prediction...")
        predictions = model.predict(adata)
        print("Prediction complete.")

    except Exception as e:
        print(f"ERROR: An error occurred during CyAnno model prediction.")
        print(f"Exception: {e}")
        exit(1)

    # QA Step 4: Validate predictions
    print("\n[QA] Validating predictions...")
    if not isinstance(predictions, pd.Series):
        print(f"ERROR: Predictions are not a pandas Series, but {type(predictions)}.")
        exit(1)
    if predictions.shape[0] != adata.n_obs:
        print(f"ERROR: Prediction shape ({predictions.shape[0]}) does not match input observation count ({adata.n_obs}).")
        exit(1)
    print(f"Predictions generated successfully. Shape: {predictions.shape}")

    # Save the output
    print("\n--- Saving output ---")
    try:
        output_df = pd.DataFrame({'prediction': predictions})
        output_df.to_csv(args.output, index=False)
        print(f"Output successfully saved to '{args.output}'")
    except Exception as e:
        print(f"ERROR: Failed to save output to '{args.output}'.")
        print(f"Exception: {e}")
        exit(1)

    # QA Step 5: Verify output file
    print("\n[QA] Verifying output file...")
    if not os.path.exists(args.output):
        print(f"ERROR: Output file was not created at '{args.output}'.")
        exit(1)
    if os.path.getsize(args.output) == 0:
        print(f"ERROR: Output file '{args.output}' is empty.")
        exit(1)
    print("Output file verified.")

    print("\n--- CyAnno Wrapper Script Finished Successfully ---")

if __name__ == '__main__':
    main()
