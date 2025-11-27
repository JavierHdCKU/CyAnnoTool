#!/usr/bin/env python3
import sys
import gzip
import pandas as pd
from pathlib import Path
from cyanno_pipeline.cyanno import CyAnnoClassifier

def main(matrix_path, labels_path, output_path):
    # Load matrix
    with gzip.open(matrix_path, 'rt') as f:
        matrix_df = pd.read_csv(f, sep=',')

    # Load true labels
    with gzip.open(labels_path, 'rt') as f:
        labels_df = pd.read_csv(f, header=None, names=['cell_type'])

    # Combine features and labels
    if len(matrix_df) != len(labels_df):
        raise ValueError("Matrix and labels row count mismatch")

    train_df = pd.concat([matrix_df, labels_df], axis=1).dropna(subset=['cell_type'])

    # Initialize classifier
    markers = list(matrix_df.columns)
    clf = CyAnnoClassifier(markers=markers)

    # Train
    clf.train(train_df, cell_type_column='cell_type')

    # Predict on full dataset
    preds, _ = clf.predict(matrix_df)

    # Save predictions (one label per line)
    pd.Series(preds).to_csv(output_path, index=False, header=False)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: run_cyanno.py <matrix.gz> <true_labels.gz> <output.txt>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
