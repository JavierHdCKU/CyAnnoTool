"""
Snakemake workflow for CyAnno cell type annotation pipeline
"""

import pandas as pd
from pathlib import Path

# Configuration
configfile: "config/config.yaml"

# Define the final output
rule all:
    input:
        "results/cyanno_predictions.csv",
        "results/evaluation_metrics.json",
        "results/cyanno_model.pkl",
        "results/performance_report.html"

# Generate mock data
rule generate_mock_data:
    output:
        training="data/training_data.csv",
        test="data/test_data.csv",
        unlabeled="data/unlabeled_data.csv",
        metadata="data/metadata.txt"
    script:
        "scripts/generate_data.py"

# Train CyAnno model
rule train_cyanno:
    input:
        training="data/training_data.csv"
    output:
        model="results/cyanno_model.pkl",
        training_metrics="results/training_metrics.json"
    params:
        markers=config["markers"],
        normalize=config["normalize"],
        random_state=config["random_state"]
    script:
        "scripts/train_model.py"

# Predict cell types
rule predict_cell_types:
    input:
        model="results/cyanno_model.pkl",
        test_data="data/test_data.csv"
    output:
        predictions="results/cyanno_predictions.csv"
    script:
        "scripts/predict.py"

# Evaluate predictions
rule evaluate_predictions:
    input:
        predictions="results/cyanno_predictions.csv",
        ground_truth="data/test_data.csv"
    output:
        metrics="results/evaluation_metrics.json",
        confusion_matrix="results/confusion_matrix.png"
    params:
        ground_truth_column=config["ground_truth_column"]
    script:
        "scripts/evaluate.py"

# Generate performance report
rule generate_report:
    input:
        predictions="results/cyanno_predictions.csv",
        metrics="results/evaluation_metrics.json",
        training_metrics="results/training_metrics.json",
        confusion_matrix="results/confusion_matrix.png"
    output:
        report="results/performance_report.html"
    script:
        "scripts/generate_report.py"
