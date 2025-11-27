# CyAnno Cell Type Annotation Pipeline

A modular cytometry cell type annotation pipeline, with model training and prediction capabilities.

## Overview

This project implements a Random Forest-based cell type annotation pipeline for flow/mass cytometry data. It provides a framework for automated cell type classification.

The pipeline uses a Random Forest classifier trained on marker expression data to predict cell types for unlabeled cells.

## Features

- ✅ **Machine Learning Classification**: Random Forest-based cell type prediction
- ✅ **Data Preprocessing**: Automated normalization and feature scaling
- ✅ **Flexible Input**: Handles CSV data.

## Quick Start

### 1. Setup Environment
```bash
# Create Python virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\\Scripts\\activate  # Windows

# Install requirements
pip install -r requirements.txt
```

### 2. Run Pipeline
The main script `run_cyanno.py` trains a model on a given dataset and then runs predictions on that same dataset.

```bash
# Usage: run_cyanno.py <matrix.gz> <true_labels.gz> <output.txt>
# Example with placeholder data:
python run_cyanno.py data/matrix.csv.gz data/labels.csv.gz results/predictions.txt
```
*Note: You will need to provide your own gzipped matrix and labels files in CSV format.*

## Project Structure

```
CyAnno_module/
├── cyanno_pipeline/           # Core CyAnno implementation
│   ├── __init__.py
│   └── cyanno.py             # Main Random Forest classifier
├── run_cyanno.py         # Main pipeline runner
├── setup.py
├── test.py
└── requirements.txt        # Python dependencies
```

## Programmatic Usage

```python
from cyanno_pipeline.cyanno import CyAnnoClassifier
import pandas as pd

# Initialize classifier with your markers
markers = ['FSC-A', 'SSC-A', 'CD3', 'CD4', 'CD8', 'CD14', 'CD19', 'CD20', 'CD45', 'CD56']
classifier = CyAnnoClassifier(
    markers=markers,
    normalize=True,
    random_state=42
)

# Train on your data (assuming you load it into a pandas DataFrame)
# The training data needs a 'cell_type' column with the labels.
training_data = pd.read_csv('path/to/your/training_data.csv')
metrics = classifier.train(training_data, cell_type_column='cell_type')
print(f"Training F1: {metrics['f1_score']:.3f}")

# Save the model
classifier.save_model('my_model.pkl')

# Load model for prediction
classifier.load_model('my_model.pkl')

# Make predictions on new data
test_data = pd.read_csv('path/to/your/test_data.csv')
predictions, _ = classifier.predict(test_data)

# Evaluate performance if test data has labels
evaluation = classifier.evaluate(test_data)
print(f"Test F1 Score: {evaluation['f1_weighted']:.3f}")
```

## Dependencies

- Python ≥ 3.8
- pandas ≥ 1.3.0
- numpy ≥ 1.20.0
- scikit-learn ≥ 1.0.0
- matplotlib >= 3.3.0
- seaborn >= 0.11.0
- pyyaml>=5.4.0

## License

This project is licensed under the MIT License.
