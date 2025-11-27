# CyAnno Cell Type Annotation Pipeline

A modular cytometry cell type annotation pipeline suitable for Snakemake workflows, with mock data generation, model training, prediction, and evaluation capabilities.

## Overview

This project implements a Random Forest-based cell type annotation pipeline for flow/mass cytometry data. It provides a complete framework for automated cell type classification with the following key features:

- **Modular Python package** (`cyanno_pipeline/`) with the core classifier
- **Snakemake workflow** for reproducible pipeline execution  
- **Mock data generation** for testing and development
- **Evaluation metrics** including F1 scores and confusion matrices
- **Easy integration** with real cytometry datasets
- **Robust train/test separation** to prevent data leakage

## What is CyAnno?

CyAnno (Cytometry Annotator) is a machine learning approach for automated cell type annotation in cytometry data. It uses surface marker expression patterns to classify immune cell types such as:

- **B cells** (CD19+, CD20+ B lymphocytes)
- **T cells CD4+** (CD3+, CD4+ helper T cells) 
- **T cells CD8+** (CD3+, CD8+ cytotoxic T cells)
- **NK cells** (CD56+ natural killer cells)
- **Monocytes** (CD14+ monocytes/macrophages)

The pipeline uses a Random Forest classifier trained on marker expression data to predict cell types for unlabeled cells. This approach is particularly useful for:

1. **Automated gating** - Replace manual gating with ML-based classification
2. **Consistent annotation** - Standardize cell type definitions across experiments
3. **High-throughput analysis** - Process large datasets efficiently
4. **Quality control** - Validate manual annotations with automated predictions

## Features

- ✅ **Machine Learning Classification**: Random Forest-based cell type prediction
- ✅ **Performance Evaluation**: Comprehensive F1 scores and accuracy metrics
- ✅ **Data Preprocessing**: Automated normalization and feature scaling
- ✅ **Visualization**: Confusion matrices and performance plots
- ✅ **Flexible Input**: Easy substitution of datasets
- ✅ **Reproducible Results**: Fixed random seeds and version control

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

### 2. Run Complete Pipeline
```bash
# Generate mock data, train model, and evaluate
python run_pipeline.py

# Or analyze existing data quality
python analyze_data.py
```

### 3. Run with Snakemake
```bash
# Run the full Snakemake workflow
snakemake --cores 4
```

## Project Structure

```
scina_module/
├── cyanno_pipeline/           # Core CyAnno implementation
│   ├── __init__.py
│   ├── cyanno.py             # Main Random Forest classifier
│   └── data_generator.py     # Mock cytometry data generation
├── scripts/                   # Snakemake workflow scripts
│   ├── generate_data.py      # Data generation
│   ├── train_model.py        # Model training  
│   ├── predict.py           # Prediction
│   ├── evaluate.py          # Evaluation
│   └── generate_report.py   # Reporting
├── config/
│   └── config.yaml          # Pipeline configuration
├── data/                    # Input data
│   ├── training_data.csv    # Training dataset with ground truth
│   ├── test_data.csv        # Test dataset with ground truth  
│   ├── unlabeled_data.csv   # Unlabeled data for prediction
│   └── metadata.txt         # Dataset metadata
├── results/                 # Pipeline outputs
│   ├── cyanno_model.pkl     # Trained Random Forest model
│   ├── cyanno_predictions.csv # Cell type predictions
│   ├── evaluation_metrics.json # F1 scores and accuracy
│   ├── training_metrics.json # Training performance
│   └── confusion_matrix.png # Classification visualization
├── Snakefile               # Snakemake workflow definition
├── run_pipeline.py         # Complete standalone pipeline runner
├── analyze_data.py         # Data quality analysis tool
└── requirements.txt        # Python dependencies
```

## Substituting Real Data

### 1. Understanding the Data Format

The pipeline expects CSV files with the following structure:

```csv
FSC-A,SSC-A,CD3,CD4,CD8,CD14,CD19,CD45,CD56,CD20,cell_type,sample_id
55767.92,48968.28,146.01,762.18,320.54,9604.05,219.40,7418.89,478.66,39.19,Monocytes,sample_1
53093.74,18188.83,7799.28,4875.57,498.16,83.56,212.21,9362.89,174.23,585.71,T_cells_CD4,sample_1
...
```

**Required columns:**
- **Marker columns**: Fluorescence intensity values for each marker (e.g., CD3, CD4, CD8, etc.)
- **cell_type**: Ground truth cell type annotations (for training and evaluation)
- **sample_id**: Sample/patient identifier (optional but recommended)

**Marker columns in this pipeline:**
- `FSC-A`, `SSC-A`: Forward and side scatter (cell size/granularity)
- `CD3`: T cell marker
- `CD4`: Helper T cell marker
- `CD8`: Cytotoxic T cell marker  
- `CD14`: Monocyte marker
- `CD19`, `CD20`: B cell markers
- `CD45`: Pan-leukocyte marker
- `CD56`: NK cell marker

### 2. Steps to Use Your Real Data

#### Step 1: Prepare Your Training Data
```bash
# Replace data/training_data.csv with your annotated training data
# Ensure it has the same column structure:
# - All marker columns you want to use
# - cell_type column with manual annotations  
# - sample_id column (optional)
```

Example cell types you can annotate:
- `B_cells`: CD19+ and/or CD20+ cells
- `T_cells_CD4`: CD3+ CD4+ CD8- cells
- `T_cells_CD8`: CD3+ CD8+ CD4- cells  
- `NK_cells`: CD3- CD56+ cells
- `Monocytes`: CD14+ cells
- Add your own cell types as needed

#### Step 2: Prepare Your Test Data
```bash
# Replace data/test_data.csv with your held-out test data
# Include cell_type column for evaluation
# Use different samples than training (avoid data leakage)
```

#### Step 3: Update Configuration
Edit `config/config.yaml` to match your data:

```yaml
# Update markers to match your panel
markers:
  - "FSC-A"
  - "SSC-A"  
  - "CD3"
  - "CD4"
  - "CD8"
  - "CD14"
  - "CD19"
  - "CD20"
  - "CD45"
  - "CD56"
  # Add or remove markers as needed

# Ensure this matches your ground truth column name
ground_truth_column: "cell_type"

# Model parameters (tune as needed)
normalize: true
random_state: 42
n_estimators: 100
```

#### Step 4: Run with Your Data
```bash
# Run the complete pipeline with your data
python run_pipeline.py

# Or use Snakemake (skip generate_data rule)
snakemake evaluate_model --cores 4
```

### 3. Data Quality Validation

Use the included data quality checker to validate your data:

```bash
python analyze_data.py
```

This will check for:
- **Data leakage**: Overlapping samples between train/test
- **Distribution balance**: Class imbalances that might affect performance
- **Marker separation**: Whether cell types are distinguishable
- **Prediction confidence**: Whether results are realistic

### 4. Integrating Different Marker Panels

If your cytometry panel uses different markers:

1. **Update marker list** in `config/config.yaml`
2. **Ensure column names match** between config and data files
3. **Consider marker relevance** - remove uninformative markers
4. **Test performance** - some marker combinations work better than others

Example for a different panel:
```yaml
markers:
  - "FSC-A"
  - "SSC-A"
  - "CD3"
  - "CD4" 
  - "CD8"
  - "CD25"    # Activation marker
  - "CD127"   # Memory marker
  - "FOXP3"   # Regulatory T cell marker
  - "CD19"
  - "IgD"     # Naive B cell marker
```

### 5. Handling Different Cell Types

To annotate different cell types than the default 5:

1. **Update your ground truth annotations** in training/test data
2. **Ensure consistent naming** (e.g., "Tregs" not "T_regulatory")
3. **Provide sufficient examples** (≥50-100 cells per type recommended)
4. **Check marker expression** - ensure your markers can distinguish the types

Example for expanded T cell subsets:
```csv
cell_type
T_cells_CD4_naive     # CD4+ CD45RA+ CCR7+
T_cells_CD4_memory    # CD4+ CD45RA- CCR7+
T_cells_CD4_effector  # CD4+ CD45RA- CCR7-
T_cells_regulatory    # CD4+ CD25+ FOXP3+
T_cells_CD8_naive     # CD8+ CD45RA+ CCR7+
T_cells_CD8_memory    # CD8+ CD45RA- 
...
```

## Performance Metrics

The pipeline provides comprehensive evaluation:

- **Overall Accuracy**: Fraction of correct predictions
- **F1 Scores**: Weighted and macro-averaged F1 scores
- **Per-Class Metrics**: Individual F1 scores for each cell type
- **Confusion Matrix**: Visual representation of classification performance
- **Classification Report**: Precision, recall, and support per class

### Current Results (Improved Mock Data)

```
Dataset Information:
  Training samples: 4,000
  Test samples: 2,000  
  Cell types: 5
  Markers used: 10

Model Performance:
  Training F1 Score: 0.987
  Test Accuracy: 0.993
  Test F1 (weighted): 0.993
  Test F1 (macro): 0.993

Per-Cell Type Performance:
  B_cells        : F1=1.000
  Monocytes      : F1=0.996
  NK_cells       : F1=0.994
  T_cells_CD4    : F1=0.988
  T_cells_CD8    : F1=0.985
```

## Programmatic Usage

```python
from cyanno_pipeline.cyanno import CyAnnoClassifier

# Initialize classifier
classifier = CyAnnoClassifier(
    markers=['FSC-A', 'SSC-A', 'CD3', 'CD4', 'CD8', 'CD14', 'CD19', 'CD20', 'CD45', 'CD56'],
    normalize=True,
    random_state=42
)

# Train on your data
import pandas as pd
training_data = pd.read_csv('data/training_data.csv')
metrics = classifier.train(training_data)
print(f"Training F1: {metrics['f1_score']:.3f}")

# Save the model
classifier.save_model('results/my_model.pkl')

# Load model for prediction
classifier.load_model('results/my_model.pkl')

# Make predictions
test_data = pd.read_csv('data/test_data.csv')
predictions = classifier.predict(test_data)

# Evaluate performance  
evaluation = classifier.evaluate(test_data)
print(f"Test F1 Score: {evaluation['f1_weighted']:.3f}")
```

## Configuration Reference

Complete `config/config.yaml` options:

```yaml
# Markers for classification (must match column names in data)
markers:
  - "FSC-A"     # Forward scatter
  - "SSC-A"     # Side scatter
  - "CD3"       # T cell marker
  - "CD4"       # Helper T cell marker
  - "CD8"       # Cytotoxic T cell marker
  - "CD14"      # Monocyte marker  
  - "CD19"      # B cell marker
  - "CD20"      # B cell marker
  - "CD45"      # Pan-leukocyte marker
  - "CD56"      # NK cell marker

# Data preprocessing
normalize: true              # Apply z-score normalization
ground_truth_column: "cell_type"  # Column name for ground truth

# Random Forest parameters
random_state: 42            # For reproducibility
n_estimators: 100           # Number of trees
max_depth: null            # Maximum tree depth (null = unlimited)
min_samples_split: 2       # Minimum samples to split node
min_samples_leaf: 1        # Minimum samples in leaf node

# Data generation (only for mock data)
samples:
  training: 4000           # Training set size
  test: 2000              # Test set size  
  unlabeled: 2000         # Unlabeled set size

cell_types:               # Cell types for mock data
  - "B_cells"
  - "T_cells_CD4" 
  - "T_cells_CD8"
  - "NK_cells"
  - "Monocytes"
```

## Integration with Larger Pipelines

### Snakemake Integration

Include in your main Snakemake workflow:

```python
# In your main Snakefile
include: "scina_module/Snakefile"

rule preprocess_fcs_files:
    input: "raw_data/{sample}.fcs"
    output: "processed_data/{sample}.csv" 
    # ... your preprocessing

rule annotate_cells:
    input:
        training="processed_data/training.csv",
        test="processed_data/test.csv"
    output:
        predictions="results/{sample}_annotations.csv"
    include: "scina_module/Snakefile"
```

### Python Module Integration

```python
# Integration example
import sys
sys.path.append('path/to/scina_module')

from cyanno_pipeline.cyanno import CyAnnoClassifier
from cyanno_pipeline.data_generator import generate_mock_data

# Use in your analysis pipeline
def analyze_cytometry_data(training_file, test_file, output_file):
    # Load and train
    classifier = CyAnnoClassifier(
        markers=['CD3', 'CD4', 'CD8', 'CD19', 'CD14', 'CD56'],
        normalize=True
    )
    
    training_data = pd.read_csv(training_file)
    classifier.train(training_data)
    
    # Predict and save
    test_data = pd.read_csv(test_file)
    predictions = classifier.predict(test_data)
    predictions.to_csv(output_file, index=False)
    
    return classifier.evaluate(test_data)
```

## Troubleshooting

### Common Issues

1. **Perfect Accuracy (100%)**
   - Data leakage between train/test sets
   - Synthetic data too well-separated
   - Run `python analyze_data.py` to diagnose

2. **Poor Performance (<80% F1)**
   - Insufficient training data
   - Inadequate marker panel  
   - Class imbalance
   - Consider feature selection or different algorithm

3. **Import Errors**
   - Ensure virtual environment is activated
   - Install requirements: `pip install -r requirements.txt`
   - Check Python path in scripts

4. **File Not Found Errors**
   - Check file paths in config.yaml
   - Ensure data files exist in expected locations
   - Use absolute paths if necessary

### Performance Optimization

- **Feature selection**: Remove uninformative markers
- **Hyperparameter tuning**: Adjust Random Forest parameters  
- **Data augmentation**: Add more training samples
- **Cross-validation**: Use k-fold CV for robust evaluation
- **Alternative algorithms**: Try XGBoost, SVM, or neural networks

## Dependencies

- Python ≥ 3.8
- pandas ≥ 1.3.0
- numpy ≥ 1.20.0
- scikit-learn ≥ 1.0.0
- matplotlib ≥ 3.3.0
- seaborn ≥ 0.11.0
- snakemake ≥ 6.0.0 (optional)

## License

This project is licensed under the MIT License.

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{cyanno_pipeline,
  title = {CyAnno Cell Type Annotation Pipeline},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/your-repo/cyanno-pipeline}
}
```
