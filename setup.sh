#!/bin/bash
"""
Setup script for CyAnno pipeline
"""

echo "Setting up CyAnno Pipeline Environment..."

# Install required Python packages
echo "Installing Python dependencies..."
pip install pandas numpy scikit-learn matplotlib seaborn snakemake pyyaml

echo "Testing imports..."
python3 -c "
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
print('✅ All dependencies installed successfully!')
"

echo ""
echo "CyAnno Pipeline Setup Complete!"
echo ""
echo "To run the pipeline:"
echo "  python3 run_pipeline.py"
echo ""
echo "To run with Snakemake:"
echo "  snakemake --cores 4"
