#!/usr/bin/env python3
"""
Data Quality Analysis - Check for data leakage and validation issues
"""

import pandas as pd
import numpy as np
from collections import Counter
import sys

def analyze_data_quality():
    """Analyze the data for potential issues causing perfect F1 score"""
    
    print("=" * 60)
    print("DATA QUALITY ANALYSIS")
    print("=" * 60)
    
    # Load data
    train_data = pd.read_csv('data/training_data.csv')
    test_data = pd.read_csv('data/test_data.csv')
    predictions = pd.read_csv('results/cyanno_predictions.csv')
    
    print(f"\n1. BASIC DATA INFO:")
    print(f"   Training samples: {len(train_data)}")
    print(f"   Test samples: {len(test_data)}")
    print(f"   Prediction samples: {len(predictions)}")
    
    # Check for data leakage - exact duplicates
    print(f"\n2. DATA LEAKAGE CHECK:")
    
    # Check exact row duplicates between train and test
    marker_cols = ['FSC-A', 'SSC-A', 'CD3', 'CD4', 'CD8', 'CD14', 'CD19', 'CD45', 'CD56', 'CD20']
    
    train_markers = train_data[marker_cols]
    test_markers = test_data[marker_cols]
    
    # Create hash of each row to check for exact duplicates
    train_hashes = set(pd.util.hash_pandas_object(train_markers))
    test_hashes = set(pd.util.hash_pandas_object(test_markers))
    
    overlap = len(train_hashes & test_hashes)
    print(f"   Exact duplicate rows between train/test: {overlap}")
    
    if overlap > 0:
        print("   ❌ DATA LEAKAGE DETECTED!")
    else:
        print("   ✅ No exact duplicate rows")
    
    # Check sample_id overlap
    train_samples = set(train_data['sample_id'].unique())
    test_samples = set(test_data['sample_id'].unique())
    sample_overlap = len(train_samples & test_samples)
    
    print(f"   Training sample IDs: {sorted(train_samples)}")
    print(f"   Test sample IDs: {sorted(test_samples)}")
    print(f"   Sample ID overlap: {sample_overlap} IDs")
    
    if sample_overlap > 0:
        print("   ⚠️  Sample ID overlap detected (potential leakage)")
    
    # Check data distribution
    print(f"\n3. DATA DISTRIBUTION ANALYSIS:")
    
    print(f"   Training cell type distribution:")
    train_dist = train_data['cell_type'].value_counts()
    for cell_type, count in train_dist.items():
        print(f"     {cell_type}: {count} ({count/len(train_data)*100:.1f}%)")
    
    print(f"   Test cell type distribution:")
    test_dist = test_data['cell_type'].value_counts()
    for cell_type, count in test_dist.items():
        print(f"     {cell_type}: {count} ({count/len(test_data)*100:.1f}%)")
    
    # Check if distributions are too uniform
    train_cv = np.std(train_dist.values) / np.mean(train_dist.values)
    test_cv = np.std(test_dist.values) / np.mean(test_dist.values)
    
    print(f"   Distribution coefficient of variation:")
    print(f"     Training: {train_cv:.3f}")
    print(f"     Test: {test_cv:.3f}")
    
    if train_cv < 0.1 and test_cv < 0.1:
        print("   ⚠️  Distributions are suspiciously uniform (synthetic data)")
    
    # Analyze marker separation
    print(f"\n4. MARKER SEPARATION ANALYSIS:")
    
    # Check if cell types are too well separated
    mean_separations = {}
    for cell_type in train_data['cell_type'].unique():
        cell_data = train_data[train_data['cell_type'] == cell_type][marker_cols]
        other_data = train_data[train_data['cell_type'] != cell_type][marker_cols]
        
        # Calculate mean distance between cell type and others
        cell_mean = cell_data.mean()
        other_mean = other_data.mean()
        
        # Euclidean distance between means
        distance = np.sqrt(((cell_mean - other_mean) ** 2).sum())
        mean_separations[cell_type] = distance
        
        print(f"   {cell_type} separation: {distance:.2f}")
    
    avg_separation = np.mean(list(mean_separations.values()))
    print(f"   Average separation: {avg_separation:.2f}")
    
    if avg_separation > 10000:
        print("   ⚠️  Cell types are extremely well separated (too easy)")
    
    # Check prediction confidence
    print(f"\n5. PREDICTION CONFIDENCE ANALYSIS:")
    confidence_stats = predictions['prediction_confidence'].describe()
    print(f"   Confidence statistics:")
    for stat, value in confidence_stats.items():
        print(f"     {stat}: {value:.3f}")
    
    high_conf = (predictions['prediction_confidence'] > 0.99).sum()
    print(f"   Predictions with >99% confidence: {high_conf}/{len(predictions)} ({high_conf/len(predictions)*100:.1f}%)")
    
    if high_conf / len(predictions) > 0.95:
        print("   ⚠️  Suspiciously high prediction confidence")
    
    # Check for perfect predictions
    perfect_matches = (predictions['cell_type'] == predictions['predicted_cell_type']).sum()
    print(f"   Perfect predictions: {perfect_matches}/{len(predictions)} ({perfect_matches/len(predictions)*100:.1f}%)")
    
    if perfect_matches == len(predictions):
        print("   ❌ ALL PREDICTIONS ARE PERFECT - LIKELY ISSUE!")
    
    print(f"\n6. RECOMMENDATIONS:")
    
    issues_found = []
    
    if overlap > 0:
        issues_found.append("Data leakage (exact duplicates)")
    if sample_overlap > 0:
        issues_found.append("Sample ID overlap")
    if avg_separation > 10000:
        issues_found.append("Cell types too well separated")
    if high_conf / len(predictions) > 0.95:
        issues_found.append("Suspiciously high confidence")
    if perfect_matches == len(predictions):
        issues_found.append("Perfect accuracy (100%)")
    
    if issues_found:
        print("   Issues detected:")
        for issue in issues_found:
            print(f"     - {issue}")
        print("\n   Recommended fixes:")
        print("     1. Ensure proper train/test split with no overlap")
        print("     2. Add more realistic noise to synthetic data")
        print("     3. Make cell types more similar (reduce separation)")
        print("     4. Use different sample IDs for train/test")
    else:
        print("   ✅ No obvious issues detected")
    
    return {
        'data_leakage': overlap > 0,
        'sample_overlap': sample_overlap > 0,
        'too_separated': avg_separation > 10000,
        'high_confidence': high_conf / len(predictions) > 0.95,
        'perfect_accuracy': perfect_matches == len(predictions)
    }

if __name__ == "__main__":
    sys.path.append('/home/javier/scina_module')
    analyze_data_quality()
