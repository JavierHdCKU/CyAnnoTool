#!/usr/bin/env python3
"""
CyAnno Implementation - Simplified version for Snakemake pipeline
A machine learning approach for cytometry cell type annotation
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class CyAnnoClassifier:
    """
    Simplified CyAnno implementation for cell type annotation
    """
    
    def __init__(self, 
                 markers: List[str],
                 normalize: bool = True,
                 random_state: int = 42):
        """
        Initialize CyAnno classifier
        
        Args:
            markers: List of marker names to use for classification
            normalize: Whether to apply normalization
            random_state: Random seed for reproducibility
        """
        self.markers = markers
        self.normalize = normalize
        self.random_state = random_state
        
        # Initialize models
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1
        )
        
        self.scaler = StandardScaler() if normalize else None
        self.cell_types = None
        self.is_trained = False
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _preprocess_data(self, data: pd.DataFrame) -> np.ndarray:
        """Preprocess the data (normalization, etc.)"""
        
        # Select only relevant markers
        marker_data = data[self.markers].copy()
        
        # Convert to numpy array
        X = marker_data.values
        
        # Apply normalization if requested
        if self.normalize and self.scaler is not None:
            if self.is_trained:
                X = self.scaler.transform(X)
            else:
                X = self.scaler.fit_transform(X)
        
        return X
    
    def train(self, 
              handgated_data: pd.DataFrame,
              cell_type_column: str = 'cell_type') -> Dict:
        """
        Train the CyAnno classifier
        
        Args:
            handgated_data: DataFrame with manually gated cells
            cell_type_column: Name of column containing cell type labels
            
        Returns:
            Dictionary with training metrics
        """
        self.logger.info("Starting CyAnno training...")
        
        # Check if required columns exist
        missing_markers = [m for m in self.markers if m not in handgated_data.columns]
        if missing_markers:
            raise ValueError(f"Missing markers in data: {missing_markers}")
        
        if cell_type_column not in handgated_data.columns:
            raise ValueError(f"Cell type column '{cell_type_column}' not found")
        
        # Prepare features and labels
        X = self._preprocess_data(handgated_data)
        y = handgated_data[cell_type_column].values
        
        # Store cell types
        self.cell_types = sorted(list(set(y)))
        self.logger.info(f"Found {len(self.cell_types)} cell types: {self.cell_types}")
        
        # Split for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, 
            random_state=self.random_state,
            stratify=y
        )
        
        # Train classifier
        self.classifier.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate on validation set
        y_pred = self.classifier.predict(X_val)
        f1 = f1_score(y_val, y_pred, average='weighted')
        
        metrics = {
            'f1_score': f1,
            'n_training_samples': len(X_train),
            'n_validation_samples': len(X_val),
            'cell_types': self.cell_types,
            'classification_report': classification_report(y_val, y_pred)
        }
        
        self.logger.info(f"Training completed. F1 Score: {f1:.3f}")
        return metrics
    
    def predict(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict cell types for new data
        
        Args:
            data: DataFrame with cell measurements
            
        Returns:
            Tuple of (predicted_labels, prediction_probabilities)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X = self._preprocess_data(data)
        
        predictions = self.classifier.predict(X)
        probabilities = self.classifier.predict_proba(X)
        
        return predictions, probabilities
    
    def evaluate(self, 
                 test_data: pd.DataFrame,
                 true_labels_column: str = 'cell_type') -> Dict:
        """
        Evaluate model performance on test data
        
        Args:
            test_data: DataFrame with test data and true labels
            true_labels_column: Name of column with true labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Predict
        predictions, probabilities = self.predict(test_data)
        true_labels = test_data[true_labels_column].values
        
        # Calculate metrics
        f1 = f1_score(true_labels, predictions, average='weighted')
        f1_macro = f1_score(true_labels, predictions, average='macro')
        
        # Per-class F1 scores
        f1_per_class = f1_score(true_labels, predictions, average=None)
        
        metrics = {
            'f1_weighted': f1,
            'f1_macro': f1_macro,
            'f1_per_class': dict(zip(self.cell_types, f1_per_class)),
            'classification_report': classification_report(true_labels, predictions),
            'confusion_matrix': confusion_matrix(true_labels, predictions).tolist(),
            'n_test_samples': len(test_data)
        }
        
        self.logger.info(f"Evaluation completed. F1 Score (weighted): {f1:.3f}")
        return metrics
    
    def save_model(self, filepath: Path):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'classifier': self.classifier,
            'scaler': self.scaler,
            'markers': self.markers,
            'cell_types': self.cell_types,
            'normalize': self.normalize,
            'random_state': self.random_state
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: Path):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.classifier = model_data['classifier']
        self.scaler = model_data['scaler']
        self.markers = model_data['markers']
        self.cell_types = model_data['cell_types']
        self.normalize = model_data['normalize']
        self.random_state = model_data['random_state']
        self.is_trained = True
        
        self.logger.info(f"Model loaded from {filepath}")


def load_fcs_data(filepath: Path, markers: List[str] = None) -> pd.DataFrame:
    """
    Load FCS file data
    
    Args:
        filepath: Path to FCS file
        markers: List of markers to extract (if None, use all)
        
    Returns:
        DataFrame with cell data
    """
    try:
        import fcsparser
        meta, data = fcsparser.parse(filepath)
        
        if markers:
            available_markers = [m for m in markers if m in data.columns]
            if len(available_markers) != len(markers):
                missing = [m for m in markers if m not in data.columns]
                logging.warning(f"Missing markers: {missing}")
            data = data[available_markers]
        
        return data
    
    except ImportError:
        raise ImportError("fcsparser is required for FCS file loading. Install with: pip install fcsparser")


def run_cyanno_pipeline(config: Dict) -> Dict:
    """
    Run complete CyAnno pipeline
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary with results and metrics
    """
    
    # Initialize classifier
    cyanno = CyAnnoClassifier(
        markers=config['markers'],
        normalize=config.get('normalize', True),
        random_state=config.get('random_state', 42)
    )
    
    # Load training data
    train_data = pd.read_csv(config['training_data'])
    
    # Train model
    training_metrics = cyanno.train(
        train_data, 
        cell_type_column=config.get('cell_type_column', 'cell_type')
    )
    
    # Save model if requested
    if 'model_output' in config:
        cyanno.save_model(Path(config['model_output']))
    
    # Load test data and predict
    test_data = pd.read_csv(config['test_data'])
    predictions, probabilities = cyanno.predict(test_data)
    
    # Save predictions
    output_data = test_data.copy()
    output_data['predicted_cell_type'] = predictions
    output_data['prediction_confidence'] = np.max(probabilities, axis=1)
    
    # Add individual class probabilities
    for i, cell_type in enumerate(cyanno.cell_types):
        output_data[f'prob_{cell_type}'] = probabilities[:, i]
    
    output_data.to_csv(config['output_predictions'], index=False)
    
    # Evaluate if ground truth is available
    evaluation_metrics = None
    if config.get('ground_truth_column') and config['ground_truth_column'] in test_data.columns:
        evaluation_metrics = cyanno.evaluate(
            test_data,
            true_labels_column=config['ground_truth_column']
        )
    
    results = {
        'training_metrics': training_metrics,
        'evaluation_metrics': evaluation_metrics,
        'n_predictions': len(predictions),
        'output_file': config['output_predictions']
    }
    
    return results
