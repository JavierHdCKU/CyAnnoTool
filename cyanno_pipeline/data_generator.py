#!/usr/bin/env python3
"""
Mock data generator for CyAnno pipeline testing
Creates synthetic cytometry data with known cell types
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CytometryDataGenerator:
    """Generate synthetic cytometry data for testing"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
    
    def generate_cell_population(self, 
                                 n_cells: int,
                                 cell_type: str,
                                 marker_means: Dict[str, float],
                                 marker_stds: Dict[str, float],
                                 noise_factor: float = 0.2) -> pd.DataFrame:
        """
        Generate a single cell population with realistic biological variation
        
        Args:
            n_cells: Number of cells to generate
            cell_type: Name of the cell type
            marker_means: Mean expression for each marker
            marker_stds: Standard deviation for each marker
            noise_factor: Additional noise factor (increased for realism)
            
        Returns:
            DataFrame with cell data
        """
        
        data = {}
        
        for marker, mean in marker_means.items():
            std = marker_stds[marker]
            
            # Generate expression values with biological variation
            expression = np.random.normal(mean, std, n_cells)
            
            # Add additional technical noise
            tech_noise = np.random.normal(0, noise_factor * std, n_cells)
            expression += tech_noise
            
            # Add some outliers (5% of cells)
            outlier_mask = np.random.random(n_cells) < 0.05
            outlier_factor = np.random.choice([-1, 1], size=np.sum(outlier_mask))
            expression[outlier_mask] += outlier_factor * std * 2
            
            # Ensure non-negative values with some low-level background
            background_level = mean * 0.02  # 2% background
            expression = np.maximum(expression, background_level)
            
            data[marker] = expression
        
        # Add cell type label
        data['cell_type'] = [cell_type] * n_cells
        
        return pd.DataFrame(data)
    
    def generate_mock_dataset(self, 
                             output_dir: Path,
                             n_training_samples: int = 1000,
                             n_test_samples: int = 500) -> Dict[str, Path]:
        """
        Generate complete mock dataset for CyAnno testing
        
        Args:
            output_dir: Directory to save generated data
            n_training_samples: Number of training samples per cell type
            n_test_samples: Number of test samples per cell type
            
        Returns:
            Dictionary with paths to generated files
        """
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define markers (typical flow cytometry markers)
        markers = [
            'FSC-A', 'SSC-A', 'CD3', 'CD4', 'CD8', 
            'CD14', 'CD19', 'CD45', 'CD56', 'CD20'
        ]
        
        # Define cell type signatures - MADE MORE REALISTIC with overlap
        cell_type_profiles = {
            'T_cells_CD4': {
                'means': {'FSC-A': 50000, 'SSC-A': 30000, 'CD3': 6000, 'CD4': 5000, 
                         'CD8': 800, 'CD14': 500, 'CD19': 600, 'CD45': 7000, 'CD56': 500, 'CD20': 600},
                'stds': {'FSC-A': 12000, 'SSC-A': 8000, 'CD3': 2000, 'CD4': 1800, 
                        'CD8': 600, 'CD14': 400, 'CD19': 500, 'CD45': 1800, 'CD56': 400, 'CD20': 500}
            },
            'T_cells_CD8': {
                'means': {'FSC-A': 48000, 'SSC-A': 28000, 'CD3': 6200, 'CD4': 700, 
                         'CD8': 4800, 'CD14': 500, 'CD19': 500, 'CD45': 6800, 'CD56': 800, 'CD20': 500},
                'stds': {'FSC-A': 11000, 'SSC-A': 7500, 'CD3': 1900, 'CD4': 500, 
                        'CD8': 1700, 'CD14': 400, 'CD19': 450, 'CD45': 1700, 'CD56': 600, 'CD20': 450}
            },
            'B_cells': {
                'means': {'FSC-A': 45000, 'SSC-A': 25000, 'CD3': 600, 'CD4': 500, 
                         'CD8': 500, 'CD14': 600, 'CD19': 5500, 'CD45': 6500, 'CD56': 500, 'CD20': 5000},
                'stds': {'FSC-A': 10000, 'SSC-A': 6000, 'CD3': 500, 'CD4': 400, 
                        'CD8': 400, 'CD14': 450, 'CD19': 1600, 'CD45': 1500, 'CD56': 400, 'CD20': 1500}
            },
            'Monocytes': {
                'means': {'FSC-A': 60000, 'SSC-A': 40000, 'CD3': 600, 'CD4': 1200, 
                         'CD8': 500, 'CD14': 6000, 'CD19': 500, 'CD45': 7500, 'CD56': 700, 'CD20': 500},
                'stds': {'FSC-A': 15000, 'SSC-A': 10000, 'CD3': 500, 'CD4': 800, 
                        'CD8': 400, 'CD14': 1800, 'CD19': 400, 'CD45': 1800, 'CD56': 500, 'CD20': 400}
            },
            'NK_cells': {
                'means': {'FSC-A': 52000, 'SSC-A': 32000, 'CD3': 700, 'CD4': 600, 
                         'CD8': 1500, 'CD14': 500, 'CD19': 500, 'CD45': 6000, 'CD56': 5800, 'CD20': 500},
                'stds': {'FSC-A': 12000, 'SSC-A': 8500, 'CD3': 600, 'CD4': 500, 
                        'CD8': 1000, 'CD14': 400, 'CD19': 400, 'CD45': 1500, 'CD56': 1700, 'CD20': 400}
            }
        }
        
        # Generate training data with specific sample IDs
        training_data = []
        for i, (cell_type, profile) in enumerate(cell_type_profiles.items()):
            cells = self.generate_cell_population(
                n_training_samples,
                cell_type,
                profile['means'],
                profile['stds'],
                noise_factor=0.2  # Increased noise
            )
            # Use distinct sample IDs for training
            cells['sample_id'] = [f'train_sample_{(j % 3) + 1}' for j in range(len(cells))]
            training_data.append(cells)
        
        training_df = pd.concat(training_data, ignore_index=True)
        training_df = training_df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        
        # Generate test data with DIFFERENT sample IDs and MORE noise
        test_data = []
        for i, (cell_type, profile) in enumerate(cell_type_profiles.items()):
            cells = self.generate_cell_population(
                n_test_samples,
                cell_type,
                profile['means'],
                profile['stds'],
                noise_factor=0.3  # Even more noise for test data
            )
            # Use DIFFERENT sample IDs for test
            cells['sample_id'] = [f'test_sample_{(j % 3) + 1}' for j in range(len(cells))]
            test_data.append(cells)
        
        test_df = pd.concat(test_data, ignore_index=True)
        test_df = test_df.sample(frac=1, random_state=self.random_state + 100).reset_index(drop=True)  # Different seed
        
        # Save files
        training_file = output_dir / 'training_data.csv'
        test_file = output_dir / 'test_data.csv'
        
        training_df.to_csv(training_file, index=False)
        test_df.to_csv(test_file, index=False)
        
        # Also create separate files for unlabeled data (test without labels)
        unlabeled_test = test_df.drop('cell_type', axis=1)
        unlabeled_file = output_dir / 'unlabeled_data.csv'
        unlabeled_test.to_csv(unlabeled_file, index=False)
        
        # Create metadata file
        metadata = {
            'markers': markers,
            'cell_types': list(cell_type_profiles.keys()),
            'n_training_samples': len(training_df),
            'n_test_samples': len(test_df),
            'generated_files': {
                'training': str(training_file),
                'test': str(test_file),
                'unlabeled': str(unlabeled_file)
            }
        }
        
        metadata_file = output_dir / 'metadata.txt'
        with open(metadata_file, 'w') as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
        
        logger.info(f"Generated mock dataset in {output_dir}")
        logger.info(f"Training samples: {len(training_df)}")
        logger.info(f"Test samples: {len(test_df)}")
        logger.info(f"Cell types: {list(cell_type_profiles.keys())}")
        
        return {
            'training_data': training_file,
            'test_data': test_file,
            'unlabeled_data': unlabeled_file,
            'metadata': metadata_file,
            'markers': markers
        }


def generate_mock_data(output_dir: str = '/home/javier/scina_module/data'):
    """Generate mock data for CyAnno testing"""
    
    generator = CytometryDataGenerator(random_state=42)
    
    files = generator.generate_mock_dataset(
        Path(output_dir),
        n_training_samples=800,  # 800 per cell type
        n_test_samples=400       # 400 per cell type
    )
    
    return files


if __name__ == "__main__":
    generate_mock_data()
