#!/usr/bin/env python3
"""
Generate mock cytometry data for CyAnno testing
"""

import sys
sys.path.append('/home/javier/scina_module')

from cyanno_pipeline.data_generator import generate_mock_data

if __name__ == "__main__":
    generate_mock_data("/home/javier/scina_module/data")
