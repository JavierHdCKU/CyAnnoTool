#!/usr/bin/env python3
"""
Download Levine 32-dimensional CyTOF dataset
This is a benchmark dataset commonly used for cytometry analysis
"""

import urllib.request
from pathlib import Path

BASE_DIR = Path("/home/javier/scina_module/test_data")
LEVINE_URL = "https://github.com/nolanlab/citrus/raw/master/inst/extdata/example1/example1_subset.fcs"
FCS_FILE = "levine_32dim_subset.fcs"

def download_levine_data():
    """Download Levine 32-dimensional dataset"""
    print("Downloading Levine 32-dimensional CyTOF data...")
    
    fcs_dir = BASE_DIR / "fcs"
    fcs_dir.mkdir(parents=True, exist_ok=True)
    
    local_path = fcs_dir / FCS_FILE
    
    try:
        urllib.request.urlretrieve(LEVINE_URL, local_path)
        print(f"Downloaded: {local_path}")
        print(f"File size: {local_path.stat().st_size:,} bytes")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    download_levine_data()
