#!/usr/bin/env python3
"""
Download FlowSOM example data - a well-known public CyTOF dataset
This uses data from the FlowSOM R package examples
"""

import os
import urllib.request
import zipfile
from pathlib import Path

# Configuration
BASE_DIR = Path("/home/javier/scina_module/test_data")
DOWNLOAD_URL = "https://github.com/SofieVG/FlowSOM/raw/master/inst/extdata/68983.fcs"
FCS_FILE = "flowsom_example.fcs"
LOCAL_FCS_PATH = BASE_DIR / "fcs" / FCS_FILE

def setup_directories():
    """Create necessary directories"""
    print("Setting up directories...")
    BASE_DIR.mkdir(exist_ok=True)
    (BASE_DIR / "fcs").mkdir(exist_ok=True)
    print("Directories created.")

def download_flowsom_data():
    """Download FlowSOM example FCS file"""
    print(f"Downloading FlowSOM example data from GitHub...")
    
    try:
        # Download the FCS file
        urllib.request.urlretrieve(DOWNLOAD_URL, LOCAL_FCS_PATH)
        print(f"Downloaded: {LOCAL_FCS_PATH}")
        
        # Check file size
        file_size = LOCAL_FCS_PATH.stat().st_size
        print(f"File size: {file_size:,} bytes")
        
        if file_size < 1000:
            raise Exception("Downloaded file seems too small - might be an error page")
            
        return True
        
    except Exception as e:
        print(f"Error downloading data: {e}")
        return False

def main():
    setup_directories()
    success = download_flowsom_data()
    
    if success:
        print("\nSuccess! FlowSOM example data downloaded.")
        print(f"FCS file location: {LOCAL_FCS_PATH}")
        print("\nNext: Run prepare_cyanno_inputs.py to create CSV files")
    else:
        print("\nFailed to download data. Try manual download.")

if __name__ == "__main__":
    main()
