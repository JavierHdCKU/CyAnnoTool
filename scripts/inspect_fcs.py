#!/usr/bin/env python3
"""
Inspect FlowSOM FCS file to get available markers
"""

import fcsparser
from pathlib import Path

FCS_FILE = Path("/home/javier/scina_module/test_data/fcs/flowsom_example.fcs")

def inspect_fcs():
    """Inspect the FCS file to see available markers"""
    print(f"Inspecting FCS file: {FCS_FILE}")
    
    try:
        # Parse the FCS file
        meta, data = fcsparser.parse(FCS_FILE, meta_data_only=False)
        
        print(f"\nFile contains {len(data)} cells")
        print(f"Available markers ({len(data.columns)} total):")
        
        for i, marker in enumerate(data.columns):
            print(f"  {i:2d}: {marker}")
        
        print(f"\nFirst 5 rows of data:")
        print(data.head())
        
        print(f"\nData shape: {data.shape}")
        print(f"Data types: {data.dtypes}")
        
        # Save marker names to a file for reference
        marker_file = Path("/home/javier/scina_module/test_data/available_markers.txt")
        with open(marker_file, 'w') as f:
            f.write("Available markers in FlowSOM example data:\n")
            for i, marker in enumerate(data.columns):
                f.write(f"{i:2d}: {marker}\n")
        
        print(f"\nMarker names saved to: {marker_file}")
        return data.columns.tolist()
        
    except Exception as e:
        print(f"Error reading FCS file: {e}")
        return None

if __name__ == "__main__":
    markers = inspect_fcs()
    if markers:
        print(f"\nFound {len(markers)} markers total")
    else:
        print("\nFailed to read FCS file")
