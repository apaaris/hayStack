#!/usr/bin/env python3
"""
Merge multiple NanoVDB grid files from distributed simulation into a single grid.

Usage:
    python merge_nanovdb.py <input_dir> <output_file> [--timestep TIMESTEP]
    
Example:
    python merge_nanovdb.py /path/to/nvdb_files merged_lambda2_004320.nvdb --timestep 004320
"""

import sys
import os
import glob
import json
import argparse
from pathlib import Path

try:
    import pyopenvdb as vdb
    import numpy as np
except ImportError as e:
    print(f"ERROR: Required Python packages not found: {e}")
    print("Install with: pip install pyopenvdb numpy")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Merge distributed NanoVDB files into a single grid',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge all .nvdb files in a directory
  %(prog)s /data/nvdb_output merged.vdb
  
  # Merge only files from a specific timestep
  %(prog)s /data/nvdb_output merged_004320.vdb --timestep 004320
  
  # Merge with pattern matching
  %(prog)s /data/nvdb_output merged_lambda2.vdb --pattern "nvdb_lambda2_*.nvdb"
        """
    )
    parser.add_argument('input_dir', type=str,
                        help='Directory containing .nvdb files and .json metadata')
    parser.add_argument('output_file', type=str,
                        help='Output merged VDB file path')
    parser.add_argument('--timestep', type=str, default=None,
                        help='Only merge files from specific timestep (e.g., 004320)')
    parser.add_argument('--pattern', type=str, default="*.nvdb",
                        help='File pattern to match (default: *.nvdb)')
    parser.add_argument('--field', type=str, default=None,
                        help='Only merge files for specific field name')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print detailed progress information')
    return parser.parse_args()


def load_metadata(nvdb_file):
    """Load metadata JSON file associated with a .nvdb file"""
    json_file = nvdb_file + ".json"
    if not os.path.exists(json_file):
        return None
    
    with open(json_file, 'r') as f:
        return json.load(f)


def merge_grids(input_dir, output_file, timestep=None, pattern="*.nvdb", field=None, verbose=False):
    """Merge NanoVDB grids from multiple files"""
    
    # Find all matching .nvdb files
    search_pattern = os.path.join(input_dir, pattern)
    nvdb_files = glob.glob(search_pattern)
    
    if not nvdb_files:
        print(f"ERROR: No .nvdb files found matching: {search_pattern}")
        return False
    
    # Filter by timestep if specified
    if timestep:
        nvdb_files = [f for f in nvdb_files if f"_{timestep}." in f]
        if not nvdb_files:
            print(f"ERROR: No files found for timestep {timestep}")
            return False
    
    # Filter by field name if specified
    if field:
        nvdb_files = [f for f in nvdb_files if f"nvdb_{field}_" in os.path.basename(f)]
        if not nvdb_files:
            print(f"ERROR: No files found for field {field}")
            return False
    
    nvdb_files = sorted(nvdb_files)
    print(f"Found {len(nvdb_files)} NanoVDB files to merge")
    
    if verbose:
        for f in nvdb_files[:5]:
            print(f"  - {os.path.basename(f)}")
        if len(nvdb_files) > 5:
            print(f"  ... and {len(nvdb_files) - 5} more")
    
    # Load metadata from first file to get grid properties
    first_meta = load_metadata(nvdb_files[0])
    if first_meta:
        print(f"\nGrid properties:")
        print(f"  Field: {first_meta['fieldName']}")
        print(f"  Spacing: {first_meta['gridSpacing']}")
        print(f"  Threshold: {first_meta['threshold']}")
    
    # Create merged grid
    # Note: Since NanoVDB files already contain global coordinates,
    # we just need to read and combine them
    print(f"\nMerging grids...")
    
    try:
        # Read first grid to initialize
        merged_grid = vdb.read(nvdb_files[0], "density")
        total_active = merged_grid.activeVoxelCount()
        
        if verbose:
            print(f"  Loaded: {os.path.basename(nvdb_files[0])} ({merged_grid.activeVoxelCount()} active voxels)")
        
        # Merge remaining grids
        for i, nvdb_file in enumerate(nvdb_files[1:], 1):
            grid = vdb.read(nvdb_file, "density")
            
            # Combine grids (this should preserve the global coordinates)
            merged_grid.merge(grid)
            
            total_active += grid.activeVoxelCount()
            
            if verbose and i % 10 == 0:
                print(f"  Merged {i}/{len(nvdb_files)-1} grids... ({merged_grid.activeVoxelCount()} active)")
        
        print(f"\nMerged grid statistics:")
        print(f"  Total active voxels: {merged_grid.activeVoxelCount()}")
        print(f"  Input voxels (sum): {total_active}")
        print(f"  Bounding box: {merged_grid.evalActiveVoxelBoundingBox()}")
        
        # Write merged grid
        print(f"\nWriting merged grid to: {output_file}")
        vdb.write(output_file, grids=[merged_grid])
        
        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"  Output file size: {file_size_mb:.2f} MB")
        print("✅ Merge complete!")
        
        return True
        
    except Exception as e:
        print(f"ERROR during merge: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    args = parse_args()
    
    if not os.path.isdir(args.input_dir):
        print(f"ERROR: Input directory not found: {args.input_dir}")
        sys.exit(1)
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    success = merge_grids(
        args.input_dir,
        args.output_file,
        timestep=args.timestep,
        pattern=args.pattern,
        field=args.field,
        verbose=args.verbose
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

