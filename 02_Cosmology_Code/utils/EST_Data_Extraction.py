import numpy as np
import pandas as pd
import csv
import json
import zipfile
import os
from pathlib import Path

def extract_est_simulation(directory):
    """Extract all data from an EST simulation directory"""
    
    est_dir = Path(directory)
    if not est_dir.exists():
        print(f"Error: Directory not found: {directory}")
        return
    
    print(f"🔍 Extracting EST data from: {est_dir.name}")
    
    # Create extraction directory
    extract_dir = est_dir / "EXTRACTED"
    extract_dir.mkdir(exist_ok=True)
    
    # 1. Load and process all .npy files
    npy_files = list(est_dir.glob("*.npy"))
    for npy_file in npy_files:
        try:
            data = np.load(npy_file, allow_pickle=True)
            
            # Save as CSV if 1D or 2D
            if data.ndim == 1:
                csv_path = extract_dir / f"{npy_file.stem}.csv"
                pd.DataFrame({npy_file.stem: data}).to_csv(csv_path, index_label="Index")
                print(f"  ✅ {npy_file.name} → CSV ({len(data)} values)")
                
            elif data.ndim == 2:
                csv_path = extract_dir / f"{npy_file.stem}.csv"
                pd.DataFrame(data).to_csv(csv_path, index=False, header=False)
                print(f"  ✅ {npy_file.name} → CSV ({data.shape} matrix)")
                
            elif data.ndim == 3:
                # For 3D data, save slices and statistics
                stats = {
                    "shape": data.shape,
                    "mean": float(data.mean()),
                    "std": float(data.std()),
                    "min": float(data.min()),
                    "max": float(data.max()),
                    "active_cells": int(data.sum()),
                    "density_percent": float(data.mean() * 100)
                }
                
                # Save statistics
                stats_path = extract_dir / f"{npy_file.stem}_stats.json"
                with open(stats_path, 'w') as f:
                    json.dump(stats, f, indent=2)
                
                # Save middle slices
                for axis, name in zip([0, 1, 2], ['X', 'Y', 'Z']):
                    slice_idx = data.shape[axis] // 2
                    if axis == 0:
                        slice_data = data[slice_idx, :, :]
                    elif axis == 1:
                        slice_data = data[:, slice_idx, :]
                    else:
                        slice_data = data[:, :, slice_idx]
                    
                    slice_path = extract_dir / f"{npy_file.stem}_slice_{name}.csv"
                    pd.DataFrame(slice_data).to_csv(slice_path, index=False, header=False)
                
                print(f"  ✅ {npy_file.name} → JSON + 3 slices")
                
        except Exception as e:
            print(f"  ❌ {npy_file.name}: {e}")
    
    # 2. Copy all CSV files (they're already in good format)
    csv_files = list(est_dir.glob("*.csv"))
    for csv_file in csv_files:
        dest = extract_dir / csv_file.name
        csv_file.copy(dest)
        print(f"  📋 Copied: {csv_file.name}")
    
    # 3. Copy all text files
    txt_files = list(est_dir.glob("*.txt"))
    for txt_file in txt_files:
        dest = extract_dir / txt_file.name
        txt_file.copy(dest)
        print(f"  📄 Copied: {txt_file.name}")
    
    # 4. Copy all images
    img_files = list(est_dir.glob("*.png")) + list(est_dir.glob("*.gif")) + list(est_dir.glob("*.jpg"))
    for img_file in img_files:
        dest = extract_dir / img_file.name
        img_file.copy(dest)
        print(f"  🖼️ Copied: {img_file.name}")
    
    # 5. Create a summary file
    summary_path = extract_dir / "EXTRACTION_SUMMARY.txt"
    with open(summary_path, 'w') as f:
        f.write(f"EST Simulation Data Extraction\n")
        f.write(f"Source: {est_dir.name}\n")
        f.write(f"Date: {pd.Timestamp.now()}\n")
        f.write("="*60 + "\n\n")
        
        f.write("Files Extracted:\n")
        f.write(f"  .npy files: {len(npy_files)}\n")
        f.write(f"  .csv files: {len(csv_files)}\n")
        f.write(f"  .txt files: {len(txt_files)}\n")
        f.write(f"  Image files: {len(img_files)}\n")
        f.write(f"  Total: {len(npy_files) + len(csv_files) + len(txt_files) + len(img_files)}\n")
    
    # 6. Create ZIP archive
    zip_path = est_dir / f"{est_dir.name}_EXTRACTED.zip"
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for file in extract_dir.rglob("*"):
            if file.is_file():
                zipf.write(file, arcname=file.relative_to(extract_dir))
    
    print(f"\n✅ Extraction complete!")
    print(f"📁 Extracted to: {extract_dir}")
    print(f"📦 ZIP archive: {zip_path}")
    
    # Show extracted files
    print("\n📋 Extracted files:")
    for file in sorted(extract_dir.glob("*")):
        if file.is_file():
            size_kb = file.stat().st_size / 1024
            print(f"  • {file.name} ({size_kb:.1f} KB)")

# Run it
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = input("Enter path to EST output directory: ").strip()
    
    extract_est_simulation(directory)