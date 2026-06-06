"""
Download all medical datasets to D drive
Configures kagglehub to use D:\medical_datasets as cache
"""

import kagglehub
import os
from pathlib import Path
import subprocess

# Configure kagglehub to download to D drive
d_drive_path = r"D:\medical_datasets"
Path(d_drive_path).mkdir(parents=True, exist_ok=True)

# Set environment variable for kagglehub cache
os.environ['KAGGLEHUB_HOME'] = d_drive_path

print("="*70)
print("DOWNLOADING ALL MEDICAL DATASETS TO D: DRIVE")
print("="*70)
print(f"Cache directory: {d_drive_path}\n")

# List of datasets to download
datasets_to_download = {
    "heartbeat": "shayanfazeli/heartbeat",
    "ecg_dataset": "devavratatripathy/ecg-dataset",
    "drug_bank": "devildev89/drug-bank-5110",
    "skin_disease": "ahmedxc4/skin-ds"
}

downloaded = {}
failed = {}

# Download each dataset
for name, dataset_id in datasets_to_download.items():
    print(f"\n{'='*70}")
    print(f"Downloading: {name}")
    print(f"Dataset ID: {dataset_id}")
    print('='*70)
    
    try:
        path = kagglehub.dataset_download(dataset_id)
        print(f"✅ SUCCESS: {name}")
        print(f"Location: {path}")
        downloaded[name] = path
    except Exception as e:
        print(f"❌ FAILED: {name}")
        print(f"Error: {str(e)[:200]}")
        failed[name] = str(e)[:100]

# Download ISIC 2024 via git
print(f"\n{'='*70}")
print("Downloading: ISIC 2024 Challenge Dataset")
print('='*70)
isic_path = os.path.join(d_drive_path, "isic-2024-challenge")
try:
    if not os.path.exists(isic_path):
        print(f"Cloning ISIC 2024 from GitHub...")
        subprocess.run([
            'git', 'clone', 
            'https://github.com/ISIC-Research/2024-challenge-dataset.git',
            isic_path
        ], check=True)
        print(f"✅ SUCCESS: ISIC 2024")
        print(f"Location: {isic_path}")
        downloaded["isic_2024"] = isic_path
    else:
        print(f"⚠️ ISIC 2024 already exists at: {isic_path}")
        downloaded["isic_2024"] = isic_path
except Exception as e:
    print(f"❌ FAILED: ISIC 2024")
    print(f"Error: {str(e)[:200]}")
    failed["isic_2024"] = str(e)[:100]

# Summary
print(f"\n{'='*70}")
print("DOWNLOAD SUMMARY")
print('='*70)
print(f"\n✅ Successfully downloaded: {len(downloaded)}")
for name, path in downloaded.items():
    print(f"   • {name}: {path}")

if failed:
    print(f"\n❌ Failed to download: {len(failed)}")
    for name, error in failed.items():
        print(f"   • {name}: {error}")

print(f"\n{'='*70}")
print(f"All downloads saved to: {d_drive_path}")
print('='*70)
