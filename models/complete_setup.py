"""
Complete Download & Cleanup Script
- Download all datasets to D:\c++ homework\python\ai doctor dataset
- Clean up C drive cache
- Organize models folder
- Remove unnecessary files
"""

import os
import shutil
import subprocess
from pathlib import Path

print("="*80)
print("COMPLETE MEDICAL AI SYSTEM SETUP")
print("="*80)

# Paths
target_dir = Path(r"D:\c++ homework\python\ai doctor dataset")
ai_doctor_path = Path(r"D:\c++ homework\python\ml projects\Ai Doctor")
models_folder = ai_doctor_path / "models"
c_cache = Path(r"C:\Users\ASUS\.cache\kagglehub")

# Create target directory
target_dir.mkdir(parents=True, exist_ok=True)
models_folder.mkdir(exist_ok=True)

# Configure environment to use target directory
os.environ['KAGGLEHUB_HOME'] = str(target_dir)

print(f"\n📁 Target Directory: {target_dir}")
print(f"🤖 Models Folder: {models_folder}")

# ==================== STEP 1: DOWNLOAD DATASETS ====================
print("\n" + "="*80)
print("STEP 1: DOWNLOADING DATASETS TO D DRIVE")
print("="*80)

import kagglehub

datasets = {
    "heartbeat": "shayanfazeli/heartbeat",
    "ecg_dataset": "devavratatripathy/ecg-dataset",
    "drug_bank": "devildev89/drug-bank-5110",
    "skin_disease": "ahmedxc4/skin-ds"
}

downloaded = {}

for name, dataset_id in datasets.items():
    print(f"\n📥 Downloading: {name}")
    print(f"   Dataset ID: {dataset_id}")
    
    try:
        path = kagglehub.dataset_download(dataset_id)
        print(f"   ✅ Downloaded successfully")
        print(f"   Location: {path}")
        downloaded[name] = path
    except KeyboardInterrupt:
        print(f"   ⏸️ Download paused/cancelled")
        break
    except Exception as e:
        print(f"   ❌ Error: {str(e)[:100]}")

# ==================== STEP 2: COPY FROM C TO TARGET ====================
print("\n" + "="*80)
print("STEP 2: COPYING DATASETS TO TARGET LOCATION")
print("="*80)

def copy_dataset(src, dest_name):
    """Copy dataset to target directory"""
    dest = target_dir / dest_name
    
    print(f"\n📋 Copying: {dest_name}")
    print(f"   From: {src}")
    print(f"   To: {dest}")
    
    try:
        # Find versions folder
        versions = Path(src) / "versions"
        if versions.exists():
            version_dirs = sorted([d for d in versions.iterdir() if d.is_dir()])
            if version_dirs:
                latest = version_dirs[-1]
                src = latest
        
        if dest.exists():
            shutil.rmtree(dest)
        
        shutil.copytree(src, dest)
        print(f"   ✅ Copied successfully")
        return True
    except Exception as e:
        print(f"   ❌ Error: {str(e)[:80]}")
        return False

# Map downloaded paths to destination names
dataset_map = {
    "heartbeat": "heartbeat",
    "ecg_dataset": "ecg_dataset",
    "drug_bank": "drug_bank",
    "skin_disease": "skin_disease"
}

copied_datasets = {}
for name, path in downloaded.items():
    dest_name = dataset_map.get(name, name)
    if copy_dataset(path, dest_name):
        copied_datasets[name] = str(target_dir / dest_name)

# ==================== STEP 3: DOWNLOAD ISIC 2024 ====================
print("\n" + "="*80)
print("STEP 3: DOWNLOADING ISIC 2024 (OPTIONAL)")
print("="*80)

isic_path = target_dir / "isic_2024_challenge"
print(f"\n🧬 ISIC 2024 Challenge Dataset")
print(f"   Target: {isic_path}")

try:
    if not isic_path.exists():
        print("   Cloning from GitHub...")
        subprocess.run([
            'git', 'clone',
            'https://github.com/ISIC-Research/2024-challenge-dataset.git',
            str(isic_path)
        ], check=False, capture_output=True)
        print("   ✅ ISIC 2024 cloned")
    else:
        print("   ℹ️ Already exists")
except Exception as e:
    print(f"   ⚠️ Could not clone: {str(e)[:80]}")

# ==================== STEP 4: ORGANIZE MODELS ====================
print("\n" + "="*80)
print("STEP 4: ORGANIZING MODELS FOLDER")
print("="*80)

# Find all .py files in Ai Doctor root
model_files_to_move = []
for file in ai_doctor_path.glob("*.py"):
    if file.name not in ["app_fixed.py", "api_simple.py", "config.yaml"]:
        model_files_to_move.append(file)

print(f"\n🤖 Found {len(model_files_to_move)} model files to organize")

moved_count = 0
for file in model_files_to_move:
    dest = models_folder / file.name
    try:
        if dest.exists():
            dest.unlink()
        shutil.move(str(file), str(dest))
        print(f"   ✅ Moved: {file.name}")
        moved_count += 1
    except Exception as e:
        print(f"   ⚠️ Could not move {file.name}: {str(e)[:50]}")

# ==================== STEP 5: REMOVE UNNECESSARY FILES ====================
print("\n" + "="*80)
print("STEP 5: REMOVING UNNECESSARY FILES")
print("="*80)

# Files to delete
unnecessary_files = [
    "ADVANCED_FEATURES.md",
    "IMPLEMENTATION_COMPLETE.md",
    "IMPLEMENTATION_COMPLETE_ADVANCED.md",
    "DATASET_MODEL_LINKS.md",
    "DATASETS_SETUP.md",
    "DATASETS_QUICK_REFERENCE.md",
    "KAGGLE_DOWNLOAD_GUIDE.md",
    "FREE_MEDICAL_DATASETS.md",
    "COMPLETE_DATASETS_GUIDE.md",
    "SESSION_SUMMARY.md",
    "README_DATASETS.md",
    "START_HERE.md",
    "FINAL_SUMMARY.txt",
    "SYSTEM_OVERVIEW.txt",
    "IMPLEMENTATION_CHECKLIST.md",
    "DOWNLOADED_DATASETS_INFO.md",
    "DATASETS_AND_MODELS_COMPLETE.md",
    "download_all_datasets_to_d.py",
    "organize_datasets_and_models.py",
]

deleted_count = 0
print("\n📝 Removing documentation & script files:")

for filename in unnecessary_files:
    file_path = ai_doctor_path / filename
    if file_path.exists():
        try:
            file_path.unlink()
            print(f"   ✅ Deleted: {filename}")
            deleted_count += 1
        except Exception as e:
            print(f"   ⚠️ Could not delete {filename}: {str(e)[:50]}")

# ==================== STEP 6: CLEAN UP C DRIVE ====================
print("\n" + "="*80)
print("STEP 6: CLEANING UP C DRIVE CACHE")
print("="*80)

if c_cache.exists():
    print(f"\n🗑️ Removing C drive cache: {c_cache}")
    try:
        shutil.rmtree(c_cache)
        print("   ✅ C drive cache removed")
    except Exception as e:
        print(f"   ⚠️ Could not remove cache: {str(e)[:80]}")
else:
    print(f"   ℹ️ Cache not found")

# ==================== FINAL SUMMARY ====================
print("\n" + "="*80)
print("✅ SETUP COMPLETE")
print("="*80)

print(f"\n📊 Summary:")
print(f"   • Datasets downloaded: {len(downloaded)}")
print(f"   • Datasets copied to target: {len(copied_datasets)}")
print(f"   • Model files moved: {moved_count}")
print(f"   • Unnecessary files deleted: {deleted_count}")

print(f"\n📁 Datasets Location: {target_dir}")
print(f"📁 Models Location: {models_folder}")

print(f"\n📂 Available Datasets:")
for item in target_dir.iterdir():
    if item.is_dir():
        size_mb = sum(f.stat().st_size for f in item.rglob('*') if f.is_file()) / (1024*1024)
        print(f"   • {item.name} ({size_mb:.0f} MB)")

print(f"\n✨ Ready to use!")
print("="*80)
