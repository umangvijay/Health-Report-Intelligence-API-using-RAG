"""
Copy downloaded datasets from C cache to D drive
And move all Python model files to models folder
"""

import shutil
import os
from pathlib import Path
import glob

print("="*70)
print("ORGANIZING DATASETS AND MODELS")
print("="*70)

# Source and destination
c_cache = Path(r"C:\Users\ASUS\.cache\kagglehub\datasets")
d_destination = Path(r"D:\medical_datasets")
d_destination.mkdir(parents=True, exist_ok=True)

# Datasets to copy
datasets_to_copy = [
    ("shayanfazeli/heartbeat", "heartbeat"),
    ("devavratatripathy/ecg-dataset", "ecg_dataset"),
    ("devildev89/drug-bank-5110", "drug_bank"),
]

print("\n📦 COPYING DATASETS FROM C: TO D:")
print("-" * 70)

for source_path, dest_name in datasets_to_copy:
    source = c_cache / source_path
    if source.exists():
        dest = d_destination / dest_name
        print(f"\nCopying: {source_path}")
        print(f"From: {source}")
        print(f"To: {dest}")
        
        # Find the versions folder
        versions_dir = source / "versions"
        if versions_dir.exists():
            # Get the latest version
            version_folders = sorted([d for d in versions_dir.iterdir() if d.is_dir()])
            if version_folders:
                latest = version_folders[-1]
                try:
                    if dest.exists():
                        shutil.rmtree(dest)
                    shutil.copytree(latest, dest)
                    print(f"✅ Copied successfully")
                except Exception as e:
                    print(f"❌ Error: {str(e)[:100]}")
        else:
            try:
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(source, dest)
                print(f"✅ Copied successfully")
            except Exception as e:
                print(f"❌ Error: {str(e)[:100]}")
    else:
        print(f"⚠️ Source not found: {source_path}")

# Organize models folder
print("\n\n" + "="*70)
print("🤖 ORGANIZING MODELS FOLDER")
print("="*70)

ai_doctor_path = Path(r"d:\c++ homework\python\ml projects\Ai Doctor")
models_folder = ai_doctor_path / "models"
models_folder.mkdir(exist_ok=True)

# Find all .py model files in root and subfolders
print("\nSearching for model Python files...")

# Files to move (pattern matching)
model_files = [
    "improve_accuracy.py",
    "improve_accuracy_advanced.py",
    "self_learning.py",
    "ml_trainer.py",
    "download_all_models_simple.py",
    "import_all_models.py",
    "use_installed_models.py",
    "test_installation.py",
    "test_system.py",
    "test_backend.py",
    "test_gemini.py",
    "run_all_in_one.py"
]

moved_files = []
for filename in model_files:
    file_path = ai_doctor_path / filename
    if file_path.exists():
        dest_path = models_folder / filename
        try:
            shutil.move(str(file_path), str(dest_path))
            print(f"✅ Moved: {filename}")
            moved_files.append(filename)
        except Exception as e:
            print(f"⚠️ Could not move {filename}: {str(e)[:50]}")
    else:
        print(f"ℹ️ Not found: {filename}")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"\n✅ Datasets copied to: {d_destination}")
print(f"   • heartbeat")
print(f"   • ecg_dataset")
print(f"   • drug_bank")

print(f"\n✅ Model files moved to models folder: {models_folder}")
print(f"   Total files moved: {len(moved_files)}")
for f in moved_files:
    print(f"   • {f}")

print(f"\n✅ Organization complete!")
print("="*70)
