"""
Quick import script for MedMCQA dataset
Usage: python import_medmcqa.py
"""

import json
import os
from pathlib import Path
from tqdm import tqdm

def import_medmcqa():
    """Import and verify MedMCQA files"""
    
    data_dir = Path("ai_doctor_data/medmcqa")
    
    print("\n" + "="*70)
    print("📥 IMPORTING MEDMCQA DATASET")
    print("="*70)
    
    files = {
        "train": data_dir / "train.json",
        "test": data_dir / "test.json",
        "dev": data_dir / "dev.json"
    }
    
    results = {}
    
    for split, filepath in files.items():
        try:
            if not filepath.exists():
                print(f"\n❌ {split.upper()}: File not found at {filepath}")
                continue
            
            print(f"\n📂 Loading {split.upper()}...")
            
            # Try JSON first, then JSONL (one JSON object per line)
            data = []
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)  # Regular JSON array
            except json.JSONDecodeError:
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data.append(json.loads(line))
            
            # Handle both list and dict formats
            if isinstance(data, dict):
                num_samples = len(data)
                sample_key = list(data.keys())[0]
                sample = data[sample_key]
            else:
                num_samples = len(data)
                sample = data[0] if data else {}
            
            file_size_mb = filepath.stat().st_size / (1024 * 1024)
            
            print(f"   ✅ Loaded: {num_samples} questions")
            print(f"   📊 File size: {file_size_mb:.2f} MB")
            print(f"   📋 Sample question: {str(sample)[:100]}...")
            
            results[split] = {
                "status": "success",
                "samples": num_samples,
                "file_size_mb": file_size_mb,
                "path": str(filepath)
            }
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            results[split] = {
                "status": "error",
                "error": str(e)
            }
    
    # Summary
    print("\n" + "="*70)
    print("📊 IMPORT SUMMARY")
    print("="*70)
    
    total_samples = sum(r.get("samples", 0) for r in results.values() if r.get("status") == "success")
    total_size = sum(r.get("file_size_mb", 0) for r in results.values() if r.get("status") == "success")
    
    for split, result in results.items():
        status = "✅" if result["status"] == "success" else "❌"
        if result["status"] == "success":
            print(f"{status} {split.upper():6} - {result['samples']:6} samples ({result['file_size_mb']:6.2f} MB)")
        else:
            print(f"{status} {split.upper():6} - {result.get('error', 'Unknown error')}")
    
    print(f"\n{'='*70}")
    print(f"Total: {total_samples} questions | {total_size:.2f} MB")
    print(f"{'='*70}")
    
    # Save import log
    log_file = Path("ai_doctor_data/medmcqa_import.json")
    with open(log_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📝 Import log saved to: {log_file}")
    
    return results


def test_load():
    """Test loading MedMCQA in your app"""
    
    print("\n" + "="*70)
    print("🧪 TESTING DATASET LOADER")
    print("="*70)
    
    try:
        from load_datasets import DatasetLoader
        
        loader = DatasetLoader()
        
        # Test loading
        print("\n1. Loading training data...")
        train_data = loader.load_medmcqa('train')
        print(f"   ✅ Loaded {len(train_data)} questions")
        
        # Show sample
        print("\n2. Sample question:")
        sample = train_data[0]
        print(f"   {json.dumps(sample, indent=2)[:300]}...")
        
        # Test search
        print("\n3. Testing search (keyword: 'patient')...")
        results = loader.query_medmcqa('patient')
        print(f"   ✅ Found {len(results)} matching questions")
        
        print("\n✅ ALL TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return False


if __name__ == "__main__":
    # Import
    results = import_medmcqa()
    
    # Test
    if all(r.get("status") == "success" for r in results.values() if "status" in r):
        test_load()
    else:
        print("\n⚠️  Some files failed to import. Fix them first.")
