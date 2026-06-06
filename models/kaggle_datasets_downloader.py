"""
Kagglehub Medical Datasets Downloader
Download all free medical datasets with one command
"""

import kagglehub
from pathlib import Path
import json
from typing import Dict, List
import os


class MedicalDatasetsDownloader:
    """Download and manage Kaggle medical datasets"""
    
    def __init__(self, data_dir: str = "ai_doctor_data/kaggle_datasets"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # All free medical datasets
        self.datasets = {
            # Blood Report & Lab Data
            "diabetes": {
                "id": "mathchi/diabetes-data-set",
                "category": "Blood Report",
                "description": "768 patients, diabetes prediction features"
            },
            "heart_disease": {
                "id": "redwankarimsony/heart-disease-data",
                "category": "Blood Report",
                "description": "303 patients, heart disease indicators"
            },
            "liver_disease": {
                "id": "uciml/indian-liver-patient-records",
                "category": "Blood Report",
                "description": "500+ patients, liver function tests"
            },
            "kidney_disease": {
                "id": "mansoordaku/ckdisease",
                "category": "Blood Report",
                "description": "400 patients, kidney function tests"
            },
            "framingham": {
                "id": "amanajmera1/framingham-heart-study-dataset",
                "category": "Blood Report",
                "description": "4000+ patients, cardiovascular data"
            },
            
            # Chest X-Ray
            "chest_xray_pneumonia": {
                "id": "paultimothymooney/chest-xray-pneumonia",
                "category": "X-Ray",
                "description": "5,863 X-ray images - pneumonia detection"
            },
            "covid_xray": {
                "id": "tawsifurrahman/covid19-radiography-database",
                "category": "X-Ray",
                "description": "13,800+ X-rays - COVID-19, pneumonia, normal"
            },
            
            # Brain Imaging
            "brain_mri_tumor": {
                "id": "sartajbhuvaji/brain-tumor-classification-mri",
                "category": "MRI",
                "description": "3,000+ brain MRI scans - tumor classification"
            },
            "alzheimer_mri": {
                "id": "uraninjo/augmented-alzheimer-mri-dataset",
                "category": "MRI",
                "description": "6,400+ MRI scans - Alzheimer's stages"
            },
            
            # Lung/CT
            "lung_ct": {
                "id": "kmader/finding-lungs-in-ct-data",
                "category": "CT Scan",
                "description": "1,010 CT scans - lung nodule detection"
            },
            
            # Skin Diseases
            "melanoma_ham10000": {
                "id": "kmader/skin-cancer-mnist-ham10000",
                "category": "Skin",
                "description": "10,015 dermatoscopic images - melanoma & benign"
            },
            "acne_dataset": {
                "id": "brsdincer/acne-dataset",
                "category": "Skin",
                "description": "2,500+ acne photos - severity classification"
            },
            "dermatology": {
                "id": "uciml/dermatology-dataset",
                "category": "Skin",
                "description": "366 cases - 6 skin diseases"
            },
            
            # Eye Diseases
            "diabetic_retinopathy": {
                "id": "mariaherrerot/eyepacs",
                "category": "Eye",
                "description": "35,000+ fundus images - diabetic retinopathy"
            },
            "cataracts": {
                "id": "jr2ngb/cataractdata",
                "category": "Eye",
                "description": "4,000+ eye images - cataract detection"
            },
            "glaucoma": {
                "id": "sshikamaru/glaucoma-detection",
                "category": "Eye",
                "description": "1,000+ fundus images - glaucoma detection"
            },
            
            # Other Diseases
            "covid_clinical": {
                "id": "maksimeren/covid19-dataset-01-20-2021",
                "category": "Clinical",
                "description": "400,000+ patient records - COVID-19 clinical data"
            },
            "dental_cavity": {
                "id": "berkozbek/cavity-detection-dataset",
                "category": "Dental",
                "description": "5,000+ X-rays - cavity detection"
            },
            "bone_fracture": {
                "id": "bmartinez1/bone-fractures-classification-dataset",
                "category": "Orthopedic",
                "description": "3,000+ X-rays - fracture classification"
            },
            # Drug datasets
            "drug_dataset": {
                "id": "shaiksha19/drug-dataset",
                "category": "Drug",
                "description": "Drug dataset for medicine/drug analysis"
            },
            "drug_classification": {
                "id": "prathamtripathi/drug-classification",
                "category": "Drug",
                "description": "Drug classification dataset"
            },
        }
    
    def download_dataset(self, dataset_name: str, verbose: bool = True) -> str:
        """
        Download a single dataset
        
        Args:
            dataset_name: Name of dataset to download
            verbose: Print progress
        
        Returns:
            Path to downloaded dataset
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found. Available: {list(self.datasets.keys())}")
        
        dataset_info = self.datasets[dataset_name]
        dataset_id = dataset_info['id']
        
        try:
            if verbose:
                print(f"\nDownloading {dataset_name}...")
                print(f"  Description: {dataset_info['description']}")
                print(f"  ID: {dataset_id}")
            
            path = kagglehub.dataset_download(dataset_id)
            
            if verbose:
                print(f"  Status: OK")
                print(f"  Path: {path}")
            
            return path
            
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            return None
    
    def download_category(self, category: str) -> Dict[str, str]:
        """
        Download all datasets in a category
        
        Args:
            category: Category name (Blood Report, X-Ray, MRI, Skin, Eye, etc.)
        
        Returns:
            Dict of dataset_name -> path
        """
        results = {}
        datasets_in_category = {
            name: info for name, info in self.datasets.items()
            if info['category'] == category
        }
        
        if not datasets_in_category:
            print(f"No datasets found in category: {category}")
            return results
        
        print(f"\nDownloading {len(datasets_in_category)} datasets from '{category}'...")
        print("="*70)
        
        for name in datasets_in_category:
            path = self.download_dataset(name, verbose=True)
            if path:
                results[name] = path
        
        return results
    
    def download_all(self, resume: bool = True) -> Dict[str, str]:
        """
        Download all datasets
        
        Args:
            resume: Skip already downloaded datasets
        
        Returns:
            Dict of dataset_name -> path
        """
        results = {}
        
        print("\n" + "="*70)
        print("DOWNLOADING ALL MEDICAL DATASETS")
        print("="*70)
        print(f"Total datasets: {len(self.datasets)}")
        print(f"Categories: {len(set(info['category'] for info in self.datasets.values()))}")
        
        for name, info in self.datasets.items():
            print(f"\n[{name}]")
            path = self.download_dataset(name, verbose=True)
            if path:
                results[name] = path
        
        # Save manifest
        self._save_manifest(results)
        
        return results
    
    def list_datasets(self) -> None:
        """List all available datasets"""
        print("\nAVAILABLE MEDICAL DATASETS")
        print("="*70)
        
        # Group by category
        categories = {}
        for name, info in self.datasets.items():
            cat = info['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append((name, info))
        
        # Print by category
        for category in sorted(categories.keys()):
            print(f"\n{category}:")
            for name, info in categories[category]:
                print(f"  - {name}")
                print(f"    {info['description']}")
    
    def get_statistics(self) -> Dict:
        """Get statistics about available datasets"""
        categories = {}
        for info in self.datasets.values():
            cat = info['category']
            categories[cat] = categories.get(cat, 0) + 1
        
        return {
            "total_datasets": len(self.datasets),
            "categories": categories,
            "dataset_names": list(self.datasets.keys())
        }
    
    def _save_manifest(self, results: Dict[str, str]) -> None:
        """Save download manifest"""
        manifest = {
            "timestamp": str(Path.ctime(Path.cwd())),
            "downloads": results,
            "datasets_info": {
                name: {
                    "id": info['id'],
                    "category": info['category'],
                    "description": info['description']
                }
                for name, info in self.datasets.items()
                if name in results
            }
        }
        
        manifest_path = self.data_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"\nManifest saved to: {manifest_path}")


# Quick download function
def quick_download(dataset_name: str):
    """Quick download single dataset"""
    downloader = MedicalDatasetsDownloader()
    return downloader.download_dataset(dataset_name)


# Test function
if __name__ == "__main__":
    downloader = MedicalDatasetsDownloader()
    
    print("\n" + "="*70)
    print("MEDICAL DATASETS DOWNLOADER")
    print("="*70)
    
    # Show all available
    downloader.list_datasets()
    
    # Show statistics
    stats = downloader.get_statistics()
    print("\n" + "="*70)
    print("STATISTICS")
    print("="*70)
    print(f"Total Datasets: {stats['total_datasets']}")
    print("\nBy Category:")
    for cat, count in sorted(stats['categories'].items()):
        print(f"  {cat}: {count}")
    
    # Example: Download specific datasets
    print("\n" + "="*70)
    print("EXAMPLE DOWNLOADS")
    print("="*70)
    print("\nTo download, use:")
    print("  downloader = MedicalDatasetsDownloader()")
    print("  downloader.download_dataset('diabetes')")
    print("  downloader.download_category('X-Ray')")
    print("  downloader.download_all()")
