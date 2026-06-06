"""
COMPLETE DATASET & MODEL INSTALLER
Installs all datasets and models from DATASET_MODEL_LINKS.md
Uses HuggingFace token for authentication
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Any
import json
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CompleteInstaller:
    """Install all medical datasets and models"""
    
    def __init__(self, hf_token: str = None, data_dir: str = "ai_doctor_data"):
        """
        Initialize installer
        
        Args:
            hf_token: HuggingFace API token
            data_dir: Directory for datasets
        """
        self.hf_token = hf_token or os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_TOKEN')
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.installation_log = {
            "timestamp": datetime.now().isoformat(),
            "datasets": {},
            "models": {},
            "errors": []
        }
        
        if not self.hf_token:
            logger.warning("⚠️  No HF_TOKEN found. Some models may require authentication.")
            logger.info("Set HF_TOKEN in .env file or environment variable")
        else:
            logger.info("✅ HuggingFace token found")
    
    def install_dependencies(self):
        """Install required Python packages"""
        logger.info("="*70)
        logger.info("STEP 1: Installing Dependencies")
        logger.info("="*70)
        
        try:
            import subprocess
            
            packages = [
                "transformers>=4.35.0",
                "torch>=2.0.0",
                "datasets>=2.14.0",
                "sentence-transformers>=2.2.2",
                "chromadb>=0.4.0",
                "huggingface-hub>=0.19.0",
                "bitsandbytes>=0.42.0",
                "accelerate>=0.24.0",
                "tqdm",
                "pandas",
                "numpy",
                "scikit-learn"
            ]
            
            logger.info(f"Installing {len(packages)} packages...")
            
            for package in packages:
                logger.info(f"  Installing {package}...")
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "-q", package
                ])
            
            logger.info("✅ All dependencies installed")
            return True
        except Exception as e:
            logger.error(f"❌ Error installing dependencies: {e}")
            self.installation_log["errors"].append(f"Dependencies: {str(e)}")
            return False
    
    def install_huggingface_datasets(self):
        """Install all free HuggingFace datasets"""
        logger.info("\n" + "="*70)
        logger.info("STEP 2: Installing HuggingFace Datasets")
        logger.info("="*70)
        
        try:
            from datasets import load_dataset
            import json
            
            datasets_to_load = [
                {
                    "name": "MedQA",
                    "repo": "openlifescienceai/medqa",
                    "description": "47K medical Q&A pairs",
                    "size": "200MB"
                },
                {
                    "name": "PubMedQA",
                    "repo": "pubmedqa/pubmedqa",
                    "description": "500K research Q&A pairs",
                    "size": "300MB"
                }
            ]
            
            for dataset_info in datasets_to_load:
                try:
                    logger.info(f"\n[{dataset_info['name']}] {dataset_info['description']} ({dataset_info['size']})")
                    logger.info(f"  Downloading from HuggingFace...")
                    
                    dataset = load_dataset(dataset_info['repo'], trust_remote_code=True)
                    
                    # Save to disk
                    dataset_dir = self.data_dir / "huggingface_datasets" / dataset_info['name']
                    dataset_dir.mkdir(parents=True, exist_ok=True)
                    dataset.save_to_disk(str(dataset_dir))
                    
                    num_samples = len(dataset.get('train', dataset))
                    logger.info(f"  ✅ Downloaded {num_samples} samples")
                    
                    self.installation_log["datasets"][dataset_info['name']] = {
                        "status": "success",
                        "samples": num_samples,
                        "location": str(dataset_dir)
                    }
                except Exception as e:
                    logger.warning(f"  ⚠️  {dataset_info['name']}: {str(e)}")
                    logger.info(f"     (Will be available for manual setup)")
                    self.installation_log["datasets"][dataset_info['name']] = {
                        "status": "skipped",
                        "note": str(e)
                    }
            
            logger.info("\n✅ HuggingFace datasets check complete")
            return True
        except Exception as e:
            logger.error(f"❌ Error with HuggingFace datasets: {e}")
            self.installation_log["errors"].append(f"HuggingFace datasets: {str(e)}")
            return False
    
    def install_models(self, skip_large: bool = False):
        """Install all medical models"""
        logger.info("\n" + "="*70)
        logger.info("STEP 3: Installing Medical Models")
        logger.info("="*70)
        
        try:
            from models.model_loader import MedicalModelLoader
            
            loader = MedicalModelLoader(hf_token=self.hf_token)
            
            # Define models to install
            models_to_install = [
                {
                    "key": "clinical_bert",
                    "name": "Clinical-BERT",
                    "size": "340MB",
                    "priority": "high"
                },
                {
                    "key": "biobert",
                    "name": "BioBERT",
                    "size": "410MB",
                    "priority": "high"
                },
                {
                    "key": "pubmedbert",
                    "name": "PubMedBERT",
                    "size": "420MB",
                    "priority": "high"
                },
                {
                    "key": "scibert",
                    "name": "SciBERT",
                    "size": "410MB",
                    "priority": "medium"
                },
                {
                    "key": "biogpt",
                    "name": "BioGPT",
                    "size": "350MB",
                    "priority": "medium"
                },
                {
                    "key": "mistral_7b",
                    "name": "Mistral-7B",
                    "size": "7B params",
                    "priority": "low",
                    "large": True
                }
            ]
            
            for model_info in models_to_install:
                # Skip large models if requested
                if skip_large and model_info.get("large", False):
                    logger.info(f"\n[{model_info['name']}] Skipping (large model)")
                    continue
                
                try:
                    logger.info(f"\n[{model_info['name']}] {model_info['size']} - Priority: {model_info['priority']}")
                    logger.info(f"  Downloading and caching...")
                    
                    result = loader.load_model(
                        model_info['key'],
                        quantized=model_info.get("large", False)
                    )
                    
                    if result.get('loaded'):
                        logger.info(f"  ✅ Successfully loaded")
                        logger.info(f"     Model ID: {result['model_id']}")
                        logger.info(f"     Device: {result['device']}")
                        
                        self.installation_log["models"][model_info['name']] = {
                            "status": "success",
                            "model_id": result['model_id'],
                            "device": result['device']
                        }
                        
                        # Unload to free memory for next model
                        loader.unload_model(model_info['key'])
                    else:
                        logger.warning(f"  ⚠️  Failed: {result.get('error', 'Unknown error')}")
                        self.installation_log["models"][model_info['name']] = {
                            "status": "error",
                            "error": result.get('error', 'Unknown error')
                        }
                except Exception as e:
                    logger.error(f"  ❌ Error: {e}")
                    self.installation_log["models"][model_info['name']] = {
                        "status": "error",
                        "error": str(e)
                    }
            
            logger.info("\n✅ Model installation complete")
            return True
        except Exception as e:
            logger.error(f"❌ Error installing models: {e}")
            self.installation_log["errors"].append(f"Models: {str(e)}")
            return False
    
    def download_manual_datasets(self):
        """Provide instructions for manual dataset downloads"""
        logger.info("\n" + "="*70)
        logger.info("STEP 4: Manual Dataset Downloads")
        logger.info("="*70)
        
        manual_datasets = [
            {
                "name": "MedMCQA",
                "url": "https://drive.google.com/uc?export=download&id=15VkJdq5eyWIkfb_aoD3oS8i4tScbHYky",
                "size": "400MB",
                "description": "194K multiple choice questions",
                "instructions": [
                    "1. Click the URL above",
                    "2. Download ZIP file",
                    f"3. Extract to: {self.data_dir / 'medmcqa'}",
                    "4. Run: loader.load_medmcqa('path/to/train.json')"
                ]
            },
            {
                "name": "MIMIC-III",
                "url": "https://physionet.org/content/mimiciii/1.4/",
                "size": "6-8GB",
                "description": "46K real patient records",
                "instructions": [
                    "1. Register at: https://physionet.org/user/register/",
                    "2. Complete CITI training (1-2 hours)",
                    "3. Request access at URL above",
                    "4. Download after approval (instant)",
                    f"5. Extract to: {self.data_dir / 'mimic-iii'}"
                ],
                "note": "⚠️  Requires PhysioNet registration (free)"
            },
            {
                "name": "DrugBank",
                "url": "https://go.drugbank.com/releases/latest",
                "size": "500MB",
                "description": "13K drugs, 900K interactions",
                "instructions": [
                    "1. Visit URL above",
                    "2. Download free version (no login required)",
                    f"3. Save to: {self.data_dir / 'drugbank'}",
                    "4. Use for drug interaction checking"
                ]
            },
            {
                "name": "NIH ChestX-ray14",
                "url": "https://nihcc.app.box.com/v/ChestXray-NIHCC/",
                "size": "40GB",
                "description": "112K chest X-ray images",
                "instructions": [
                    "1. Visit URL above",
                    "2. Download image archives",
                    f"3. Extract to: {self.data_dir / 'chestxray14'}",
                    "4. Use for radiology AI"
                ],
                "note": "⚠️  Large download (40GB)"
            }
        ]
        
        logger.info("\nThe following datasets require manual download:\n")
        
        for dataset in manual_datasets:
            logger.info(f"📦 {dataset['name']} - {dataset['description']} ({dataset['size']})")
            logger.info(f"   URL: {dataset['url']}")
            if dataset.get('note'):
                logger.info(f"   {dataset['note']}")
            logger.info("   Instructions:")
            for instruction in dataset['instructions']:
                logger.info(f"      {instruction}")
            logger.info("")
        
        self.installation_log["manual_datasets"] = manual_datasets
        return True
    
    def verify_installation(self):
        """Verify all installations"""
        logger.info("\n" + "="*70)
        logger.info("STEP 5: Verification")
        logger.info("="*70)
        
        # Count successes
        datasets_success = sum(1 for d in self.installation_log["datasets"].values() if d.get("status") == "success")
        datasets_total = len(self.installation_log["datasets"])
        
        models_success = sum(1 for m in self.installation_log["models"].values() if m.get("status") == "success")
        models_total = len(self.installation_log["models"])
        
        logger.info(f"\n📊 Installation Summary:")
        logger.info(f"   Datasets: {datasets_success}/{datasets_total} successful")
        logger.info(f"   Models: {models_success}/{models_total} successful")
        logger.info(f"   Errors: {len(self.installation_log['errors'])}")
        
        if self.installation_log['errors']:
            logger.info(f"\n⚠️  Errors encountered:")
            for error in self.installation_log['errors']:
                logger.info(f"   - {error}")
        
        # Save log
        log_file = self.data_dir / "installation_log.json"
        with open(log_file, 'w') as f:
            json.dump(self.installation_log, f, indent=2)
        logger.info(f"\n📝 Installation log saved to: {log_file}")
        
        return datasets_success > 0 and models_success > 0
    
    def run_full_installation(self, skip_large_models: bool = False):
        """Run complete installation process"""
        logger.info("\n" + "="*70)
        logger.info("🚀 COMPLETE MEDICAL AI INSTALLATION - ALL MODELS & DATASETS")
        logger.info("="*70)
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"HF Token: {'✅ Found' if self.hf_token else '❌ Not found'}")
        logger.info(f"Installing: ALL models (including large ones)")
        logger.info("="*70)
        
        # Step 1: Dependencies
        if not self.install_dependencies():
            logger.error("❌ Dependency installation failed")
            return False
        
        # Step 2: HuggingFace datasets
        self.install_huggingface_datasets()
        
        # Step 3: Models
        self.install_models(skip_large=skip_large_models)
        
        # Step 4: Manual dataset instructions
        self.download_manual_datasets()
        
        # Step 5: Verification
        success = self.verify_installation()
        
        if success:
            logger.info("\n" + "="*70)
            logger.info("✅ INSTALLATION COMPLETE!")
            logger.info("="*70)
            logger.info("\nNext steps:")
            logger.info("1. Review manual dataset downloads above")
            logger.info("2. Run the app: python app_fixed.py")
            logger.info("="*70)
        else:
            logger.warning("\n" + "="*70)
            logger.warning("⚠️  INSTALLATION COMPLETED WITH ERRORS")
            logger.warning("="*70)
            logger.warning("Check the log file for details")
        
        return success


def main():
    """Main installation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Install all medical datasets and models")
    parser.add_argument("--hf-token", type=str, help="HuggingFace API token")
    parser.add_argument("--data-dir", type=str, default="ai_doctor_data", help="Data directory")
    
    args = parser.parse_args()
    
    # Get token from args, env, or .env file
    hf_token = args.hf_token or os.getenv('HF_TOKEN')
    
    if not hf_token:
        # Try to load from .env file
        try:
            from dotenv import load_dotenv
            load_dotenv()
            hf_token = os.getenv('HF_TOKEN')
        except:
            pass
    
    # Create installer
    installer = CompleteInstaller(hf_token=hf_token, data_dir=args.data_dir)
    
    # Run FULL installation (no skip-large option)
    success = installer.run_full_installation(skip_large_models=False)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
