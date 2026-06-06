"""
Complete Dataset Importer - All Medical Datasets
Downloads and integrates ALL datasets into your models
Use: from models.all_datasets_importer import DatasetManager
"""

import os
import json
import requests
from pathlib import Path
from typing import Dict, List, Optional
from datasets import load_dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetManager:
    """Manages ALL medical datasets in one place"""
    
    def __init__(self, data_dir: str = "ai_doctor_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.datasets = {}
    
    # ==================== HUGGINGFACE DATASETS ====================
    
    def load_medqa(self, num_samples: int = 47000) -> Dict:
        """Load MedQA - 47K medical Q&A pairs"""
        try:
            logger.info("Loading MedQA dataset...")
            dataset = load_dataset("openlifescienceai/medqa")
            
            samples = []
            for i, item in enumerate(dataset['train']):
                if i >= num_samples:
                    break
                samples.append({
                    'question': item.get('question', ''),
                    'options': item.get('options', []),
                    'answer': item.get('answer', ''),
                    'source': 'medqa'
                })
            
            self.datasets['medqa'] = samples
            logger.info(f"✅ Loaded {len(samples)} MedQA samples")
            return {'success': True, 'count': len(samples), 'data': samples}
        except Exception as e:
            logger.error(f"❌ MedQA error: {e}")
            return {'success': False, 'error': str(e)}
    
    def load_pubmedqa(self, num_samples: int = 50000) -> Dict:
        """Load PubMedQA - 500K research Q&A pairs"""
        try:
            logger.info("Loading PubMedQA dataset...")
            dataset = load_dataset("pubmedqa", "pqa_labeled")
            
            samples = []
            for i, item in enumerate(dataset['train']):
                if i >= num_samples:
                    break
                samples.append({
                    'question': item.get('question', ''),
                    'context': item.get('context', []),
                    'answer': item.get('long_answer', ''),
                    'source': 'pubmedqa'
                })
            
            self.datasets['pubmedqa'] = samples
            logger.info(f"✅ Loaded {len(samples)} PubMedQA samples")
            return {'success': True, 'count': len(samples), 'data': samples}
        except Exception as e:
            logger.error(f"❌ PubMedQA error: {e}")
            return {'success': False, 'error': str(e)}
    
    # ==================== MANUAL DOWNLOAD DATASETS ====================
    
    def download_medmcqa(self) -> Dict:
        """
        MedMCQA - 194K multiple choice questions
        Manual: https://drive.google.com/uc?export=download&id=15VkJdq5eyWIkfb_aoD3oS8i4tScbHYky
        """
        medmcqa_dir = self.data_dir / "medmcqa"
        medmcqa_file = medmcqa_dir / "train.json"
        
        if medmcqa_file.exists():
            try:
                logger.info("Loading MedMCQA from local file...")
                with open(medmcqa_file, 'r') as f:
                    data = json.load(f)
                
                # Convert to standard format
                samples = []
                for item in data if isinstance(data, list) else [data]:
                    samples.append({
                        'question': item.get('question', ''),
                        'options': {
                            'a': item.get('opa', ''),
                            'b': item.get('opb', ''),
                            'c': item.get('opc', ''),
                            'd': item.get('opd', '')
                        },
                        'answer': item.get('cop', ''),
                        'explanation': item.get('exp', ''),
                        'source': 'medmcqa'
                    })
                
                self.datasets['medmcqa'] = samples
                logger.info(f"✅ Loaded MedMCQA: {len(samples)} samples")
                return {'success': True, 'count': len(samples), 'data': samples}
            except Exception as e:
                logger.error(f"Error loading MedMCQA: {e}")
                return {'success': False, 'error': str(e)}
        else:
            logger.warning("MedMCQA not found locally")
            return {
                'success': False,
                'error': 'Not downloaded',
                'instructions': [
                    '1. Download from: https://drive.google.com/uc?export=download&id=15VkJdq5eyWIkfb_aoD3oS8i4tScbHYky',
                    f'2. Extract to: {medmcqa_dir}',
                    '3. Ensure train.json exists',
                    '4. Run this function again'
                ],
                'auto_download': 'Visit the link above to download manually'
            }
    
    def load_drugbank(self) -> Dict:
        """
        DrugBank - 13K drugs, 900K interactions
        Download: https://go.drugbank.com/releases/latest
        """
        drugbank_dir = self.data_dir / "drugbank"
        
        # Try to find DrugBank files
        if drugbank_dir.exists():
            csv_files = list(drugbank_dir.glob("*.csv"))
            xml_files = list(drugbank_dir.glob("*.xml"))
            
            if csv_files or xml_files:
                logger.info(f"✅ Found DrugBank data: {len(csv_files)} CSV, {len(xml_files)} XML files")
                
                # Load CSV files if available
                drug_data = {}
                for csv_file in csv_files:
                    try:
                        import pandas as pd
                        df = pd.read_csv(csv_file)
                        drug_data[csv_file.stem] = df.to_dict('records')
                        logger.info(f"  Loaded {csv_file.name}: {len(df)} records")
                    except Exception as e:
                        logger.warning(f"  Could not load {csv_file.name}: {e}")
                
                if drug_data:
                    self.datasets['drugbank'] = drug_data
                    total_drugs = sum(len(v) for v in drug_data.values())
                    return {
                        'success': True,
                        'path': str(drugbank_dir),
                        'files': len(csv_files) + len(xml_files),
                        'total_drugs': total_drugs,
                        'data': drug_data
                    }
                else:
                    self.datasets['drugbank'] = {
                        'path': str(drugbank_dir),
                        'files': [str(f) for f in csv_files + xml_files]
                    }
                    return {
                        'success': True,
                        'path': str(drugbank_dir),
                        'files': len(csv_files) + len(xml_files),
                        'note': 'Files found but not parsed yet'
                    }
        
        return {
            'success': False,
            'error': 'Not downloaded',
            'instructions': [
                '1. Visit: https://go.drugbank.com/releases/latest',
                '2. Download free version (no login required)',
                f'3. Extract to: {drugbank_dir}',
                '4. Run this function again'
            ],
            'download_url': 'https://go.drugbank.com/releases/latest'
        }
    
    def load_bioasq(self) -> Dict:
        """
        BioASQ - 5M+ biomedical documents
        Website: http://www.bioasq.org/
        Download: http://participants-area.bioasq.org/datasets/
        """
        bioasq_dir = self.data_dir / "bioasq"
        
        if bioasq_dir.exists():
            json_files = list(bioasq_dir.glob("*.json"))
            if json_files:
                datasets = {}
                for json_file in json_files:
                    try:
                        with open(json_file, 'r') as f:
                            datasets[json_file.stem] = json.load(f)
                    except:
                        pass
                
                if datasets:
                    self.datasets['bioasq'] = datasets
                    logger.info(f"✅ Loaded BioASQ: {len(datasets)} files")
                    return {'success': True, 'count': len(datasets)}
        
        return {
            'success': False,
            'error': 'Not downloaded',
            'instructions': [
                '1. Register at: http://www.bioasq.org/',
                '2. Download from: http://participants-area.bioasq.org/datasets/',
                f'3. Extract to: {bioasq_dir}',
                '4. Run this function again'
            ]
        }
    
    def load_covid_chestxray(self) -> Dict:
        """
        COVID Chest X-ray Dataset
        GitHub: https://github.com/ieee8023/covid-chestxray-dataset
        """
        covid_dir = self.data_dir / "covid-chestxray"
        metadata_file = covid_dir / "metadata.csv"
        
        if metadata_file.exists():
            try:
                import pandas as pd
                df = pd.read_csv(metadata_file)
                self.datasets['covid_chestxray'] = df
                logger.info(f"✅ Loaded COVID Chest X-ray: {len(df)} images")
                return {'success': True, 'count': len(df), 'path': str(covid_dir)}
            except Exception as e:
                return {'success': False, 'error': str(e)}
        
        return {
            'success': False,
            'error': 'Not downloaded',
            'instructions': [
                '1. Clone: git clone https://github.com/ieee8023/covid-chestxray-dataset',
                f'2. Move to: {covid_dir}',
                '3. Run this function again'
            ]
        }
    
    def load_clinicalbert_data(self) -> Dict:
        """
        ClinicalBERT Data
        GitHub: https://github.com/EmilyAlsentzer/clinicalBERT/tree/master/data
        """
        clinicalbert_dir = self.data_dir / "clinicalbert"
        
        if clinicalbert_dir.exists():
            files = list(clinicalbert_dir.rglob("*.txt")) + list(clinicalbert_dir.rglob("*.json"))
            if files:
                logger.info(f"✅ Found ClinicalBERT data: {len(files)} files")
                self.datasets['clinicalbert'] = {'path': str(clinicalbert_dir), 'files': [str(f) for f in files]}
                return {'success': True, 'count': len(files)}
        
        return {
            'success': False,
            'error': 'Not downloaded',
            'instructions': [
                '1. Clone: git clone https://github.com/EmilyAlsentzer/clinicalBERT',
                f'2. Copy data folder to: {clinicalbert_dir}',
                '3. Run this function again'
            ]
        }
    
    def load_chestxray14(self) -> Dict:
        """
        NIH ChestX-ray14 - 112K chest X-rays
        GitHub: https://github.com/NIH-LCS/ChestX-ray14
        Direct: https://nihcc.app.box.com/v/ChestXray-NIHCC/
        """
        chestxray_dir = self.data_dir / "chestxray14"
        metadata_file = chestxray_dir / "Data_Entry_2017.csv"
        
        if metadata_file.exists():
            try:
                import pandas as pd
                df = pd.read_csv(metadata_file)
                self.datasets['chestxray14'] = df
                logger.info(f"✅ Loaded ChestX-ray14: {len(df)} images")
                return {'success': True, 'count': len(df), 'path': str(chestxray_dir)}
            except Exception as e:
                return {'success': False, 'error': str(e)}
        
        return {
            'success': False,
            'error': 'Not downloaded',
            'instructions': [
                '1. Download from: https://nihcc.app.box.com/v/ChestXray-NIHCC/',
                f'2. Extract to: {chestxray_dir}',
                '3. Run this function again',
                'Note: Large download (40GB)'
            ]
        }
    
    # ==================== QUERY METHODS ====================
    
    def query(self, query: str, dataset_name: str = 'all', top_k: int = 5) -> List[Dict]:
        """Query any loaded dataset"""
        results = []
        
        if dataset_name == 'all':
            # Search all datasets
            for name, data in self.datasets.items():
                if isinstance(data, list):
                    for item in data[:top_k]:
                        if isinstance(item, dict):
                            # Simple keyword matching
                            text = str(item.get('question', '')) + str(item.get('answer', ''))
                            if query.lower() in text.lower():
                                results.append({**item, 'dataset': name})
        else:
            # Search specific dataset
            if dataset_name in self.datasets:
                data = self.datasets[dataset_name]
                if isinstance(data, list):
                    for item in data[:top_k]:
                        if isinstance(item, dict):
                            text = str(item.get('question', '')) + str(item.get('answer', ''))
                            if query.lower() in text.lower():
                                results.append(item)
        
        return results[:top_k]
    
    def get_dataset(self, name: str):
        """Get a specific dataset"""
        return self.datasets.get(name)
    
    def list_datasets(self) -> List[str]:
        """List all loaded datasets"""
        return list(self.datasets.keys())
    
    def get_stats(self) -> Dict:
        """Get statistics about loaded datasets"""
        stats = {}
        for name, data in self.datasets.items():
            if isinstance(data, list):
                stats[name] = {'count': len(data), 'type': 'list'}
            elif isinstance(data, dict):
                stats[name] = {'type': 'dict', 'keys': list(data.keys())[:5]}
        return stats
    
    # ==================== LOAD ALL ====================
    
    def load_all_available(self) -> Dict:
        """Load all available datasets"""
        logger.info("="*70)
        logger.info("LOADING ALL AVAILABLE DATASETS")
        logger.info("="*70)
        
        results = {}
        
        # HuggingFace datasets (automatic - FREE)
        logger.info("\n[1] HuggingFace Datasets (FREE)...")
        results['medqa'] = self.load_medqa()
        results['pubmedqa'] = self.load_pubmedqa()
        results['medical_meadow'] = self.load_medical_meadow()
        results['healthsearchqa'] = self.load_healthsearchqa()
        results['medquad'] = self.load_medquad()
        
        # Manual datasets (if available)
        logger.info("\n[2] Manual Datasets...")
        results['medmcqa'] = self.download_medmcqa()
        results['drugbank'] = self.load_drugbank()
        results['bioasq'] = self.load_bioasq()
        results['covid_chestxray'] = self.load_covid_chestxray()
        results['clinicalbert'] = self.load_clinicalbert_data()
        results['chestxray14'] = self.load_chestxray14()
        results['mimic'] = self.load_mimic()
        
        # Summary
        loaded = sum(1 for r in results.values() if r.get('success'))
        logger.info(f"\n✅ Loaded {loaded}/{len(results)} datasets")
        
        return results
    
    def load_medical_meadow(self, num_samples: int = 10000) -> Dict:
        """
        Medical Meadow - 1.5M medical Q&A pairs (FREE)
        HuggingFace: medalpaca/medical_meadow_medical_flashcards
        """
        try:
            logger.info("Loading Medical Meadow dataset...")
            dataset = load_dataset("medalpaca/medical_meadow_medical_flashcards", split="train")
            
            samples = []
            for i, item in enumerate(dataset):
                if i >= num_samples:
                    break
                samples.append({
                    'question': item.get('input', ''),
                    'answer': item.get('output', ''),
                    'source': 'medical_meadow'
                })
            
            self.datasets['medical_meadow'] = samples
            logger.info(f"✅ Loaded {len(samples)} Medical Meadow samples")
            return {'success': True, 'count': len(samples), 'data': samples}
        except Exception as e:
            logger.error(f"❌ Medical Meadow error: {e}")
            return {'success': False, 'error': str(e)}
    
    def load_healthsearchqa(self, num_samples: int = 5000) -> Dict:
        """
        HealthSearchQA - Health search queries (FREE)
        HuggingFace: bigbio/healthsearchqa
        """
        try:
            logger.info("Loading HealthSearchQA dataset...")
            dataset = load_dataset("bigbio/healthsearchqa", split="train")
            
            samples = []
            for i, item in enumerate(dataset):
                if i >= num_samples:
                    break
                samples.append({
                    'question': item.get('question', ''),
                    'answer': item.get('answer', ''),
                    'source': 'healthsearchqa'
                })
            
            self.datasets['healthsearchqa'] = samples
            logger.info(f"✅ Loaded {len(samples)} HealthSearchQA samples")
            return {'success': True, 'count': len(samples), 'data': samples}
        except Exception as e:
            logger.error(f"❌ HealthSearchQA error: {e}")
            return {'success': False, 'error': str(e)}
    
    def load_medquad(self, num_samples: int = 10000) -> Dict:
        """
        MedQuAD - Medical Question Answering Dataset (FREE)
        HuggingFace: keivalya/MedQuad-MedicalQnADataset
        """
        try:
            logger.info("Loading MedQuAD dataset...")
            dataset = load_dataset("keivalya/MedQuad-MedicalQnADataset", split="train")
            
            samples = []
            for i, item in enumerate(dataset):
                if i >= num_samples:
                    break
                samples.append({
                    'question': item.get('Question', ''),
                    'answer': item.get('Answer', ''),
                    'source': 'medquad'
                })
            
            self.datasets['medquad'] = samples
            logger.info(f"✅ Loaded {len(samples)} MedQuAD samples")
            return {'success': True, 'count': len(samples), 'data': samples}
        except Exception as e:
            logger.error(f"❌ MedQuAD error: {e}")
            return {'success': False, 'error': str(e)}
    
    def load_mimic(self) -> Dict:
        """
        MIMIC-III - 46K real patient records (FREE but requires registration)
        Website: https://physionet.org/content/mimiciii/1.4/
        """
        mimic_dir = self.data_dir / "mimic-iii"
        
        if mimic_dir.exists():
            csv_files = list(mimic_dir.glob("*.csv"))
            if csv_files:
                logger.info(f"✅ Found MIMIC-III data: {len(csv_files)} files")
                self.datasets['mimic'] = {
                    'path': str(mimic_dir),
                    'files': [str(f) for f in csv_files]
                }
                return {
                    'success': True,
                    'path': str(mimic_dir),
                    'files': len(csv_files)
                }
        
        return {
            'success': False,
            'error': 'Not downloaded',
            'instructions': [
                '1. Register at: https://physionet.org/register/',
                '2. Complete CITI training (1-2 hours)',
                '3. Request access at: https://physionet.org/content/mimiciii/1.4/',
                f'4. Download and extract to: {mimic_dir}',
                '5. Run this function again'
            ],
            'note': 'FREE but requires registration and training'
        }


# ==================== EASY ACCESS ====================

# Global instance
_dataset_manager = None

def get_dataset_manager() -> DatasetManager:
    """Get global dataset manager instance"""
    global _dataset_manager
    if _dataset_manager is None:
        _dataset_manager = DatasetManager()
    return _dataset_manager


def load_all_datasets() -> Dict:
    """Load all available datasets"""
    manager = get_dataset_manager()
    return manager.load_all_available()


def query_datasets(query: str, dataset: str = 'all', top_k: int = 5) -> List[Dict]:
    """Query loaded datasets"""
    manager = get_dataset_manager()
    return manager.query(query, dataset, top_k)


def get_dataset(name: str):
    """Get specific dataset"""
    manager = get_dataset_manager()
    return manager.get_dataset(name)


# ==================== USAGE EXAMPLE ====================

if __name__ == "__main__":
    # Initialize
    manager = DatasetManager()
    
    # Load all available datasets
    results = manager.load_all_available()
    
    # Show stats
    print("\n" + "="*70)
    print("DATASET STATISTICS")
    print("="*70)
    stats = manager.get_stats()
    for name, stat in stats.items():
        print(f"{name}: {stat}")
    
    # Query example
    print("\n" + "="*70)
    print("QUERY EXAMPLE")
    print("="*70)
    results = manager.query("diabetes treatment", top_k=3)
    print(f"Found {len(results)} results")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result.get('question', 'N/A')[:100]}...")
