"""
Simple dataset loader for MedMCQA and DrugBank
Use after manually downloading the datasets
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any


class DatasetLoader:
    """Load MedMCQA and DrugBank datasets"""
    
    def __init__(self, data_dir: str = "ai_doctor_data"):
        self.data_dir = Path(data_dir)
        self.medmcqa_dir = self.data_dir / "medmcqa"
        self.drugbank_dir = self.data_dir / "drugbank"
    
    def load_medmcqa(self, split: str = "train") -> List[Dict[str, Any]]:
        """
        Load MedMCQA dataset
        
        Args:
            split: 'train', 'test', or 'dev'
        
        Returns:
            List of questions
        """
        file_path = self.medmcqa_dir / f"{split}.json"
        
        if not file_path.exists():
            raise FileNotFoundError(f"MedMCQA {split} file not found at {file_path}")
        
        questions = []
        
        try:
            # Try reading as regular JSON array first
            with open(file_path, 'r', encoding='utf-8') as f:
                questions = json.load(f)
        except json.JSONDecodeError:
            # If that fails, try reading as JSONL (one JSON object per line)
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        questions.append(json.loads(line))
        
        print(f"✅ Loaded {len(questions)} MedMCQA {split} questions")
        return questions
    
    def load_drugbank(self) -> Dict[str, Any]:
        """
        Load DrugBank dataset
        
        Returns:
            Drug database dictionary
        """
        # Try to find drugbank file
        json_path = self.drugbank_dir / "drugbank.json"
        xml_path = self.drugbank_dir / "drugbank.xml"
        
        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                drugbank = json.load(f)
            print(f"✅ Loaded DrugBank from JSON")
            return drugbank
        
        elif xml_path.exists():
            try:
                import xml.etree.ElementTree as ET
                tree = ET.parse(xml_path)
                root = tree.getroot()
                print(f"✅ Loaded DrugBank from XML")
                return {"root": root}
            except ImportError:
                raise ImportError("Install 'lxml' to parse XML: pip install lxml")
        else:
            raise FileNotFoundError(f"DrugBank file not found at {self.drugbank_dir}")
    
    def query_medmcqa(self, query: str, split: str = "train") -> List[Dict[str, Any]]:
        """
        Search MedMCQA questions by keyword
        
        Args:
            query: Search term
            split: 'train', 'test', or 'dev'
        
        Returns:
            Matching questions
        """
        questions = self.load_medmcqa(split)
        
        results = [
            q for q in questions 
            if query.lower() in str(q).lower()
        ]
        
        print(f"Found {len(results)} questions matching '{query}'")
        return results
    
    def get_drug_info(self, drug_name: str) -> Dict[str, Any]:
        """
        Get drug information from DrugBank
        
        Args:
            drug_name: Name of drug
        
        Returns:
            Drug information
        """
        drugbank = self.load_drugbank()
        
        # Simple search (adapt based on actual structure)
        if isinstance(drugbank, dict) and 'drugs' in drugbank:
            for drug in drugbank['drugs']:
                if drug_name.lower() in str(drug).lower():
                    return drug
        
        return {"error": f"Drug '{drug_name}' not found"}


# Example usage
if __name__ == "__main__":
    loader = DatasetLoader()
    
    print("\n" + "="*70)
    print("Dataset Loader - Usage Examples")
    print("="*70)
    
    print("\n1. Loading MedMCQA training questions:")
    print("   loader = DatasetLoader()")
    print("   questions = loader.load_medmcqa('train')")
    print("   print(questions[0])  # First question")
    
    print("\n2. Searching MedMCQA:")
    print("   results = loader.query_medmcqa('diabetes')")
    
    print("\n3. Loading DrugBank:")
    print("   drugbank = loader.load_drugbank()")
    print("   drug_info = loader.get_drug_info('aspirin')")
    
    print("\n" + "="*70)
    print("⚠️  NOTE: Download datasets first!")
    print("="*70)
    print("\nMedMCQA:")
    print("  1. Download: https://drive.google.com/uc?export=download&id=15VkJdq5eyWIkfb_aoD3oS8i4tScbHYky")
    print("  2. Extract ZIP to: ai_doctor_data/medmcqa/")
    print("  3. Ensure train.json, test.json, dev.json exist")
    
    print("\nDrugBank:")
    print("  1. Visit: https://go.drugbank.com/releases/latest")
    print("  2. Download drugbank.xml or drugbank.json")
    print("  3. Save to: ai_doctor_data/drugbank/")
    print("="*70 + "\n")
