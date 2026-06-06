"""
EXPANDED MODEL LOADER
=====================
Loads ALL available free medical models and datasets

NEW MODELS ADDED:
- PubMedBERT (microsoft/BiomedNLP-PubMedBERT-base-uncased)
- BioLinkBERT (michiyasunaga/BioLinkBERT-base)
- BioLinkBERT-Large (michiyasunaga/BioLinkBERT-large)
- ClinicalT5 (luqh/ClinicalT5-base)
- RadBERT (StanfordAIMI/RadBERT)
- GatorTron-base (UFNLP/gatortron-base)
- MedBERT (Charangan/MedBERT)
- BlueBERT (bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12)
- BioClinicalBERT (emilyalsentzer/Bio_ClinicalBERT)

NEW DATASETS:
- MedMCQA (medmcqa)
- PubMedQA (pubmed_qa)
- MedQA (bigbio/med_qa)
- BioASQ (bioasq)
- MIMIC-III (requires PhysioNet access)
- ChestX-ray14 (NIH)
"""

import os
import torch
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

HF_TOKEN = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_TOKEN')


# ============================================================================
# ALL AVAILABLE MEDICAL MODELS
# ============================================================================

ALL_MEDICAL_MODELS = {
    # ============ BERT-based Models ============
    "pubmedbert": {
        "model_id": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "type": "encoder",
        "task": "fill-mask",
        "description": "BERT trained on PubMed abstracts and full-text",
        "size": "110M",
        "recommended_for": ["medical NER", "classification", "QA"]
    },
    "biolinkbert": {
        "model_id": "michiyasunaga/BioLinkBERT-base",
        "type": "encoder",
        "task": "fill-mask",
        "description": "BERT with link prediction for biomedical text",
        "size": "110M",
        "recommended_for": ["relation extraction", "knowledge graphs"]
    },
    "biolinkbert_large": {
        "model_id": "michiyasunaga/BioLinkBERT-large",
        "type": "encoder",
        "task": "fill-mask",
        "description": "Large BioLinkBERT",
        "size": "340M",
        "recommended_for": ["high-accuracy tasks"]
    },
    "bioclinicalbert": {
        "model_id": "emilyalsentzer/Bio_ClinicalBERT",
        "type": "encoder",
        "task": "fill-mask",
        "description": "BERT trained on clinical notes (MIMIC-III)",
        "size": "110M",
        "recommended_for": ["clinical NER", "EHR analysis"]
    },
    "bluebert": {
        "model_id": "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12",
        "type": "encoder",
        "task": "fill-mask",
        "description": "BERT trained on PubMed + MIMIC-III",
        "size": "110M",
        "recommended_for": ["clinical + biomedical tasks"]
    },
    "medbert": {
        "model_id": "Charangan/MedBERT",
        "type": "encoder",
        "task": "fill-mask",
        "description": "Medical BERT for clinical text",
        "size": "110M",
        "recommended_for": ["medical text classification"]
    },
    "radbert": {
        "model_id": "StanfordAIMI/RadBERT",
        "type": "encoder",
        "task": "fill-mask",
        "description": "BERT for radiology reports",
        "size": "110M",
        "recommended_for": ["radiology NLP", "X-ray reports"]
    },
    "gatortron_base": {
        "model_id": "UFNLP/gatortron-base",
        "type": "encoder",
        "task": "fill-mask",
        "description": "Large clinical language model (UF Health)",
        "size": "345M",
        "recommended_for": ["clinical NLP", "EHR"]
    },
    
    # ============ T5/Generation Models ============
    "clinical_t5": {
        "model_id": "luqh/ClinicalT5-base",
        "type": "seq2seq",
        "task": "text2text-generation",
        "description": "T5 for clinical text generation",
        "size": "220M",
        "recommended_for": ["summarization", "question answering"]
    },
    "flan_t5_medical": {
        "model_id": "google/flan-t5-base",
        "type": "seq2seq",
        "task": "text2text-generation",
        "description": "Instruction-tuned T5 (works well for medical)",
        "size": "250M",
        "recommended_for": ["instruction following", "QA"]
    },
    
    # ============ GPT/Causal Models ============
    "biogpt": {
        "model_id": "microsoft/biogpt",
        "type": "causal",
        "task": "text-generation",
        "description": "GPT for biomedical text generation",
        "size": "347M",
        "recommended_for": ["text generation", "relation extraction"]
    },
    "biogpt_large": {
        "model_id": "microsoft/biogpt-large",
        "type": "causal",
        "task": "text-generation",
        "description": "Large BioGPT",
        "size": "1.5B",
        "recommended_for": ["high-quality generation"]
    },
    
    # ============ Specialized Models ============
    "biomedclip": {
        "model_id": "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        "type": "vision-text",
        "task": "image-classification",
        "description": "CLIP for medical images",
        "size": "400M",
        "recommended_for": ["medical image analysis", "X-ray classification"]
    },
    "medclip": {
        "model_id": "flaviagiammarino/pubmed-clip-vit-base-patch32",
        "type": "vision-text",
        "task": "image-classification",
        "description": "CLIP trained on PubMed images",
        "size": "150M",
        "recommended_for": ["medical image search"]
    },
    
    # ============ Large LLMs (via API or quantized) ============
    "meditron_7b": {
        "model_id": "epfl-llm/meditron-7b",
        "type": "causal",
        "task": "text-generation",
        "description": "7B medical LLM",
        "size": "7B",
        "recommended_for": ["medical consultation", "diagnosis"],
        "requires_gpu": True,
        "min_vram_gb": 8
    },
    "meditron_70b": {
        "model_id": "epfl-llm/meditron-70b",
        "type": "causal",
        "task": "text-generation",
        "description": "70B medical specialist LLM",
        "size": "70B",
        "recommended_for": ["expert medical consultation"],
        "api_only": True
    },
    "mistral_7b": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "type": "causal",
        "task": "text-generation",
        "description": "General instruction-following LLM",
        "size": "7B",
        "recommended_for": ["general QA"],
        "requires_gpu": True,
        "min_vram_gb": 8
    },
    "llama2_7b": {
        "model_id": "meta-llama/Llama-2-7b-chat-hf",
        "type": "causal",
        "task": "text-generation",
        "description": "Meta's LLaMA 2 chat model",
        "size": "7B",
        "recommended_for": ["conversational AI"],
        "requires_gpu": True,
        "min_vram_gb": 8
    }
}


# ============================================================================
# ALL AVAILABLE MEDICAL DATASETS
# ============================================================================

ALL_MEDICAL_DATASETS = {
    # ============ Question Answering ============
    "medqa": {
        "dataset_id": "bigbio/med_qa",
        "alt_id": "openlifescienceai/medqa",
        "description": "Medical licensing exam questions (USMLE)",
        "size": "47K questions",
        "task": "multiple choice QA",
        "load_code": 'load_dataset("bigbio/med_qa", "med_qa_en_bigbio_qa")'
    },
    "medmcqa": {
        "dataset_id": "medmcqa",
        "description": "Medical MCQ from AIIMS/NEET",
        "size": "194K questions",
        "task": "multiple choice QA",
        "load_code": 'load_dataset("medmcqa")'
    },
    "pubmedqa": {
        "dataset_id": "pubmed_qa",
        "alt_id": "bigbio/pubmed_qa",
        "description": "QA from PubMed abstracts",
        "size": "273K questions",
        "task": "yes/no/maybe QA",
        "load_code": 'load_dataset("pubmed_qa", "pqa_labeled")'
    },
    "bioasq": {
        "dataset_id": "bigbio/bioasq",
        "description": "BioASQ challenge dataset",
        "size": "~5K questions",
        "task": "biomedical QA",
        "load_code": 'load_dataset("bigbio/bioasq")',
        "note": "May require registration"
    },
    
    # ============ Text Classification ============
    "medical_transcriptions": {
        "dataset_id": "rungalileo/medical_transcription_40",
        "description": "Medical transcription classification",
        "size": "40 categories",
        "task": "classification",
        "load_code": 'load_dataset("rungalileo/medical_transcription_40")'
    },
    
    # ============ Named Entity Recognition ============
    "bc5cdr": {
        "dataset_id": "bigbio/bc5cdr",
        "description": "Chemical-Disease NER",
        "size": "1.5K documents",
        "task": "NER",
        "load_code": 'load_dataset("bigbio/bc5cdr")'
    },
    "ncbi_disease": {
        "dataset_id": "bigbio/ncbi_disease",
        "description": "Disease mention NER",
        "size": "793 abstracts",
        "task": "NER",
        "load_code": 'load_dataset("bigbio/ncbi_disease")'
    },
    
    # ============ Relation Extraction ============
    "chemprot": {
        "dataset_id": "bigbio/chemprot",
        "description": "Chemical-Protein relations",
        "size": "2.4K abstracts",
        "task": "relation extraction",
        "load_code": 'load_dataset("bigbio/chemprot")'
    },
    "ddi_corpus": {
        "dataset_id": "bigbio/ddi_corpus",
        "description": "Drug-Drug Interaction extraction",
        "size": "1K documents",
        "task": "relation extraction",
        "load_code": 'load_dataset("bigbio/ddi_corpus")'
    },
    
    # ============ Image Datasets ============
    "chestxray14": {
        "dataset_id": "alkzar90/NIH-Chest-X-ray-dataset",
        "description": "112K chest X-rays with 14 disease labels",
        "size": "112K images",
        "task": "multi-label classification",
        "load_code": 'load_dataset("alkzar90/NIH-Chest-X-ray-dataset")',
        "note": "Large dataset, may take time to download"
    },
    "covid_chestxray": {
        "dataset_id": "ieee8023/covid-chestxray-dataset",
        "description": "COVID-19 chest X-rays",
        "size": "~1K images",
        "task": "classification",
        "note": "GitHub dataset"
    },
    
    # ============ Clinical Notes (require access) ============
    "mimic_iii": {
        "dataset_id": "physionet/mimic-iii",
        "description": "Critical care clinical data",
        "size": "40K+ patients",
        "task": "various",
        "note": "Requires PhysioNet credentialed access",
        "access_url": "https://physionet.org/content/mimiciii/"
    },
    "mimic_iv": {
        "dataset_id": "physionet/mimic-iv",
        "description": "Updated MIMIC dataset",
        "size": "300K+ patients",
        "task": "various",
        "note": "Requires PhysioNet credentialed access",
        "access_url": "https://physionet.org/content/mimiciv/"
    }
}


class ExpandedModelLoader:
    """
    Loader for all medical models and datasets
    """
    
    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir or str(Path.home() / ".cache" / "medical_ai")
        self.loaded_models = {}
        self.loaded_datasets = {}
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if torch.cuda.is_available():
            self.gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        else:
            self.gpu_memory = 0
    
    def list_models(self) -> List[Dict]:
        """List all available models"""
        models = []
        for name, info in ALL_MEDICAL_MODELS.items():
            can_load = True
            reason = ""
            
            if info.get("api_only"):
                can_load = False
                reason = "API only (too large for local)"
            elif info.get("requires_gpu") and self.device == "cpu":
                can_load = False
                reason = "Requires GPU"
            elif info.get("min_vram_gb", 0) > self.gpu_memory:
                can_load = False
                reason = f"Needs {info.get('min_vram_gb')}GB VRAM"
            
            models.append({
                "name": name,
                "model_id": info["model_id"],
                "type": info["type"],
                "size": info["size"],
                "description": info["description"],
                "can_load_locally": can_load,
                "reason": reason
            })
        
        return models
    
    def list_datasets(self) -> List[Dict]:
        """List all available datasets"""
        datasets = []
        for name, info in ALL_MEDICAL_DATASETS.items():
            datasets.append({
                "name": name,
                "dataset_id": info["dataset_id"],
                "description": info["description"],
                "size": info["size"],
                "task": info["task"],
                "load_code": info.get("load_code", ""),
                "note": info.get("note", "")
            })
        return datasets
    
    def load_model(self, model_name: str, quantize: bool = True) -> Any:
        """Load a model by name"""
        if model_name not in ALL_MEDICAL_MODELS:
            raise ValueError(f"Unknown model: {model_name}")
        
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        
        info = ALL_MEDICAL_MODELS[model_name]
        model_id = info["model_id"]
        model_type = info["type"]
        
        logger.info(f"Loading {model_name} ({model_id})...")
        
        try:
            from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
            from transformers import AutoModelForSeq2SeqLM, AutoProcessor
            
            tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
            
            if model_type == "encoder":
                model = AutoModel.from_pretrained(model_id, token=HF_TOKEN)
            elif model_type == "causal":
                if quantize and info.get("size", "").endswith("B"):
                    try:
                        from transformers import BitsAndBytesConfig
                        config = BitsAndBytesConfig(load_in_8bit=True)
                        model = AutoModelForCausalLM.from_pretrained(
                            model_id, token=HF_TOKEN,
                            quantization_config=config,
                            device_map="auto"
                        )
                    except:
                        model = AutoModelForCausalLM.from_pretrained(
                            model_id, token=HF_TOKEN,
                            torch_dtype=torch.float16
                        )
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id, token=HF_TOKEN
                    )
            elif model_type == "seq2seq":
                model = AutoModelForSeq2SeqLM.from_pretrained(model_id, token=HF_TOKEN)
            elif model_type == "vision-text":
                processor = AutoProcessor.from_pretrained(model_id, token=HF_TOKEN)
                model = AutoModel.from_pretrained(model_id, token=HF_TOKEN)
                self.loaded_models[f"{model_name}_processor"] = processor
            else:
                model = AutoModel.from_pretrained(model_id, token=HF_TOKEN)
            
            if self.device == "cuda" and not info.get("api_only"):
                if hasattr(model, 'to'):
                    model = model.to(self.device)
            
            self.loaded_models[model_name] = model
            self.loaded_models[f"{model_name}_tokenizer"] = tokenizer
            
            logger.info(f"✅ {model_name} loaded successfully")
            return model
            
        except Exception as e:
            logger.error(f"❌ Failed to load {model_name}: {e}")
            raise
    
    def load_dataset(self, dataset_name: str) -> Any:
        """Load a dataset by name"""
        if dataset_name not in ALL_MEDICAL_DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        if dataset_name in self.loaded_datasets:
            return self.loaded_datasets[dataset_name]
        
        info = ALL_MEDICAL_DATASETS[dataset_name]
        
        logger.info(f"Loading dataset: {dataset_name}...")
        
        try:
            from datasets import load_dataset
            
            dataset_id = info["dataset_id"]
            alt_id = info.get("alt_id")
            
            try:
                if "pqa_labeled" in info.get("load_code", ""):
                    dataset = load_dataset(dataset_id, "pqa_labeled")
                elif "bigbio" in dataset_id:
                    # BigBio datasets often need specific config
                    configs = ["source", "bigbio_qa", "bigbio_text"]
                    for config in configs:
                        try:
                            dataset = load_dataset(dataset_id, config)
                            break
                        except:
                            continue
                else:
                    dataset = load_dataset(dataset_id)
            except:
                if alt_id:
                    dataset = load_dataset(alt_id)
                else:
                    raise
            
            self.loaded_datasets[dataset_name] = dataset
            logger.info(f"✅ {dataset_name} loaded successfully")
            return dataset
            
        except Exception as e:
            logger.error(f"❌ Failed to load {dataset_name}: {e}")
            raise
    
    def get_model(self, model_name: str) -> Optional[Any]:
        """Get loaded model"""
        return self.loaded_models.get(model_name)
    
    def get_tokenizer(self, model_name: str) -> Optional[Any]:
        """Get model's tokenizer"""
        return self.loaded_models.get(f"{model_name}_tokenizer")
    
    def unload_model(self, model_name: str):
        """Unload a model to free memory"""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            if f"{model_name}_tokenizer" in self.loaded_models:
                del self.loaded_models[f"{model_name}_tokenizer"]
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info(f"Unloaded {model_name}")


# Convenience functions
def load_pubmedbert():
    """Quick load PubMedBERT"""
    loader = ExpandedModelLoader()
    return loader.load_model("pubmedbert")

def load_biolinkbert():
    """Quick load BioLinkBERT"""
    loader = ExpandedModelLoader()
    return loader.load_model("biolinkbert")

def load_bioclinicalbert():
    """Quick load BioClinicalBERT"""
    loader = ExpandedModelLoader()
    return loader.load_model("bioclinicalbert")

def load_medmcqa():
    """Quick load MedMCQA dataset"""
    loader = ExpandedModelLoader()
    return loader.load_dataset("medmcqa")

def load_pubmedqa():
    """Quick load PubMedQA dataset"""
    loader = ExpandedModelLoader()
    return loader.load_dataset("pubmedqa")

def load_chestxray14():
    """Quick load ChestX-ray14 dataset"""
    loader = ExpandedModelLoader()
    return loader.load_dataset("chestxray14")


# Export
__all__ = [
    "ALL_MEDICAL_MODELS",
    "ALL_MEDICAL_DATASETS",
    "ExpandedModelLoader",
    "load_pubmedbert",
    "load_biolinkbert",
    "load_bioclinicalbert",
    "load_medmcqa",
    "load_pubmedqa",
    "load_chestxray14",
]
