"""
Comprehensive Model Loader for All Medical AI Models
Loads all Hugging Face models from DATASET_MODEL_LINKS.md
Supports Hugging Face token authentication
"""

import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    pipeline
)
from huggingface_hub import login, HfApi
import logging
from typing import Dict, Optional, Any
import json

logger = logging.getLogger(__name__)

# All models from DATASET_MODEL_LINKS.md
ALL_MODELS = {
    # Language Models (NLP)
    "clinical_bert": {
        "model_id": "medicalai/Clinical-BERT",
        "type": "masked_lm",
        "use_case": "Clinical text understanding",
        "size": "340MB",
        "class": "AutoModelForMaskedLM"
    },
    "biobert": {
        "model_id": "dmis-lab/biobert-base-cased-v1.2",
        "type": "token_classification",
        "use_case": "Medical entity extraction",
        "size": "410MB",
        "class": "AutoModelForTokenClassification"
    },
    "pubmedbert": {
        "model_id": "microsoft/PubMedBERT",
        "type": "masked_lm",
        "use_case": "Biomedical classification",
        "size": "420MB",
        "class": "AutoModelForMaskedLM"
    },
    "scibert": {
        "model_id": "allenai/scibert_scivocab_cased",
        "type": "masked_lm",
        "use_case": "Scientific text",
        "size": "410MB",
        "class": "AutoModelForMaskedLM"
    },
    "biogpt": {
        "model_id": "microsoft/biogpt",
        "type": "causal_lm",
        "use_case": "Biomedical text generation",
        "size": "350MB",
        "class": "AutoModelForCausalLM"
    },
    "mistral_7b": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "type": "causal_lm",
        "use_case": "General medical LLM",
        "size": "7B params",
        "class": "AutoModelForCausalLM",
        "quantized": True
    },
    "llama2_7b": {
        "model_id": "meta-llama/Llama-2-7b-chat-hf",
        "type": "causal_lm",
        "use_case": "General medical LLM",
        "size": "7B params",
        "class": "AutoModelForCausalLM",
        "quantized": True,
        "requires_auth": True
    },
    # Vision Models (if needed)
    "resnet50": {
        "model_id": "microsoft/resnet-50",
        "type": "image_classification",
        "use_case": "Medical image classification",
        "size": "98MB",
        "class": "AutoModel"
    },
    "mobilevit": {
        "model_id": "apple/mobilevit-small",
        "type": "image_classification",
        "use_case": "Mobile medical imaging",
        "size": "10MB",
        "class": "AutoModel"
    },
    # Additional BioBERT variants
    "biobert_v1_1": {
        "model_id": "dmis-lab/biobert-base-cased-v1.1",
        "type": "token_classification",
        "use_case": "Medical entity extraction (v1.1)",
        "size": "410MB",
        "class": "AutoModelForTokenClassification"
    },
    "biobert_v1_1_base": {
        "model_id": "dmis-lab/biobert-v1.1",
        "type": "token_classification",
        "use_case": "Medical entity extraction (base)",
        "size": "410MB",
        "class": "AutoModelForTokenClassification"
    },
    # Meditron (if accessible)
    "meditron_70b": {
        "model_id": "epfl-llm/meditron-70b",
        "type": "causal_lm",
        "use_case": "Medical LLM (70B)",
        "size": "70B params",
        "class": "AutoModelForCausalLM",
        "quantized": True,
        "requires_auth": True
    },
    # BiomedCLIP
    "biomedclip": {
        "model_id": "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        "type": "multimodal",
        "use_case": "Biomedical vision-language",
        "size": "Variable",
        "class": "AutoModel"
    }
}


class MedicalModelLoader:
    """
    Comprehensive loader for all medical AI models
    Supports Hugging Face token authentication
    """
    
    def __init__(self, hf_token: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize model loader
        
        Args:
            hf_token: Hugging Face API token (or from env HF_TOKEN)
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.hf_token = hf_token or os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_TOKEN')
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.loaded_models = {}
        self.model_configs = ALL_MODELS
        
        # Login to Hugging Face if token provided
        if self.hf_token:
            try:
                login(token=self.hf_token)
                logger.info("✅ Logged in to Hugging Face")
            except Exception as e:
                logger.warning(f"Could not login to Hugging Face: {e}")
    
    def load_model(self, model_key: str, quantized: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Load a specific model by key
        
        Args:
            model_key: Key from ALL_MODELS dict
            quantized: Use 8-bit quantization (for large models)
            **kwargs: Additional model loading parameters
            
        Returns:
            dict: Model info and loaded components
        """
        if model_key not in self.model_configs:
            raise ValueError(f"Unknown model: {model_key}. Available: {list(self.model_configs.keys())}")
        
        config = self.model_configs[model_key]
        model_id = config["model_id"]
        
        # Check if already loaded
        if model_key in self.loaded_models:
            logger.info(f"Model {model_key} already loaded, returning cached version")
            return self.loaded_models[model_key]
        
        logger.info(f"Loading {model_key} ({model_id})...")
        
        try:
            # Determine model class
            model_class = getattr(__import__('transformers'), config["class"])
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                token=self.hf_token,
                **kwargs.get('tokenizer_kwargs', {})
            )
            
            # Load model with appropriate settings
            if config["type"] == "causal_lm" and quantized and torch.cuda.is_available():
                # Use quantization for large language models
                try:
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=6.0
                    )
                    model = model_class.from_pretrained(
                        model_id,
                        token=self.hf_token,
                        quantization_config=quantization_config,
                        device_map="auto",
                        torch_dtype=torch.float16,
                        **kwargs.get('model_kwargs', {})
                    )
                except Exception as e:
                    logger.warning(f"Quantization failed, loading full precision: {e}")
                    model = model_class.from_pretrained(
                        model_id,
                        token=self.hf_token,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        **kwargs.get('model_kwargs', {})
                    )
            else:
                # Standard loading
                model = model_class.from_pretrained(
                    model_id,
                    token=self.hf_token,
                    torch_dtype=torch.float16 if torch.cuda.is_available() and config["type"] == "causal_lm" else None,
                    **kwargs.get('model_kwargs', {})
                )
            
            # Move to device if not using device_map
            if not hasattr(model, 'device_map') or model.device_map is None:
                if self.device == "cuda" and torch.cuda.is_available():
                    model = model.to(self.device)
            
            model.eval()
            
            # Create pipeline if applicable
            pipeline_obj = None
            if config["type"] == "token_classification":
                pipeline_obj = pipeline(
                    "ner",
                    model=model,
                    tokenizer=tokenizer,
                    device=0 if self.device == "cuda" else -1,
                    aggregation_strategy="simple"
                )
            elif config["type"] == "masked_lm":
                pipeline_obj = pipeline(
                    "fill-mask",
                    model=model,
                    tokenizer=tokenizer,
                    device=0 if self.device == "cuda" else -1
                )
            elif config["type"] == "causal_lm":
                # Causal LM doesn't need a pipeline, use model directly
                pass
            
            result = {
                "model_key": model_key,
                "model_id": model_id,
                "model": model,
                "tokenizer": tokenizer,
                "pipeline": pipeline_obj,
                "config": config,
                "device": self.device,
                "loaded": True
            }
            
            self.loaded_models[model_key] = result
            logger.info(f"✅ Successfully loaded {model_key}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error loading {model_key}: {str(e)}")
            return {
                "model_key": model_key,
                "model_id": model_id,
                "error": str(e),
                "loaded": False
            }
    
    def load_all_models(self, quantized: bool = True, skip_large: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Load all available models
        
        Args:
            quantized: Use quantization for large models
            skip_large: Skip models larger than 1B parameters
            
        Returns:
            dict: All loaded models
        """
        results = {}
        
        for model_key, config in self.model_configs.items():
            # Skip large models if requested
            if skip_large and any(x in config["size"].lower() for x in ["7b", "70b", "b params"]):
                logger.info(f"Skipping large model: {model_key}")
                continue
            
            # Use quantization for large models
            use_quantized = quantized and config.get("quantized", False)
            
            result = self.load_model(model_key, quantized=use_quantized)
            results[model_key] = result
        
        return results
    
    def get_model(self, model_key: str) -> Optional[Dict[str, Any]]:
        """Get a loaded model"""
        return self.loaded_models.get(model_key)
    
    def list_available_models(self) -> list:
        """List all available model keys"""
        return list(self.model_configs.keys())
    
    def get_model_info(self, model_key: str) -> Dict[str, Any]:
        """Get information about a model without loading it"""
        if model_key not in self.model_configs:
            return {"error": f"Unknown model: {model_key}"}
        
        return self.model_configs[model_key]
    
    def unload_model(self, model_key: str):
        """Unload a model to free memory"""
        if model_key in self.loaded_models:
            del self.loaded_models[model_key]
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            logger.info(f"Unloaded {model_key}")
    
    def unload_all(self):
        """Unload all models"""
        self.loaded_models.clear()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        logger.info("Unloaded all models")


# Convenience functions for quick access
def load_clinical_bert(hf_token: Optional[str] = None) -> Dict[str, Any]:
    """Quick load Clinical-BERT"""
    loader = MedicalModelLoader(hf_token=hf_token)
    return loader.load_model("clinical_bert")


def load_biobert(hf_token: Optional[str] = None) -> Dict[str, Any]:
    """Quick load BioBERT"""
    loader = MedicalModelLoader(hf_token=hf_token)
    return loader.load_model("biobert")


def load_mistral_7b(hf_token: Optional[str] = None, quantized: bool = True) -> Dict[str, Any]:
    """Quick load Mistral-7B"""
    loader = MedicalModelLoader(hf_token=hf_token)
    return loader.load_model("mistral_7b", quantized=quantized)


def load_all_medical_models(hf_token: Optional[str] = None, quantized: bool = True) -> Dict[str, Dict[str, Any]]:
    """Quick load all medical models"""
    loader = MedicalModelLoader(hf_token=hf_token)
    return loader.load_all_models(quantized=quantized, skip_large=False)


if __name__ == "__main__":
    # Test the model loader
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    # Get token from environment or command line
    token = os.getenv('HF_TOKEN') or (sys.argv[1] if len(sys.argv) > 1 else None)
    
    if not token:
        print("⚠️  No HF_TOKEN found. Some models may require authentication.")
        print("Set HF_TOKEN environment variable or pass as argument")
    
    loader = MedicalModelLoader(hf_token=token)
    
    print("\n" + "="*70)
    print("MEDICAL MODEL LOADER TEST")
    print("="*70)
    
    print(f"\nAvailable models: {len(loader.list_available_models())}")
    for key in loader.list_available_models():
        info = loader.get_model_info(key)
        print(f"  - {key}: {info['use_case']} ({info['size']})")
    
    # Test loading a small model
    print("\n" + "-"*70)
    print("Testing Clinical-BERT load...")
    result = loader.load_model("clinical_bert")
    
    if result.get("loaded"):
        print("✅ Clinical-BERT loaded successfully!")
        print(f"   Model: {result['model_id']}")
        print(f"   Device: {result['device']}")
    else:
        print(f"❌ Failed to load: {result.get('error', 'Unknown error')}")
    
    print("\n" + "="*70)

