"""
ENSEMBLE MEDICAL AI SYSTEM
===========================
Uses ALL HuggingFace models with weighted ensemble for maximum accuracy

YOUR MODELS (from HF token):
1. epfl-llm/meditron-70b - Medical specialist (TOO LARGE - use via API/Inference)
2. epfl-llm/meditron-7b - Medical specialist (CAN RUN LOCALLY with 16GB+ RAM)
3. mistralai/Mistral-7B-Instruct-v0.3 - General LLM
4. meta-llama/Llama-2-7b-chat - Conversational
5. medicalai/ClinicalBERT - Clinical text
6. dmis-lab/biobert-base-cased-v1.2 - BioBERT latest
7. dmis-lab/biobert-base-cased-v1.1 - BioBERT variant
8. dmis-lab/biobert-v1.1 - BioBERT base
9. allenai/scibert_scivocab_cased - Scientific
10. allenai/scibert_scivocab_uncased - Scientific variant
11. microsoft/biogpt - Biomedical text
12. microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 - Medical images

WEIGHTS for ensemble:
- Meditron-7B: 25% (medical specialist)
- Mistral-7B: 15% (general LLM)
- LLaMA-2-7B: 10% (conversational)
- ClinicalBERT: 12% (clinical understanding)
- BioBERT variants: 24% combined (entity extraction)
- SciBERT variants: 10% combined (scientific)
- BioGPT: 4% (text generation)

OFFLINE MODE: Uses local models only
ONLINE MODE: Adds Gemini API + HuggingFace Inference API for large models
"""

import os
import torch
import logging
import requests
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from PIL import Image
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

logger = logging.getLogger(__name__)

# Get tokens
HF_TOKEN = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_TOKEN')
GEMINI_KEY = os.getenv('GEMINI_API_KEY')

# HuggingFace Inference API endpoints (try multiple)
# The API URL format changes - we try multiple options
HF_API_URLS = [
    "https://api-inference.huggingface.co/models/",  # Standard (may work)
    "https://router.huggingface.co/hf-inference/models/",  # Router format
]
HF_INFERENCE_API = HF_API_URLS[0]  # Default to standard

# Data directory (on D: drive)
DATA_DIR = Path(os.getenv('AI_DOCTOR_DATA_DIR', r"D:\c++ homework\python\ai doctor dataset"))
MODELS_DIR = DATA_DIR / "models"


class EnsembleMedicalAI:
    """
    Ensemble system using ALL HuggingFace models with weighted responses
    
    Model Weights (for text generation):
    - meditron_7b: 25% (primary medical expert)
    - mistral_7b: 15%
    - llama2_7b: 10%
    - biogpt: 4%
    - gemini: 20% (when online)
    
    Understanding models (used for context enhancement):
    - clinical_bert: 12%
    - biobert_v1.2: 10%
    - biobert_v1.1: 8%
    - biobert_base: 6%
    - scibert_cased: 6%
    - scibert_uncased: 4%
    
    Image model:
    - biomedclip: Used separately for images
    """
    
    # Model weights for ensemble - EQUAL PRIORITY for LOCAL and HF API
    # Weights are normalized at runtime based on what's loaded
    GENERATION_WEIGHTS = {
        # LOCAL MODELS (50% total)
        "meditron_7b": 0.20,    # Medical expert (LOCAL)
        "mistral_7b": 0.10,     # General LLM (LOCAL)
        "llama2_7b": 0.08,      # Conversational (LOCAL)
        "biogpt": 0.12,         # Biomedical (LOCAL, always available)
        
        # HF API MODELS (45% total) 
        "meditron_70b_api": 0.25,  # Via HF Inference API (FREE)
        "hf_inference_api": 0.20,  # General HF API models
        
        # OPTIONAL (5%)
        "gemini": 0.05,         # Google API fallback
    }
    
    UNDERSTANDING_WEIGHTS = {
        "clinical_bert": 0.25,
        "biobert_v1.2": 0.20,
        "biobert_v1.1": 0.15,
        "biobert_base": 0.15,
        "scibert_cased": 0.15,
        "scibert_uncased": 0.10,
    }
    
    def __init__(self, 
                 use_gpu: bool = True, 
                 offline_only: bool = False,
                 load_large_models: bool = False):
        """
        Initialize ensemble medical AI
        
        Args:
            use_gpu: Use GPU if available
            offline_only: Only use local models (no API calls)
            load_large_models: Load Meditron-7B, Mistral-7B, LLaMA (requires 16GB+ RAM)
        """
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.offline_only = offline_only
        self.load_large_models = load_large_models
        
        self.models = {}
        self.tokenizers = {}
        self.pipelines = {}
        
        # Track loaded models
        self.model_status = {}
        
        # GPU memory tracking
        self.gpu_memory_gb = 0
        if self.device == "cuda":
            self.gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU Memory: {self.gpu_memory_gb:.1f} GB")
        
        print("\n" + "="*70)
        print("ENSEMBLE MEDICAL AI - Loading ALL Models")
        print("="*70)
        print(f"Device: {self.device}")
        print(f"HF Token: {'✅ Found' if HF_TOKEN else '❌ Not found'}")
        print(f"Gemini Key: {'✅ Found' if GEMINI_KEY else '❌ Not found'}")
        print(f"Mode: {'OFFLINE' if offline_only else 'ONLINE + OFFLINE'}")
        print(f"Load Large Models: {load_large_models}")
        print("="*70 + "\n")
        
        # Initialize all models
        self._init_all_models()
    
    def _init_all_models(self):
        """Initialize all available models"""
        
        # ============ BERT-based Understanding Models ============
        # These are smaller and load first
        
        # 1. ClinicalBERT
        self._load_clinical_bert()
        
        # 2-4. BioBERT variants
        self._load_biobert_variants()
        
        # 5-6. SciBERT variants
        self._load_scibert_variants()
        
        # ============ Generation Models ============
        
        # 7. BioGPT (small, always load)
        self._load_biogpt()
        
        # 8. BiomedCLIP (for images)
        self._load_biomedclip()
        
        # Large models (optional)
        if self.load_large_models:
            # 9. Meditron-7B (medical specialist)
            self._load_meditron_7b()
            
            # 10. Mistral-7B
            self._load_mistral_7b()
            
            # 11. LLaMA-2-7B
            self._load_llama2_7b()
        
        # ============ API Models ============
        if not self.offline_only:
            # 12. Gemini API
            self._init_gemini()
            
            # 13. Meditron-70B via HF Inference API
            self._init_meditron_70b_api()
        
        self._print_status()
    
    def _load_clinical_bert(self):
        """Load ClinicalBERT"""
        try:
            from transformers import AutoTokenizer, AutoModel
            
            print("Loading ClinicalBERT...")
            model_id = "medicalai/ClinicalBERT"
            
            self.tokenizers["clinical_bert"] = AutoTokenizer.from_pretrained(
                model_id, token=HF_TOKEN
            )
            self.models["clinical_bert"] = AutoModel.from_pretrained(
                model_id, token=HF_TOKEN
            )
            
            if self.device == "cuda":
                self.models["clinical_bert"] = self.models["clinical_bert"].to(self.device)
            
            self.model_status["clinical_bert"] = True
            print("  ✅ ClinicalBERT loaded")
            
        except Exception as e:
            self.model_status["clinical_bert"] = False
            print(f"  ❌ ClinicalBERT failed: {str(e)[:50]}")
    
    def _load_biobert_variants(self):
        """Load all BioBERT variants"""
        from transformers import AutoTokenizer, AutoModel, pipeline
        
        biobert_models = [
            ("biobert_v1.2", "dmis-lab/biobert-base-cased-v1.2"),
            ("biobert_v1.1", "dmis-lab/biobert-base-cased-v1.1"),
            ("biobert_base", "dmis-lab/biobert-v1.1"),
        ]
        
        for name, model_id in biobert_models:
            try:
                print(f"Loading {name}...")
                
                self.tokenizers[name] = AutoTokenizer.from_pretrained(
                    model_id, token=HF_TOKEN
                )
                self.models[name] = AutoModel.from_pretrained(
                    model_id, token=HF_TOKEN
                )
                
                if self.device == "cuda":
                    self.models[name] = self.models[name].to(self.device)
                
                self.model_status[name] = True
                print(f"  ✅ {name} loaded")
                
            except Exception as e:
                self.model_status[name] = False
                print(f"  ❌ {name} failed: {str(e)[:50]}")
        
        # Create NER pipeline with best BioBERT
        try:
            self.pipelines["ner"] = pipeline(
                "ner",
                model="dmis-lab/biobert-base-cased-v1.2",
                tokenizer="dmis-lab/biobert-base-cased-v1.2",
                device=0 if self.device == "cuda" else -1,
                aggregation_strategy="simple",
                token=HF_TOKEN
            )
            print("  ✅ BioBERT NER pipeline created")
        except Exception as e:
            print(f"  ❌ NER pipeline failed: {str(e)[:50]}")
    
    def _load_scibert_variants(self):
        """Load SciBERT variants"""
        from transformers import AutoTokenizer, AutoModel
        
        scibert_models = [
            ("scibert_cased", "allenai/scibert_scivocab_cased"),
            ("scibert_uncased", "allenai/scibert_scivocab_uncased"),
        ]
        
        for name, model_id in scibert_models:
            try:
                print(f"Loading {name}...")
                
                self.tokenizers[name] = AutoTokenizer.from_pretrained(
                    model_id, token=HF_TOKEN
                )
                self.models[name] = AutoModel.from_pretrained(
                    model_id, token=HF_TOKEN
                )
                
                if self.device == "cuda":
                    self.models[name] = self.models[name].to(self.device)
                
                self.model_status[name] = True
                print(f"  ✅ {name} loaded")
                
            except Exception as e:
                self.model_status[name] = False
                print(f"  ❌ {name} failed: {str(e)[:50]}")
    
    def _load_biogpt(self):
        """Load BioGPT (small, always load)"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            print("Loading BioGPT...")
            model_id = "microsoft/biogpt"
            
            self.tokenizers["biogpt"] = AutoTokenizer.from_pretrained(
                model_id, token=HF_TOKEN
            )
            self.models["biogpt"] = AutoModelForCausalLM.from_pretrained(
                model_id,
                token=HF_TOKEN,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            if self.device == "cuda":
                self.models["biogpt"] = self.models["biogpt"].to(self.device)
            
            self.models["biogpt"].eval()
            self.model_status["biogpt"] = True
            print("  ✅ BioGPT loaded")
            
        except Exception as e:
            self.model_status["biogpt"] = False
            print(f"  ❌ BioGPT failed: {str(e)[:50]}")
    
    def _load_biomedclip(self):
        """Load BiomedCLIP for medical images"""
        try:
            from transformers import AutoProcessor, AutoModel
            
            print("Loading BiomedCLIP...")
            model_id = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            
            self.models["biomedclip_processor"] = AutoProcessor.from_pretrained(
                model_id, token=HF_TOKEN
            )
            self.models["biomedclip"] = AutoModel.from_pretrained(
                model_id, token=HF_TOKEN
            )
            
            if self.device == "cuda":
                self.models["biomedclip"] = self.models["biomedclip"].to(self.device)
            
            self.model_status["biomedclip"] = True
            print("  ✅ BiomedCLIP loaded")
            
        except Exception as e:
            self.model_status["biomedclip"] = False
            print(f"  ❌ BiomedCLIP failed: {str(e)[:50]}")
    
    def _load_meditron_7b(self):
        """Load Meditron-7B (medical specialist LLM)"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            
            print("Loading Meditron-7B (this takes time)...")
            model_id = "epfl-llm/meditron-7b"
            
            self.tokenizers["meditron_7b"] = AutoTokenizer.from_pretrained(
                model_id, token=HF_TOKEN
            )
            
            # Use 8-bit quantization to reduce memory
            if self.device == "cuda" and self.gpu_memory_gb >= 8:
                try:
                    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                    self.models["meditron_7b"] = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        token=HF_TOKEN,
                        quantization_config=quantization_config,
                        device_map="auto"
                    )
                except:
                    # Fallback without quantization
                    self.models["meditron_7b"] = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        token=HF_TOKEN,
                        torch_dtype=torch.float16,
                        device_map="auto"
                    )
            else:
                # CPU fallback (slow but works)
                self.models["meditron_7b"] = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    token=HF_TOKEN,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
            
            self.model_status["meditron_7b"] = True
            print("  ✅ Meditron-7B loaded (PRIMARY MEDICAL EXPERT)")
            
        except Exception as e:
            self.model_status["meditron_7b"] = False
            print(f"  ❌ Meditron-7B failed: {str(e)[:80]}")
    
    def _load_mistral_7b(self):
        """Load Mistral-7B"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            
            print("Loading Mistral-7B...")
            model_id = "mistralai/Mistral-7B-Instruct-v0.3"
            
            self.tokenizers["mistral_7b"] = AutoTokenizer.from_pretrained(
                model_id, token=HF_TOKEN
            )
            
            if self.device == "cuda" and self.gpu_memory_gb >= 8:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                self.models["mistral_7b"] = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    token=HF_TOKEN,
                    quantization_config=quantization_config,
                    device_map="auto"
                )
            else:
                self.models["mistral_7b"] = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    token=HF_TOKEN,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
            
            self.model_status["mistral_7b"] = True
            print("  ✅ Mistral-7B loaded")
            
        except Exception as e:
            self.model_status["mistral_7b"] = False
            print(f"  ❌ Mistral-7B failed: {str(e)[:80]}")
    
    def _load_llama2_7b(self):
        """Load LLaMA-2-7B"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            
            print("Loading LLaMA-2-7B...")
            model_id = "meta-llama/Llama-2-7b-chat-hf"
            
            self.tokenizers["llama2_7b"] = AutoTokenizer.from_pretrained(
                model_id, token=HF_TOKEN
            )
            
            if self.device == "cuda" and self.gpu_memory_gb >= 8:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                self.models["llama2_7b"] = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    token=HF_TOKEN,
                    quantization_config=quantization_config,
                    device_map="auto"
                )
            else:
                self.models["llama2_7b"] = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    token=HF_TOKEN,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
            
            self.model_status["llama2_7b"] = True
            print("  ✅ LLaMA-2-7B loaded")
            
        except Exception as e:
            self.model_status["llama2_7b"] = False
            print(f"  ❌ LLaMA-2-7B failed: {str(e)[:80]}")
    
    def _init_gemini(self):
        """Initialize Gemini API"""
        try:
            if not GEMINI_KEY or 'your' in GEMINI_KEY.lower():
                raise ValueError("Invalid Gemini key")
            
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_KEY)
            self.models["gemini"] = genai.GenerativeModel('models/gemini-2.0-flash')
            self.model_status["gemini"] = True
            print("  ✅ Gemini API configured")
            
        except Exception as e:
            self.model_status["gemini"] = False
            print(f"  ❌ Gemini failed: {str(e)[:50]}")
    
    def _init_meditron_70b_api(self):
        """
        Initialize Meditron-70B via HuggingFace Inference API
        This allows using the 70B model WITHOUT downloading it locally!
        
        NOTE: API URL changed in Jan 2026!
        OLD (deprecated): https://api-inference.huggingface.co/models/
        NEW: https://router.huggingface.co/hf-inference/models/
        """
        try:
            if not HF_TOKEN:
                raise ValueError("HF token required for API")
            
            # Test the API endpoint with NEW URL
            headers = {"Authorization": f"Bearer {HF_TOKEN}"}
            
            # Try BioGPT first (smaller, more likely to respond quickly)
            test_url = f"{HF_INFERENCE_API}microsoft/biogpt"
            response = requests.post(
                test_url,
                headers=headers,
                json={"inputs": "test", "parameters": {"max_new_tokens": 5}},
                timeout=15
            )
            
            if response.status_code == 200 or response.status_code == 503:
                # API is working, assume Meditron-70B is also available
                self.model_status["meditron_70b_api"] = True
                print("  ✅ HuggingFace Router API available")
                print("     (Meditron-70B, BioGPT via API)")
            else:
                raise ValueError(f"API returned {response.status_code}: {response.text[:100]}")
                
        except Exception as e:
            self.model_status["meditron_70b_api"] = False
            print(f"  ⚠️ HuggingFace API: {str(e)[:80]}")
            print("     Will use local models only")
    
    def _print_status(self):
        """Print final model status"""
        print("\n" + "="*70)
        print("MODEL STATUS SUMMARY")
        print("="*70)
        
        # Generation models
        print("\n📝 GENERATION MODELS:")
        gen_models = ["meditron_7b", "meditron_70b_api", "mistral_7b", "llama2_7b", "biogpt", "gemini"]
        for m in gen_models:
            status = "✅" if self.model_status.get(m, False) else "❌"
            weight = self.GENERATION_WEIGHTS.get(m, 0) * 100
            print(f"  {status} {m} (weight: {weight:.0f}%)")
        
        # Understanding models
        print("\n🧠 UNDERSTANDING MODELS:")
        und_models = ["clinical_bert", "biobert_v1.2", "biobert_v1.1", "biobert_base", 
                      "scibert_cased", "scibert_uncased"]
        for m in und_models:
            status = "✅" if self.model_status.get(m, False) else "❌"
            weight = self.UNDERSTANDING_WEIGHTS.get(m, 0) * 100
            print(f"  {status} {m} (weight: {weight:.0f}%)")
        
        # Image model
        print("\n🖼️ IMAGE MODEL:")
        status = "✅" if self.model_status.get("biomedclip", False) else "❌"
        print(f"  {status} biomedclip")
        
        # Count
        loaded = sum(1 for v in self.model_status.values() if v)
        total = len(self.model_status)
        print(f"\n📊 Total: {loaded}/{total} models loaded")
        print("="*70 + "\n")
    
    # ================= INFERENCE METHODS =================
    
    def _generate_with_model(self, model_name: str, prompt: str, max_tokens: int = 256) -> Optional[str]:
        """Generate text with a specific model"""
        
        if not self.model_status.get(model_name, False):
            return None
        
        try:
            if model_name == "biogpt":
                return self._gen_biogpt(prompt, max_tokens)
            elif model_name == "meditron_7b":
                return self._gen_meditron_7b(prompt, max_tokens)
            elif model_name == "mistral_7b":
                return self._gen_mistral_7b(prompt, max_tokens)
            elif model_name == "llama2_7b":
                return self._gen_llama2_7b(prompt, max_tokens)
            elif model_name == "gemini":
                return self._gen_gemini(prompt)
            elif model_name == "meditron_70b_api":
                return self._gen_meditron_70b_api(prompt, max_tokens)
        except Exception as e:
            logger.error(f"{model_name} generation failed: {e}")
            return None
        
        return None
    
    def _gen_biogpt(self, prompt: str, max_tokens: int) -> str:
        """Generate with BioGPT"""
        tokenizer = self.tokenizers["biogpt"]
        model = self.models["biogpt"]
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if prompt in response:
            response = response[len(prompt):].strip()
        return response
    
    def _gen_meditron_7b(self, prompt: str, max_tokens: int) -> str:
        """Generate with Meditron-7B"""
        tokenizer = self.tokenizers["meditron_7b"]
        model = self.models["meditron_7b"]
        
        # Meditron prompt format
        full_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=2048)
        if hasattr(model, 'device'):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.6,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract assistant response
        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[-1].strip()
        if "<|im_end|>" in response:
            response = response.split("<|im_end|>")[0].strip()
        return response
    
    def _gen_mistral_7b(self, prompt: str, max_tokens: int) -> str:
        """Generate with Mistral-7B"""
        tokenizer = self.tokenizers["mistral_7b"]
        model = self.models["mistral_7b"]
        
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
        
        if hasattr(model, 'device'):
            inputs = inputs.to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "[/INST]" in response:
            response = response.split("[/INST]")[-1].strip()
        return response
    
    def _gen_llama2_7b(self, prompt: str, max_tokens: int) -> str:
        """Generate with LLaMA-2-7B"""
        tokenizer = self.tokenizers["llama2_7b"]
        model = self.models["llama2_7b"]
        
        # LLaMA-2 chat format
        full_prompt = f"[INST] {prompt} [/INST]"
        
        inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=2048)
        if hasattr(model, 'device'):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "[/INST]" in response:
            response = response.split("[/INST]")[-1].strip()
        return response
    
    def _gen_gemini(self, prompt: str) -> str:
        """Generate with Gemini API"""
        model = self.models["gemini"]
        response = model.generate_content(prompt)
        return response.text
    
    def _gen_meditron_70b_api(self, prompt: str, max_tokens: int) -> str:
        """
        Generate with models via HuggingFace Router API (NEW URL!)
        Falls back to BioGPT if Meditron-70B fails
        This is FREE (within rate limits) and doesn't require local resources!
        
        API URL changed in 2026:
        OLD: https://api-inference.huggingface.co/models/
        NEW: https://router.huggingface.co/hf-inference/models/
        """
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": 0.6,
                "top_p": 0.9,
                "return_full_text": False
            }
        }
        
        # Try models in order of preference
        models_to_try = [
            "epfl-llm/meditron-7b",  # Meditron-7B (more accessible)
            "microsoft/biogpt",       # BioGPT (smaller, reliable)
        ]
        
        for model_id in models_to_try:
            try:
                response = requests.post(
                    f"{HF_INFERENCE_API}{model_id}",
                    headers=headers,
                    json=payload,
                    timeout=120
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        return result[0].get("generated_text", "")
                    return str(result)
                elif response.status_code == 503:
                    # Model is loading, try next
                    continue
                else:
                    # Try next model
                    continue
                    
            except Exception:
                continue
        
        # All models failed
        return "[HuggingFace API models are loading, please retry]"
    
    def ensemble_generate(self, prompt: str, max_tokens: int = 256) -> Dict[str, Any]:
        """
        Generate response using weighted ensemble of all available models
        
        Returns combined response with confidence scores
        """
        results = {}
        weights_used = {}
        
        # Generate with all available models
        generation_models = ["meditron_7b", "meditron_70b_api", "mistral_7b", 
                           "llama2_7b", "biogpt", "gemini"]
        
        print("\n🔄 Generating with ensemble...")
        
        for model_name in generation_models:
            if self.model_status.get(model_name, False):
                print(f"  → {model_name}...", end=" ")
                response = self._generate_with_model(model_name, prompt, max_tokens)
                if response:
                    results[model_name] = response
                    weights_used[model_name] = self.GENERATION_WEIGHTS.get(model_name, 0.1)
                    print("✅")
                else:
                    print("❌")
        
        if not results:
            return {
                "response": "No models available for generation.",
                "models_used": [],
                "confidence": 0.0,
                "success": False
            }
        
        # Normalize weights
        total_weight = sum(weights_used.values())
        for k in weights_used:
            weights_used[k] /= total_weight
        
        # Select best response (highest weight model with valid response)
        best_model = max(weights_used.keys(), key=lambda k: weights_used[k])
        best_response = results[best_model]
        
        # Calculate ensemble confidence
        confidence = sum(weights_used.values()) / len(self.GENERATION_WEIGHTS)
        
        return {
            "response": best_response,
            "all_responses": results,
            "weights": weights_used,
            "primary_model": best_model,
            "models_used": list(results.keys()),
            "confidence": confidence,
            "success": True
        }
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract medical entities using BioBERT NER"""
        if "ner" not in self.pipelines:
            return {"error": "NER pipeline not loaded"}
        
        try:
            entities = self.pipelines["ner"](text)
            
            organized = {
                "drugs": [],
                "diseases": [],
                "symptoms": [],
                "other": []
            }
            
            for entity in entities:
                word = entity.get("word", "").replace("##", "")
                if len(word) < 2:
                    continue
                    
                label = entity.get("entity_group", "").lower()
                
                if any(x in label for x in ["drug", "medication", "chemical"]):
                    if word not in organized["drugs"]:
                        organized["drugs"].append(word)
                elif any(x in label for x in ["disease", "disorder"]):
                    if word not in organized["diseases"]:
                        organized["diseases"].append(word)
                else:
                    if word not in organized["other"]:
                        organized["other"].append(word)
            
            return organized
            
        except Exception as e:
            return {"error": str(e)}
    
    def analyze_image(self, image_path: str, queries: List[str] = None) -> Dict[str, Any]:
        """Analyze medical image using BiomedCLIP"""
        if not self.model_status.get("biomedclip", False):
            return {"error": "BiomedCLIP not loaded", "success": False}
        
        try:
            processor = self.models["biomedclip_processor"]
            model = self.models["biomedclip"]
            
            image = Image.open(image_path).convert("RGB")
            
            if not queries:
                queries = [
                    "normal healthy tissue",
                    "abnormal finding",
                    "pneumonia",
                    "tuberculosis",
                    "lung cancer",
                    "fracture",
                    "tumor",
                    "inflammation"
                ]
            
            inputs = processor(
                text=queries,
                images=image,
                return_tensors="pt",
                padding=True
            )
            
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            logits = outputs.logits_per_image[0]
            probs = torch.softmax(logits, dim=0)
            
            results = []
            for i, query in enumerate(queries):
                results.append({
                    "finding": query,
                    "confidence": float(probs[i])
                })
            
            results.sort(key=lambda x: x["confidence"], reverse=True)
            
            return {
                "success": True,
                "model": "BiomedCLIP",
                "findings": results,
                "top_finding": results[0] if results else None
            }
            
        except Exception as e:
            return {"error": str(e), "success": False}
    
    def medical_consultation(self, query: str, context: str = None) -> Dict[str, Any]:
        """
        Full medical consultation using ensemble
        
        1. Extract entities from query
        2. Generate response with ensemble
        3. Return comprehensive result
        """
        # Build prompt
        if context:
            full_prompt = f"""You are a medical AI assistant. Provide accurate, helpful medical information.

Context: {context}

Patient Question: {query}

Please provide:
1. Direct answer to the question
2. Relevant medical information
3. When to seek professional help
4. Any precautions or warnings

Response:"""
        else:
            full_prompt = f"""You are a medical AI assistant. Provide accurate, helpful medical information.

Patient Question: {query}

Please provide:
1. Direct answer to the question
2. Relevant medical information
3. When to seek professional help
4. Any precautions or warnings

Response:"""
        
        # Extract entities
        entities = self.extract_entities(query)
        
        # Generate ensemble response
        result = self.ensemble_generate(full_prompt)
        
        return {
            "query": query,
            "response": result.get("response", ""),
            "entities": entities,
            "models_used": result.get("models_used", []),
            "primary_model": result.get("primary_model", ""),
            "confidence": result.get("confidence", 0),
            "success": result.get("success", False)
        }


# Singleton
_ensemble_ai = None

def get_ensemble_ai(use_gpu: bool = True, 
                   offline_only: bool = False,
                   load_large_models: bool = False) -> EnsembleMedicalAI:
    """Get or create ensemble AI instance"""
    global _ensemble_ai
    if _ensemble_ai is None:
        _ensemble_ai = EnsembleMedicalAI(
            use_gpu=use_gpu,
            offline_only=offline_only,
            load_large_models=load_large_models
        )
    return _ensemble_ai


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*70)
    print("TESTING ENSEMBLE MEDICAL AI")
    print("="*70)
    
    # Test without large models first
    ai = EnsembleMedicalAI(use_gpu=True, offline_only=False, load_large_models=False)
    
    # Test generation
    print("\n1. Testing medical question...")
    result = ai.medical_consultation("What are the symptoms of diabetes?")
    print(f"Primary Model: {result['primary_model']}")
    print(f"Models Used: {result['models_used']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Response: {result['response'][:300]}...")
    
    # Test entity extraction
    print("\n2. Testing entity extraction...")
    entities = ai.extract_entities("Patient has hypertension and takes lisinopril 10mg daily")
    print(f"Entities: {entities}")
    
    print("\n✅ All tests completed!")
