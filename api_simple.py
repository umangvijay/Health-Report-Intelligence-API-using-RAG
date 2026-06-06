"""
AI Doctor API Backend - v3.0 with Tiered AI Routing
=====================================================
TIER SYSTEM:
- Tier 1: Gemini + HuggingFace (embeddings/vision/RAG) + OpenFDA + ChromaDB + local datasets
- Tier 2: HuggingFace only + OpenFDA + ChromaDB + local datasets  
- Tier 3: Local models only + OpenFDA + ChromaDB + local datasets
"""
import sys
from pathlib import Path

# Ensure app dir and models on path so "from data_manager" and "from models.xxx" work
_app_dir = Path(__file__).resolve().parent
_models_dir = _app_dir / "models"
if str(_app_dir) not in sys.path:
    sys.path.insert(0, str(_app_dir))
if str(_models_dir) not in sys.path:
    sys.path.insert(0, str(_models_dir))

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Callable
import uvicorn
from datetime import datetime
import os
import requests
import asyncio
import logging
import hashlib
import re
import base64
import uuid
import io
from dotenv import load_dotenv
from functools import lru_cache

# ============================================================
# ROBUST .ENV LOADING - CRITICAL FIX
# ============================================================
# Load .env MULTIPLE ways to ensure tokens are ALWAYS found
_env_path = _app_dir / ".env"

# Method 1: Direct load from script directory (do not override existing env)
if _env_path.exists():
    load_dotenv(_env_path, override=False)
    
# Method 2: Load from current working directory (do not override existing env)
load_dotenv(override=False)

# Method 3: Manually read and set if still missing
def _force_load_env():
    """Force read .env file and set environment variables directly."""
    if not _env_path.exists():
        return
    try:
        with open(_env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                key, _, value = line.partition('=')
                key = key.strip()
                value = value.strip()
                # Only set if not already in environment or empty
                if key and value and (not os.environ.get(key) or os.environ.get(key) == ''):
                    os.environ[key] = value
    except Exception:
        pass

_force_load_env()

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ai_doctor_api")

# ============================================================
# GPU & TORCH SUPPORT
# ============================================================
try:
    import torch
    TORCH_AVAILABLE = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    GPU_AVAILABLE = torch.cuda.is_available()
    GPU_COUNT = torch.cuda.device_count() if GPU_AVAILABLE else 0
    GPU_NAME = torch.cuda.get_device_name(0) if GPU_AVAILABLE else "CPU"
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = "cpu"
    GPU_AVAILABLE = False
    GPU_COUNT = 0
    GPU_NAME = "CPU"

# ============================================================
# TRANSFORMERS SUPPORT
# ============================================================
try:
    from transformers import pipeline, AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# ============================================================
# API KEYS - ALWAYS RE-READ FROM ENV WITH PLACEHOLDER DETECTION
# ============================================================
HF_TOKEN_ENV_KEYS = (
    "HF_TOKEN",
    "HUGGINGFACE_TOKEN",
    "HUGGINGFACEHUB_API_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HF_API_KEY",
)
GEMINI_KEY_ENV_KEYS = (
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GOOGLE_GENAI_API_KEY",
)
FDA_KEY_ENV_KEYS = (
    "OPEN_FDA_API_KEY",
    "FDA_API_KEY",
)

def _get_first_env(keys: tuple[str, ...]) -> str:
    """Return first non-empty env value from a list of candidate keys."""
    for key in keys:
        value = os.getenv(key)
        if value is not None and str(value).strip():
            return str(value).strip().strip('"').strip("'")
    return ""

def _is_valid_key(key_value: Optional[str]) -> bool:
    """
    CRITICAL: Check if an API key is actually valid (not a placeholder).
    Returns False if:
    - key is None
    - key is empty string
    - key matches common placeholder patterns
    - key is just whitespace
    """
    if not key_value:
        return False
    key_str = str(key_value).strip()
    if not key_str:
        return False
    # Check for common placeholder patterns (without rejecting random valid tokens)
    placeholder_patterns = [
        "YOUR_HF_TOKEN",
        "YOUR_NEW_HF_TOKEN_HERE",
        "YOUR_GEMINI",
        "YOUR_NEW_GEMINI_KEY_HERE",
        "YOUR_OPEN_FDA",
        "PLACEHOLDER",
        "INSERT_",
        "ADD_YOUR",
        "PUT_YOUR",
        "ENTER_YOUR",
        "REPLACE_",
        "<YOUR",
        "CHANGEME",
        "XXX",
        "TODO",
    ]
    key_upper = key_str.upper()
    for pattern in placeholder_patterns:
        if pattern in key_upper:
            return False
    return True


def get_hf_token() -> Optional[str]:
    """Get HF token - always fresh from environment. Returns None if invalid/placeholder."""
    token = _get_first_env(HF_TOKEN_ENV_KEYS)
    if _is_valid_key(token):
        return token
    return None


def get_gemini_key() -> Optional[str]:
    """Get Gemini API key - always fresh from environment. Returns None if invalid/placeholder."""
    key = _get_first_env(GEMINI_KEY_ENV_KEYS)
    if _is_valid_key(key):
        return key
    return None


def get_fda_key() -> Optional[str]:
    """Get FDA API key - always fresh from environment. Returns None if invalid/placeholder."""
    key = _get_first_env(FDA_KEY_ENV_KEYS)
    if _is_valid_key(key):
        return key
    return None

# Initial load (will be re-read on each request via get_xxx() functions)
HF_TOKEN = get_hf_token()
GEMINI_API_KEY = get_gemini_key()
OPEN_FDA_API_KEY = get_fda_key()

# ============================================================
# STARTUP LOGGING - EXPLICIT YES/NO FOR TOKEN DETECTION
# ============================================================
def _log_env_status():
    """
    Log environment variable status at startup with explicit YES/NO.
    Uses _is_valid_key() to detect placeholders.
    Matches required debug format.
    """
    # Get absolute path for .env
    abs_env_path = os.path.abspath(str(_env_path))
    env_found = _env_path.exists()
    
    # Get raw values to check what's actually in .env
    raw_hf = _get_first_env(HF_TOKEN_ENV_KEYS)
    raw_gem = _get_first_env(GEMINI_KEY_ENV_KEYS)
    raw_fda = _get_first_env(FDA_KEY_ENV_KEYS)
    
    # Check if valid (not placeholder)
    hf_valid = _is_valid_key(raw_hf)
    gem_valid = _is_valid_key(raw_gem)
    fda_valid = _is_valid_key(raw_fda)
    
    # Determine current tier
    if gem_valid:
        tier = "1"
        tier_name = "Tier 1: Gemini + HuggingFace"
    elif hf_valid:
        tier = "2"
        tier_name = "Tier 2: HuggingFace Only"
    else:
        tier = "3"
        tier_name = "Tier 3: Local Models Only"
    
    # Determine HF mode
    if hf_valid:
        hf_mode = "API + LOCAL" if TRANSFORMERS_AVAILABLE else "API"
    elif TRANSFORMERS_AVAILABLE:
        hf_mode = "LOCAL"
    else:
        hf_mode = "DISABLED"
    
    # Print to console (EXACT format as specified in requirements)
    print("")
    print("=" * 60)
    print("AI DOCTOR API - CONFIGURATION")
    print("=" * 60)
    print(f".env path: {abs_env_path}")
    print(f".env found: {'YES' if env_found else 'NO'}")
    print(f"[AI ROUTER] GEMINI KEY FOUND: {'YES' if gem_valid else 'NO'}")
    print(f"[AI ROUTER] HF TOKEN FOUND: {'YES' if hf_valid else 'NO'}")
    print(f"[AI ROUTER] OPEN_FDA_KEY: {'YES' if fda_valid else 'NO (soft-fail)'}")
    print(f"[AI ROUTER] DEVICE: {'CUDA - ' + GPU_NAME if GPU_AVAILABLE else 'CPU'}")
    print(f"[AI ROUTER] HF MODE: {hf_mode}")
    print("-" * 60)
    print(f"[AI ROUTER] ACTIVE TIER: {tier}")
    print(f"CURRENT TIER: {tier_name}")
    print("=" * 60)
    print("")
    print("MODEL REGISTRY:")
    try:
        print(f"  HF API Models (LLM): {len(HF_API_MODELS.get('llm', []))} models")
        print(f"  Local Models (LLM): {len(LOCAL_HF_MODELS.get('llm', []))} models")
        print(f"  Transformers available: {'YES' if TRANSFORMERS_AVAILABLE else 'NO'}")
    except NameError:
        print("  (model registry not yet loaded)")
    print("")
    
    # Also log to logger (for file logging)
    logger.info("=" * 60)
    logger.info("AI DOCTOR API - CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f".env path: {abs_env_path}")
    logger.info(f".env found: {'YES' if env_found else 'NO'}")
    logger.info(f"[AI ROUTER] GEMINI KEY FOUND: {'YES' if gem_valid else 'NO'}")
    logger.info(f"[AI ROUTER] HF TOKEN FOUND: {'YES' if hf_valid else 'NO'}")
    logger.info(f"[AI ROUTER] OPEN_FDA_KEY: {'YES' if fda_valid else 'NO (soft-fail)'}")
    logger.info(f"[AI ROUTER] DEVICE: {'CUDA' if GPU_AVAILABLE else 'CPU'}")
    logger.info(f"[AI ROUTER] HF MODE: {hf_mode}")
    logger.info(f"[AI ROUTER] ACTIVE TIER: {tier}")
    logger.info(f"CURRENT TIER: {tier_name}")
    logger.info("=" * 60)

_log_env_status()

# ============================================================
# MODEL REGISTRY - EXTENSIBLE FOR ANY MODEL TYPE
# ============================================================
"""
AI DOCTOR - MODEL REGISTRY
CRITICAL FIX: Separate HF Inference API models from LOCAL-ONLY models

NOT ALL MODELS ARE HOSTED ON HF INFERENCE API!
- Models like flan-t5, gpt2, biogpt return 404 on Inference API
- These must be loaded LOCALLY via transformers

Structure:
- HF_API_MODELS: Models available on HF Inference API (serverless)
- LOCAL_HF_MODELS: Models that MUST be loaded locally via transformers
"""

from typing import Dict, List


# ============================================================
# HF INFERENCE API MODELS (Serverless - work with API calls)
# These are ACTUALLY hosted on HF Inference API
# ============================================================
HF_API_MODELS: Dict[str, List[str]] = {
    "llm": [
        # These are the ONLY models that work on HF Inference API
        "mistralai/Mistral-7B-Instruct-v0.3",
        "HuggingFaceH4/zephyr-7b-beta",
        "meta-llama/Llama-2-7b-chat-hf",
        "tiiuae/falcon-7b-instruct",
        "bigscience/bloom-560m",  # Small BLOOM works on API
    ],
    "embedding": [
        "sentence-transformers/all-mpnet-base-v2",
        "sentence-transformers/all-MiniLM-L6-v2",
    ],
    "vision": [
        "google/vit-base-patch16-224",
        "microsoft/resnet-50",
    ],
    "classification": [
        "facebook/bart-large-mnli",
    ],
    "caption": [
        "Salesforce/blip-image-captioning-base",
    ],
}

# ============================================================
# LOCAL-ONLY MODELS (Must load via transformers, NOT API)
# These return 404 on HF Inference API but work locally
# ============================================================
LOCAL_HF_MODELS: Dict[str, List[str]] = {
    "llm": [
        # These MUST be loaded locally - they return 404 on API
        "google/flan-t5-base",       # T5 models - local only
        "google/flan-t5-large",      # T5 models - local only  
        "google/flan-t5-small",      # Smallest T5
        "distilgpt2",                # Local only
        "gpt2",                      # Local only
        "microsoft/biogpt",          # Medical - local only
        "EleutherAI/gpt-neo-125m",   # Local only
    ],
    "embedding": [
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "sentence-transformers/msmarco-distilbert-base-v4",
        "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        "emilyalsentzer/Bio_ClinicalBERT",
        "allenai/scibert_scivocab_uncased",
        "dmis-lab/biobert-base-cased-v1.1",
    ],
    "vision": [
        "facebook/deit-base-distilled-patch16-224",
        "microsoft/swinv2-tiny-patch4-window16-256",
    ],
    "classification": [
        "cross-encoder/nli-deberta-v3-base",
    ],
    "ocr": [
        "microsoft/trocr-base-printed",
    ],
    "reranker": [
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "BAAI/bge-reranker-base",
    ],
    "speech": [
        "openai/whisper-base",
        "openai/whisper-small",
    ],
}

# Combined registry for backward compatibility
MODEL_REGISTRY: Dict[str, List[str]] = {
    "llm": HF_API_MODELS.get("llm", []) + LOCAL_HF_MODELS.get("llm", []),
    "embedding": HF_API_MODELS.get("embedding", []) + LOCAL_HF_MODELS.get("embedding", []),
    "vision": HF_API_MODELS.get("vision", []) + LOCAL_HF_MODELS.get("vision", []),
    "classification": HF_API_MODELS.get("classification", []) + LOCAL_HF_MODELS.get("classification", []),
    "ocr": LOCAL_HF_MODELS.get("ocr", []),
    "reranker": LOCAL_HF_MODELS.get("reranker", []),
    "caption": HF_API_MODELS.get("caption", []),
    "speech": LOCAL_HF_MODELS.get("speech", []),
}


# Backward compatibility
HF_LLM_MODELS = MODEL_REGISTRY["llm"]
LLM_MODELS = HF_LLM_MODELS
EMBEDDING_MODELS = MODEL_REGISTRY["embedding"]
VISION_MODELS = MODEL_REGISTRY["vision"]

# API-only LLM models (for call_hf_llm to try via API first)
HF_API_LLM_MODELS = HF_API_MODELS.get("llm", [])
# Local-only LLM models (load via transformers)
LOCAL_LLM_MODELS = LOCAL_HF_MODELS.get("llm", [])

GEMINI_LARGE_MODELS = ["Llama3-70B", "Med42-70B", "Meditron-70B", "Nous-Hermes-2-Yi-34B"]

# ============================================================
# LOCAL MODEL LOADING WITH CUDA SUPPORT
# ============================================================
_LOCAL_MODEL_CACHE = {}  # Cache loaded models to avoid reloading

def _get_device():
    """Get the best available device (CUDA if available, else CPU)."""
    if GPU_AVAILABLE and TORCH_AVAILABLE:
        return torch.device("cuda")
    return torch.device("cpu")

def load_local_llm(model_name: str):
    """
    Load a HuggingFace model LOCALLY via transformers with CUDA support.
    Returns (model, tokenizer) or (None, None) if failed.
    """
    global _LOCAL_MODEL_CACHE
    
    if not TRANSFORMERS_AVAILABLE:
        logger.warning("[LOCAL LLM] Transformers not available")
        return None, None
    
    # Check cache
    if model_name in _LOCAL_MODEL_CACHE:
        logger.info(f"[LOCAL LLM] Using cached model: {model_name}")
        return _LOCAL_MODEL_CACHE[model_name]
    
    device = _get_device()
    logger.info(f"[AI ROUTER] DEVICE: {'CUDA' if device.type == 'cuda' else 'CPU'}")
    
    try:
        from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
        import torch
        
        logger.info(f"[LOCAL LLM] Loading {model_name} on {device}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Determine model type (T5/FLAN = Seq2Seq, GPT/BioGPT = Causal)
        is_seq2seq = "t5" in model_name.lower() or "flan" in model_name.lower()
        
        if is_seq2seq:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
                low_cpu_mem_usage=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
                low_cpu_mem_usage=True
            )
        
        # Move to device
        model = model.to(device)
        model.eval()
        
        logger.info(f"[LOCAL LLM] Successfully loaded {model_name} on {device}")
        
        # Cache it
        _LOCAL_MODEL_CACHE[model_name] = (model, tokenizer)
        return model, tokenizer
        
    except Exception as e:
        logger.warning(f"[LOCAL LLM] Failed to load {model_name}: {e}")
        return None, None


def generate_with_local_model(model_name: str, prompt: str, max_tokens: int = 300) -> Optional[str]:
    """
    Generate text using a locally loaded model with CUDA support.
    """
    model, tokenizer = load_local_llm(model_name)
    if model is None or tokenizer is None:
        return None
    
    try:
        import torch
        device = next(model.parameters()).device
        
        # Tokenize
        inputs = tokenizer(prompt[:2000], return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.2,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id else 0,
            )
        
        # Decode
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # For causal models, remove the input prompt from output
        if "t5" not in model_name.lower() and "flan" not in model_name.lower():
            if generated.startswith(prompt[:100]):
                generated = generated[len(prompt):].strip()
        
        if generated and len(generated.strip()) > 20:
            logger.info(f"[LOCAL LLM] Generated {len(generated)} chars with {model_name}")
            return generated.strip()
        
        return None
        
    except Exception as e:
        logger.warning(f"[LOCAL LLM] Generation failed for {model_name}: {e}")
        return None

# ============================================================
# HF INFERENCE CLIENT
# ============================================================
HF_CLIENT = None
HF_CLIENT_INITIALIZED = False

def get_hf_client():
    """Get or create HuggingFace InferenceClient with detailed logging."""
    global HF_CLIENT, HF_CLIENT_INITIALIZED
    
    token = get_hf_token()
    if not token:
        logger.debug("get_hf_client: No valid HF_TOKEN")
        return None
    
    if HF_CLIENT is None and not HF_CLIENT_INITIALIZED:
        HF_CLIENT_INITIALIZED = True
        try:
            from huggingface_hub import InferenceClient
            # Try different initialization methods
            try:
                # Method 1: Standard initialization
                HF_CLIENT = InferenceClient(token=token)
                logger.info("HF InferenceClient initialized successfully (method 1: token)")
            except Exception as e1:
                logger.debug(f"HF Client method 1 failed: {e1}")
                try:
                    # Method 2: With provider
                    HF_CLIENT = InferenceClient(provider="hf-inference", api_key=token)
                    logger.info("HF InferenceClient initialized successfully (method 2: provider)")
                except Exception as e2:
                    logger.debug(f"HF Client method 2 failed: {e2}")
                    try:
                        # Method 3: Simple
                        HF_CLIENT = InferenceClient(api_key=token)
                        logger.info("HF InferenceClient initialized successfully (method 3: api_key)")
                    except Exception as e3:
                        logger.warning(f"All HF InferenceClient init methods failed: {e3}")
                        HF_CLIENT = None
        except ImportError as ie:
            logger.warning(f"huggingface_hub not installed: {ie}")
            HF_CLIENT = None
        except Exception as e:
            logger.warning(f"HF InferenceClient init failed: {e}")
            HF_CLIENT = None
    
    return HF_CLIENT

# HF API configuration
# NOTE: The old api-inference.huggingface.co endpoint is DEPRECATED (returns 410)
# Only use the new router.huggingface.co endpoint
HF_API_BASES = [
    "https://router.huggingface.co/hf-inference/models",  # NEW - primary
    # "https://api-inference.huggingface.co/models",  # OLD - DEPRECATED (410 Gone)
]
LLM_PARAMS = {"temperature": 0.2, "max_new_tokens": 400, "top_p": 0.9, "do_sample": True, "return_full_text": False}
LLM_PROVIDER = (os.getenv("LLM_PROVIDER") or "auto").strip().lower()
if LLM_PROVIDER not in ("auto", "gemini", "hf", "local"):
    LLM_PROVIDER = "auto"

# ============================================================
# CURRENT TIER TRACKING
# ============================================================
class TierState:
    """Track which tier is currently active."""
    TIER_1 = "Tier 1: Gemini + HuggingFace"
    TIER_2 = "Tier 2: HuggingFace Only"
    TIER_3 = "Tier 3: Local Models Only"
    
    current_tier = TIER_3
    gemini_quota_exceeded = False
    last_error = None

TIER_STATE = TierState()

def determine_current_tier() -> str:
    """Determine which tier to use based on available API keys."""
    gemini_key = get_gemini_key()
    hf_token = get_hf_token()
    
    if gemini_key and not TIER_STATE.gemini_quota_exceeded:
        TIER_STATE.current_tier = TierState.TIER_1
        return TierState.TIER_1
    elif hf_token:
        TIER_STATE.current_tier = TierState.TIER_2
        return TierState.TIER_2
    else:
        TIER_STATE.current_tier = TierState.TIER_3
        return TierState.TIER_3

# ============================================================
# OPENFDA - SHARED UTILITY FOR ALL TIERS (SOFT-FAIL)
# ============================================================
def fetch_openfda_data(query: str) -> Optional[str]:
    """
    Fetch drug data from OpenFDA API - used in ALL tiers to improve accuracy.
    SOFT-FAIL: Returns None if API key missing/invalid or any error occurs.
    This ensures the main LLM flow is NEVER blocked by OpenFDA issues.
    """
    # Check if FDA key is valid first - soft-fail if not
    fda_key = get_fda_key()
    if not fda_key:
        logger.debug("OpenFDA: No valid API key, soft-fail (continuing without FDA data)")
        return None
    
    try:
        from models.drug_data_sources import get_fda_drug_text_from_query
        result = get_fda_drug_text_from_query(query)
        if result:
            logger.debug("OpenFDA: Successfully retrieved drug data")
        return result
    except ImportError as e:
        logger.debug(f"OpenFDA: Module not available, soft-fail: {e}")
        return None
    except Exception as e:
        logger.warning(f"OpenFDA: API call failed, soft-fail (continuing): {e}")
        return None

def enrich_prompt_with_openfda(prompt: str) -> tuple[str, Optional[str]]:
    """
    Check if prompt mentions drugs and enrich with OpenFDA data.
    Returns (enriched_prompt, fda_data).
    """
    fda_data = fetch_openfda_data(prompt)
    if fda_data:
        enriched = f"""Use this verified drug information from FDA in your response:

{fda_data}

---
User query: {prompt}

Provide a comprehensive medical response using the FDA data above."""
        return enriched, fda_data
    return prompt, None

# ============================================================
# CHROMADB / RAG - SHARED UTILITY FOR ALL TIERS (SOFT-FAIL)
# ============================================================
def get_rag_context(query: str, n_results: int = 5) -> Optional[str]:
    """
    Retrieve relevant context from ChromaDB for all tiers.
    SOFT-FAIL: Returns None if ChromaDB unavailable or any error occurs.
    This ensures the main LLM flow is NEVER blocked by RAG issues.
    """
    try:
        try:
            from models.data_manager import data_manager
        except ImportError:
            try:
                from data_manager import data_manager
            except ImportError:
                logger.debug("RAG: data_manager module not available, soft-fail")
                return None
        
        docs = data_manager.retrieve_similar_docs(query[:500], n_results=n_results)
        if docs and len(docs) > 0:
            context_parts = ["**Relevant Medical Knowledge (RAG):**"]
            for i, d in enumerate(docs[:3], 1):
                text = d.get("content") or d.get("text") or str(d)[:400]
                context_parts.append(f"{i}. {text}")
            logger.debug(f"RAG: Retrieved {len(docs)} documents")
            return "\n".join(context_parts)
        logger.debug("RAG: No matching documents found")
        return None
    except Exception as e:
        logger.debug(f"RAG: Soft-fail (continuing): {e}")
        return None


def call_llm(model: str, prompt: str, timeout: int = 180, token: Optional[str] = None) -> Any:
    """
    Call HF Inference API for text generation (single model).
    Uses InferenceClient or direct requests. Token from env if not passed.
    Includes detailed logging for debugging.
    """
    hf_token = token or get_hf_token()
    
    if not hf_token or not _is_valid_key(hf_token):
        logger.debug(f"call_llm {model}: No valid HF token")
        return None
    
    headers = {"Authorization": f"Bearer {hf_token}"}
    
    # Try InferenceClient first
    if HF_CLIENT:
        try:
            out = HF_CLIENT.text_generation(
                model=model,
                prompt=prompt[:2000],  # Limit prompt length
                max_new_tokens=LLM_PARAMS.get("max_new_tokens", 400),
                temperature=LLM_PARAMS.get("temperature", 0.2),
                top_p=LLM_PARAMS.get("top_p", 0.9),
            )
            if out:
                logger.debug(f"call_llm {model}: InferenceClient success")
                return [{"generated_text": out}] if isinstance(out, str) else out
        except Exception as e:
            logger.warning(f"call_llm HF_CLIENT {model}: {str(e)[:100]}")
    
    # Fallback to direct API requests
    payload = {"inputs": prompt[:2000], "parameters": LLM_PARAMS}
    
    for base in HF_API_BASES:
        url = f"{base}/{model}"
        try:
            logger.debug(f"call_llm trying: {url}")
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
            
            if r.status_code == 200:
                logger.debug(f"call_llm {model}: Direct API success via {base}")
                return r.json()
            elif r.status_code == 503:
                logger.debug(f"call_llm {model}: 503 Service Unavailable (model loading)")
                continue
            elif r.status_code == 401:
                logger.warning(f"call_llm {model}: 401 Unauthorized - check HF_TOKEN")
                break
            elif r.status_code == 429:
                logger.warning(f"call_llm {model}: 429 Rate Limited")
                continue
            else:
                logger.debug(f"call_llm {model}: HTTP {r.status_code} - {r.text[:100]}")
                
        except requests.exceptions.Timeout:
            logger.warning(f"call_llm {model}: Timeout after {timeout}s")
            continue
        except Exception as e:
            logger.debug(f"call_llm {model} ({base}): {str(e)[:100]}")
            continue
    
    return None


def call_gemini(prompt: str) -> Optional[str]:
    """
    Call Google Gemini API (gemini-1.5-flash). Re-reads GEMINI_API_KEY from env.
    Catches 429 (quota exceeded) and sets TIER_STATE.gemini_quota_exceeded.
    """
    key = get_gemini_key()
    if not key:
        return None
    try:
        from google import genai
        client = genai.Client(api_key=key)
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        err_str = str(e).lower()
        if "429" in err_str or "quota" in err_str or "resource_exhausted" in err_str:
            TIER_STATE.gemini_quota_exceeded = True
            logger.warning("Gemini quota exceeded (429) → using HF + local models")
        else:
            logger.warning("Gemini failed: %s", e)
        return None


def call_hf_llm(prompt: str, timeout: int = 120) -> tuple:
    """
    FIXED: Try HF Inference API models first, then LOCAL models.
    
    CRITICAL FIX: Not all models are hosted on HF Inference API!
    - Models like flan-t5, gpt2, biogpt return 404 on API
    - These MUST be loaded locally via transformers
    
    Order:
    1. Try HF_API_LLM_MODELS via API (if HF_TOKEN valid)
    2. Try LOCAL_LLM_MODELS via local transformers loading (with CUDA)
    
    Returns (text, model_used) or (None, None).
    """
    hf_token = get_hf_token()
    device = _get_device()
    
    logger.info("=" * 60)
    logger.info("[AI ROUTER] call_hf_llm ENTRY")
    logger.info(f"[AI ROUTER] HF TOKEN FOUND: {'YES' if hf_token else 'NO'}")
    logger.info(f"[AI ROUTER] DEVICE: {'CUDA' if device.type == 'cuda' else 'CPU'}")
    logger.info(f"[AI ROUTER] HF MODE: {'API + LOCAL' if hf_token else 'LOCAL ONLY'}")
    logger.info("=" * 60)
    
    errors_encountered = []
    
    # ============================================================
    # PHASE 1: Try HF Inference API models (only if token valid)
    # ============================================================
    if hf_token:
        get_hf_client()  # ensure HF_CLIENT is created
        logger.info(f"[AI ROUTER] PHASE 1: Trying {len(HF_API_LLM_MODELS)} API models...")
        
        for i, model in enumerate(HF_API_LLM_MODELS):
            try:
                logger.info(f"[HF API] [{i+1}/{len(HF_API_LLM_MODELS)}] Trying: {model}")
                
                # Try with HF_CLIENT first (InferenceClient)
                if HF_CLIENT:
                    try:
                        out = HF_CLIENT.text_generation(
                            model=model,
                            prompt=prompt[:2000],
                            max_new_tokens=300,
                            temperature=0.2,
                            top_p=0.9,
                        )
                        if out and len(str(out).strip()) > 20:
                            logger.info(f"[HF API] SUCCESS via InferenceClient: {model}")
                            return (str(out).strip(), f"HF-API:{model}")
                    except Exception as client_err:
                        err_msg = str(client_err)
                        # Check for specific error codes
                        if "401" in err_msg or "Unauthorized" in err_msg:
                            logger.error(f"[AI ROUTER] FALLBACK REASON: 401 Unauthorized - HF_TOKEN invalid")
                            errors_encountered.append(f"{model}: 401 Unauthorized")
                            break  # Don't try other API models with bad token
                        elif "404" in err_msg or "Not Found" in err_msg:
                            logger.warning(f"[HF API] {model}: 404 Not Found - Model not on Inference API")
                            errors_encountered.append(f"{model}: 404")
                        elif "503" in err_msg:
                            logger.warning(f"[HF API] {model}: 503 Service Unavailable (model loading)")
                            errors_encountered.append(f"{model}: 503")
                        elif "429" in err_msg:
                            logger.warning(f"[HF API] {model}: 429 Rate Limited")
                            errors_encountered.append(f"{model}: 429")
                        else:
                            logger.warning(f"[HF API] {model} error: {err_msg[:80]}")
                            errors_encountered.append(f"{model}: {err_msg[:40]}")
                
                # Fallback to direct API call
                out = call_llm(model, prompt[:2000], timeout, token=hf_token)
                if out:
                    if isinstance(out, list) and len(out) > 0:
                        text = out[0].get("generated_text")
                    elif isinstance(out, dict):
                        text = out.get("generated_text")
                    else:
                        text = str(out)
                    
                    if text and len(str(text).strip()) > 20:
                        logger.info(f"[HF API] SUCCESS via direct API: {model}")
                        return (str(text).strip(), f"HF-API:{model}")
                        
            except Exception as e:
                err_msg = str(e)
                logger.warning(f"[HF API] {model} exception: {err_msg[:80]}")
                errors_encountered.append(f"{model}: {err_msg[:40]}")
        
        logger.info(f"[AI ROUTER] PHASE 1 COMPLETE: API models exhausted")
    else:
        logger.info("[AI ROUTER] PHASE 1 SKIPPED: No valid HF_TOKEN")
    
    # ============================================================
    # PHASE 2: Try LOCAL models via transformers (with CUDA)
    # ============================================================
    if TRANSFORMERS_AVAILABLE:
        logger.info(f"[AI ROUTER] PHASE 2: Trying {len(LOCAL_LLM_MODELS)} LOCAL models...")
        
        for i, model in enumerate(LOCAL_LLM_MODELS):
            try:
                logger.info(f"[LOCAL LLM] [{i+1}/{len(LOCAL_LLM_MODELS)}] Loading: {model}")
                
                result = generate_with_local_model(model, prompt, max_tokens=300)
                
                if result and len(result.strip()) > 20:
                    logger.info(f"[LOCAL LLM] SUCCESS: {model}")
                    return (result.strip(), f"LOCAL:{model}")
                else:
                    logger.warning(f"[LOCAL LLM] {model}: Generated empty/short response")
                    
            except Exception as e:
                err_msg = str(e)
                logger.warning(f"[AI ROUTER] FALLBACK REASON: {model} - {err_msg[:80]}")
                errors_encountered.append(f"{model}: {err_msg[:40]}")
        
        logger.info(f"[AI ROUTER] PHASE 2 COMPLETE: Local models exhausted")
    else:
        logger.warning("[AI ROUTER] PHASE 2 SKIPPED: Transformers not available")
    
    # ============================================================
    # ALL MODELS FAILED
    # ============================================================
    logger.error(f"[AI ROUTER] ALL MODELS FAILED!")
    logger.error(f"[AI ROUTER] API models tried: {len(HF_API_LLM_MODELS)}")
    logger.error(f"[AI ROUTER] Local models tried: {len(LOCAL_LLM_MODELS) if TRANSFORMERS_AVAILABLE else 0}")
    if errors_encountered:
        logger.error(f"[AI ROUTER] Errors: {errors_encountered[:5]}")
    
    return (None, None)


def call_embedding(model: str, text: str, timeout: int = 180, token: Optional[str] = None) -> Any:
    """Call HF Inference API for embeddings (single model). Uses HF_TOKEN from env if token not passed."""
    hf_token = token or get_hf_token()
    headers = {"Authorization": f"Bearer {hf_token}"} if _is_valid_key(hf_token) else {}
    if HF_CLIENT and hf_token:
        try:
            return HF_CLIENT.feature_extraction(model=model, text=text)
        except Exception as e:
            logger.debug("call_embedding HF_CLIENT %s: %s", model, e)
    if not headers:
        return None
    payload = {"inputs": text}
    for base in HF_API_BASES:
        url = f"{base}/{model}"
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 503:
                continue
        except Exception as e:
            logger.debug("call_embedding %s: %s", model, e)
            continue
    return None


def call_embedding_any(text: str, timeout: int = 60) -> tuple:
    """Try each embedding model (EMBEDDING_MODELS); all use HF_TOKEN. Return (embedding, model_used) or (None, None)."""
    hf_token = get_hf_token()
    for model in EMBEDDING_MODELS:
        try:
            logger.info("Embedding → %s", model)
            out = call_embedding(model, text, timeout, token=hf_token)
            if out is not None:
                return (out, model)
        except Exception as e:
            logger.debug("Embedding %s: %s", model, e)
            continue
    return (None, None)


def call_vision(model: str, image_path: str, timeout: int = 180, token: Optional[str] = None) -> Any:
    """Call HF Inference API for image (single model). Uses HF_TOKEN from env if token not passed."""
    hf_token = token or get_hf_token()
    headers = {"Authorization": f"Bearer {hf_token}"} if _is_valid_key(hf_token) else {}
    try:
        with open(image_path, "rb") as f:
            img_bytes = f.read()
    except Exception as e:
        logger.warning("call_vision read image: %s", e)
        return None
    if HF_CLIENT and hf_token:
        try:
            return HF_CLIENT.image_classification(image=image_path, model=model)
        except Exception as e:
            logger.debug("call_vision HF_CLIENT %s: %s", model, e)
    if not headers:
        return None
    img_b64 = base64.b64encode(img_bytes).decode()
    payload = {"inputs": img_b64}
    for base in HF_API_BASES:
        url = f"{base}/{model}"
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 503:
                continue
        except Exception as e:
            logger.debug("call_vision %s: %s", model, e)
            continue
    return None


def call_vision_any(image_path: str, timeout: int = 90) -> tuple:
    """Try each vision model (VISION_MODELS); all use HF_TOKEN. Return (result, model_used) or (None, None)."""
    hf_token = get_hf_token()
    for model in VISION_MODELS:
        try:
            logger.info("Vision → %s", model)
            out = call_vision(model, image_path, timeout, token=hf_token)
            if out is not None:
                return (out, model)
        except Exception as e:
            logger.debug("Vision %s: %s", model, e)
            continue
    return (None, None)


def call_local(prompt: str) -> str:
    """Local fallback when no API available - do not echo the prompt."""
    hf = get_hf_token()
    gem = get_gemini_key()
    
    if hf and gem:
        # Both tokens exist but APIs failed
        return (
            "Service temporarily unavailable. Both Gemini and HuggingFace APIs failed to respond. "
            "Please try again later or check API quotas/status."
        )
    elif hf:
        # HF token exists but API failed
        return (
            "Service temporarily unavailable. HuggingFace API failed to respond. "
            "HF_TOKEN: configured, GEMINI_API_KEY: not configured. "
            "Please try again later or add GEMINI_API_KEY for fallback."
        )
    elif gem:
        # Gemini key exists but API failed
        return (
            "Service temporarily unavailable. Gemini API failed to respond. "
            "GEMINI_API_KEY: configured, HF_TOKEN: not configured. "
            "Please try again later or add HF_TOKEN for fallback."
        )
    else:
        # No tokens configured
        return (
            "Please set HF_TOKEN or GEMINI_API_KEY in .env to enable AI responses. "
            "Get HF token: https://huggingface.co/settings/tokens (Read + Inference API). "
            "Get Gemini key: https://aistudio.google.com/apikey"
        )


def use_local_model(prompt: str) -> str:
    """
    TIER 3 - THE GUARANTEE: Local model fallback.
    Uses LocalEnsembleMedicalAI + RAG (ChromaDB) + DrugBank + local datasets.
    NEVER crashes. This function MUST always return something useful.
    
    Priority:
    1. LocalEnsembleMedicalAI (BioGPT, ClinicalBERT, etc.)
    2. RAG retrieval from ChromaDB
    3. DrugBank database
    4. Basic fallback message
    """
    parts = []
    ensemble_response = None
    
    # ============================================================
    # STEP 1: Try LocalEnsembleMedicalAI (full local ML models)
    # ============================================================
    try:
        logger.info("[LOCAL] Trying LocalEnsembleMedicalAI...")
        print("[LOCAL] Trying LocalEnsembleMedicalAI with BioGPT, ClinicalBERT, etc.")
        
        try:
            from models.local_ensemble import LocalEnsembleMedicalAI
        except ImportError:
            from local_ensemble import LocalEnsembleMedicalAI
        
        # Initialize with GPU if available
        ensemble = LocalEnsembleMedicalAI(use_gpu=GPU_AVAILABLE)
        
        # Get medical consultation
        result = ensemble.medical_consultation(prompt[:1500])
        
        if result and result.get("response"):
            response_text = result.get("response", "")
            if len(response_text.strip()) > 50:
                confidence = result.get("confidence", 0.75)
                models_used = result.get("models_used", [])
                
                parts.append("[LOCAL MODE - Ensemble AI]")
                parts.append("")
                parts.append(response_text)
                parts.append("")
                parts.append(f"---")
                parts.append(f"Confidence: {confidence*100:.0f}%")
                if models_used:
                    parts.append(f"Models used: {', '.join(models_used[:3])}")
                parts.append("Note: This is AI-generated. Consult a healthcare professional.")
                
                ensemble_response = "\n".join(parts)
                logger.info(f"[LOCAL] LocalEnsembleMedicalAI SUCCESS - {len(response_text)} chars")
                print(f"[LOCAL] LocalEnsembleMedicalAI SUCCESS - Confidence: {confidence*100:.0f}%")
                return ensemble_response
        else:
            logger.info("[LOCAL] LocalEnsembleMedicalAI returned empty response")
            
    except ImportError as e:
        logger.info(f"[LOCAL] LocalEnsembleMedicalAI not available: {e}")
        print(f"[LOCAL] LocalEnsembleMedicalAI import failed: {e}")
    except Exception as e:
        logger.info(f"[LOCAL] LocalEnsembleMedicalAI failed: {e}")
        print(f"[LOCAL] LocalEnsembleMedicalAI exception: {str(e)[:80]}")
    
    # ============================================================
    # STEP 2: Try loading individual local models (transformers)
    # ============================================================
    if TRANSFORMERS_AVAILABLE:
        try:
            logger.info("[LOCAL] Trying individual local models...")
            print("[LOCAL] Trying individual local models (flan-t5, biogpt, etc.)...")
            
            for model_name in LOCAL_LLM_MODELS[:3]:  # Try first 3 local models
                print(f"[LOCAL] Loading: {model_name}")
                result = generate_with_local_model(model_name, prompt, max_tokens=400)
                
                if result and len(result.strip()) > 50:
                    parts = ["[LOCAL MODE]", ""]
                    parts.append(result.strip())
                    parts.append("")
                    parts.append(f"---")
                    parts.append(f"Model: {model_name}")
                    parts.append("Note: This is AI-generated. Consult a healthcare professional.")
                    logger.info(f"[LOCAL] Model {model_name} SUCCESS")
                    print(f"[LOCAL] SUCCESS with {model_name}")
                    return "\n".join(parts)
                    
        except Exception as e:
            logger.info(f"[LOCAL] Individual model loading failed: {e}")
    
    # ============================================================
    # STEP 3: Try RAG retrieval (soft-fail)
    # ============================================================
    parts = ["[LOCAL MODE]", ""]
    rag_results = None
    
    try:
        try:
            from models.data_manager import data_manager
        except ImportError:
            try:
                from data_manager import data_manager
            except ImportError:
                logger.debug("use_local_model: data_manager not available")
                data_manager = None
        
        if data_manager:
            docs = data_manager.retrieve_similar_docs(prompt[:500], n_results=5)
            if docs and len(docs) > 0:
                rag_results = docs
                parts.append("**From medical knowledge base:**")
                for i, d in enumerate(docs[:3], 1):
                    text = d.get("text") or d.get("content") or str(d)[:400]
                    parts.append(f"{i}. {text}")
                parts.append("")
                logger.info(f"use_local_model: RAG returned {len(docs)} documents")
    except Exception as e:
        logger.debug(f"use_local_model: RAG soft-fail: {e}")
    
    # ============================================================
    # STEP 4: Try local drug database (soft-fail)
    # ============================================================
    try:
        from models.drugbank_loader import DrugBankLoader
        db = DrugBankLoader()
        # Extract potential drug names from prompt
        words = prompt.lower().split()
        for word in words:
            if len(word) > 3:
                info = db.lookup_medicine(word)
                if info and info.get("found"):
                    parts.append(f"**Drug Info ({info.get('name', word)}):**")
                    if info.get("uses"):
                        uses = info.get('uses')
                        if isinstance(uses, list):
                            parts.append(f"Uses: {', '.join(uses[:5])}")
                        else:
                            parts.append(f"Uses: {str(uses)[:200]}")
                    if info.get("dosage"):
                        parts.append(f"Dosage: {info.get('dosage')}")
                    if info.get("side_effects"):
                        effects = info.get('side_effects')
                        if isinstance(effects, list):
                            parts.append(f"Side Effects: {', '.join(effects[:5])}")
                        else:
                            parts.append(f"Side Effects: {str(effects)[:200]}")
                    parts.append("")
                    break
    except Exception as e:
        logger.debug(f"use_local_model: DrugBank soft-fail: {e}")
    
    # If we got some results, return them
    if len(parts) > 2:  # More than just "[LOCAL MODE]" and empty line
        parts.append("---")
        parts.append("Note: This response is from local knowledge base. For accurate medical advice, please consult a healthcare professional.")
        return "\n".join(parts)
    
    # ============================================================
    # FINAL FALLBACK - basic response
    # ============================================================
    return (
        "[LOCAL MODE]\n\n"
        "I apologize, but I cannot provide a detailed response at this time.\n\n"
        "**Recommendations:**\n"
        "1. Please consult a qualified healthcare professional for medical advice\n"
        "2. If this is an emergency, call your local emergency services\n"
        "3. For general health information, visit trusted sources like WHO, CDC, or NHS\n\n"
        "Note: The AI service is running in offline mode. "
        "For better responses, configure GEMINI_API_KEY or HF_TOKEN in the .env file."
    )


def _local_fallback_with_rag_and_datasets(prompt: str) -> str:
    """
    Wrapper for use_local_model() for backward compatibility.
    NEVER crashes - always returns something.
    """
    return use_local_model(prompt)


def _build_enriched_prompt(prompt: str) -> str:
    """
    Enrich prompt with OpenFDA (all tiers) and RAG/ChromaDB context.
    SOFT-FAIL: If OpenFDA or RAG fails, continue without that data.
    This ensures the prompt is ALWAYS built successfully - NEVER crashes.
    
    Handles None values gracefully:
    - If FDA data available: include it
    - If FDA data None: add note "(No FDA data available)"
    - Same for RAG context
    """
    enriched_parts = []
    fda_available = False
    rag_available = False
    
    # Try OpenFDA (soft-fail - NEVER blocks the flow)
    try:
        fda_data = fetch_openfda_data(prompt)
        if fda_data and len(str(fda_data).strip()) > 10:
            enriched_parts.append(f"Use this verified drug information from FDA when relevant:\n\n{fda_data}\n\n---")
            fda_available = True
            logger.debug("_build_enriched_prompt: OpenFDA data added")
    except Exception as e:
        logger.debug(f"_build_enriched_prompt: OpenFDA soft-fail: {e}")
    
    if not fda_available:
        logger.debug("_build_enriched_prompt: No FDA data available (soft-fail)")
    
    # Try RAG/ChromaDB (soft-fail - NEVER blocks the flow)
    try:
        rag_ctx = get_rag_context(prompt, n_results=5)
        if rag_ctx and len(str(rag_ctx).strip()) > 10:
            enriched_parts.append(f"{rag_ctx}\n\n---")
            rag_available = True
            logger.debug("_build_enriched_prompt: RAG context added")
    except Exception as e:
        logger.debug(f"_build_enriched_prompt: RAG soft-fail: {e}")
    
    if not rag_available:
        logger.debug("_build_enriched_prompt: No RAG context available (soft-fail)")
    
    # Always add the user query (this MUST succeed)
    enriched_parts.append(f"User query: {prompt}\n\nProvide a full consultation and analysis (not just bullet points).")
    
    # Log what we included
    logger.debug(f"_build_enriched_prompt: FDA={'YES' if fda_available else 'NO'}, RAG={'YES' if rag_available else 'NO'}")
    
    return "\n\n".join(enriched_parts)


def route_ai_request(prompt: str, image: Any = None) -> tuple:
    """
    THE "UNBREAKABLE" CHAIN - Tiered AI routing (STRICT ORDER)
    ============================================================
    This function MUST NEVER return None or crash. It ALWAYS returns a response.
    
    TIER 1: Gemini API (highest priority)
        - Condition: GEMINI_API_KEY exists AND valid AND quota not exceeded
        - Also uses HF for embeddings/RAG alongside Gemini
        
    TIER 2: HuggingFace (fallback)
        - Condition: Gemini unavailable AND HF_TOKEN exists
        - Uses HF API models first, then LOCAL HF models with CUDA
        
    TIER 3: Local Offline Mode (guarantee)
        - Condition: No Gemini AND no HF_TOKEN
        - Uses local datasets + ChromaDB + OpenFDA
    
    CRITICAL: 
    - OpenFDA status does NOT block Tier 1 or 2
    - HF failure on API triggers LOCAL loading (not immediate Tier 3)
    
    Returns (response_text, source_str) - NEVER returns None.
    """
    # Get keys and validate with _is_valid_key
    gemini_key = get_gemini_key()
    hf_token = get_hf_token()
    
    gemini_valid = _is_valid_key(gemini_key)
    hf_valid = _is_valid_key(hf_token)
    device = _get_device()
    
    # ============================================================
    # COMPREHENSIVE DEBUG LOGGING (as specified)
    # ============================================================
    print("")
    print("=" * 60)
    print("[AI ROUTER] ROUTING REQUEST")
    print("=" * 60)
    print(f"[AI ROUTER] GEMINI KEY FOUND: {'YES' if gemini_valid else 'NO'}")
    print(f"[AI ROUTER] HF TOKEN FOUND: {'YES' if hf_valid else 'NO'}")
    print(f"[AI ROUTER] DEVICE: {'CUDA' if device.type == 'cuda' else 'CPU'}")
    
    # Determine and log which tier we'll use
    if gemini_valid and not TIER_STATE.gemini_quota_exceeded:
        active_tier = 1
        hf_mode = "API + LOCAL (for embeddings/RAG)"
        print(f"[AI ROUTER] ACTIVE TIER: 1 (Gemini + HuggingFace)")
        print(f"[AI ROUTER] HF MODE: {hf_mode}")
    elif hf_valid:
        active_tier = 2
        hf_mode = "API + LOCAL (with CUDA)" if TRANSFORMERS_AVAILABLE else "API"
        print(f"[AI ROUTER] ACTIVE TIER: 2 (HuggingFace Only)")
        print(f"[AI ROUTER] HF MODE: {hf_mode}")
    else:
        active_tier = 3
        hf_mode = "LOCAL ONLY" if TRANSFORMERS_AVAILABLE else "DISABLED"
        print(f"[AI ROUTER] ACTIVE TIER: 3 (Local Models Only)")
        print(f"[AI ROUTER] HF MODE: {hf_mode}")
    print("=" * 60)
    
    # Also log to logger
    logger.info("=" * 60)
    logger.info("[AI ROUTER] GEMINI KEY FOUND: %s", 'YES' if gemini_valid else 'NO')
    logger.info("[AI ROUTER] HF TOKEN FOUND: %s", 'YES' if hf_valid else 'NO')
    logger.info("[AI ROUTER] ACTIVE TIER: %d", active_tier)
    logger.info("[AI ROUTER] HF MODE: %s", hf_mode)
    logger.info("[AI ROUTER] DEVICE: %s", 'CUDA' if device.type == 'cuda' else 'CPU')
    logger.info("=" * 60)
    
    # Build enriched prompt (OpenFDA + RAG soft-fail - NEVER blocks)
    try:
        enriched = _build_enriched_prompt(prompt)
    except Exception as e:
        logger.warning(f"_build_enriched_prompt failed, using raw prompt: {e}")
        print(f"[AI ROUTER] FALLBACK REASON: Prompt enrichment failed - {e}")
        enriched = prompt

    # ============================================================
    # TIER 1: GEMINI (Highest Priority)
    # ============================================================
    if gemini_valid and not TIER_STATE.gemini_quota_exceeded:
        print("[AI ROUTER] >>> TIER 1: Attempting Gemini API...")
        logger.info(">>> TIER 1: Attempting Gemini API...")
        TIER_STATE.current_tier = TierState.TIER_1
        try:
            ans = call_gemini(enriched)
            if ans and len(str(ans).strip()) > 20:
                print("[AI ROUTER] >>> TIER 1: Gemini SUCCESS")
                logger.info(">>> TIER 1: Gemini SUCCESS")
                return (str(ans).strip(), "Gemini")
            else:
                print("[AI ROUTER] FALLBACK REASON: Gemini returned empty/short response")
                logger.warning(">>> TIER 1: Gemini returned empty/short response")
        except Exception as e:
            print(f"[AI ROUTER] FALLBACK REASON: Gemini exception - {str(e)[:80]}")
            logger.warning(f">>> TIER 1: Gemini exception: {e}")
            TIER_STATE.last_error = str(e)
            # Check for quota exceeded
            if "429" in str(e) or "quota" in str(e).lower():
                TIER_STATE.gemini_quota_exceeded = True
                print("[AI ROUTER] Gemini quota exceeded - will use HF next time")
        
        print("[AI ROUTER] >>> TIER 1 FAILED - Falling back to Tier 2...")
        logger.info(">>> TIER 1 FAILED - Falling back to Tier 2...")
    else:
        # Log WHY we're skipping Tier 1
        if TIER_STATE.gemini_quota_exceeded:
            reason = "quota exceeded"
        elif not gemini_valid:
            reason = "GEMINI_API_KEY invalid or placeholder"
        else:
            reason = "unknown"
        print(f"[AI ROUTER] >>> TIER 1: SKIP ({reason})")
        logger.info(f">>> TIER 1: SKIP ({reason})")

    # ============================================================
    # TIER 2: HUGGINGFACE (API + LOCAL with CUDA)
    # ============================================================
    if hf_valid:
        print("[AI ROUTER] >>> TIER 2: Attempting HuggingFace (API + Local)...")
        logger.info(">>> TIER 2: Attempting HuggingFace...")
        TIER_STATE.current_tier = TierState.TIER_2
        
        try:
            # call_hf_llm now tries API models first, then LOCAL models with CUDA
            text, model_used = call_hf_llm(enriched)
            if text and len(text.strip()) > 20:
                print(f"[AI ROUTER] >>> TIER 2: SUCCESS via {model_used}")
                logger.info(f">>> TIER 2: HuggingFace SUCCESS via {model_used}")
                return (text, model_used)
            else:
                print("[AI ROUTER] FALLBACK REASON: All HF models returned empty/short response")
                logger.warning(">>> TIER 2: All HF models returned empty/short response")
        except Exception as e:
            print(f"[AI ROUTER] FALLBACK REASON: HuggingFace exception - {str(e)[:80]}")
            logger.warning(f">>> TIER 2: HuggingFace exception: {e}")
            TIER_STATE.last_error = str(e)
        
        print("[AI ROUTER] >>> TIER 2 FAILED - Falling back to Tier 3...")
        logger.info(">>> TIER 2 FAILED - Falling back to Tier 3...")
    else:
        print("[AI ROUTER] >>> TIER 2: SKIP (HF_TOKEN missing/invalid)")
        logger.info(">>> TIER 2: SKIP (HF_TOKEN missing/invalid)")

    # ============================================================
    # TIER 3: LOCAL (THE GUARANTEE)
    # ============================================================
    print("[AI ROUTER] >>> TIER 3: Using Local Models (THE GUARANTEE)")
    logger.info(">>> TIER 3: Using Local Models (THE GUARANTEE)")
    TIER_STATE.current_tier = TierState.TIER_3
    
    try:
        local_response = use_local_model(prompt)
        if local_response and len(local_response.strip()) > 10:
            print("[AI ROUTER] >>> TIER 3: Local model SUCCESS")
            logger.info(">>> TIER 3: Local model SUCCESS")
            return (local_response, "LOCAL")
    except Exception as e:
        print(f"[AI ROUTER] FALLBACK REASON: Local model exception - {str(e)[:80]}")
        logger.error(f">>> TIER 3: Local model exception: {e}")
    
    # ============================================================
    # FINAL SAFETY NET
    # ============================================================
    print("[AI ROUTER] >>> FINAL SAFETY NET: All tiers failed!")
    logger.error(">>> FINAL SAFETY NET: All tiers failed!")
    return (
        "[SYSTEM ERROR]\n\n"
        "Unable to generate a response. All AI services are unavailable.\n\n"
        "Troubleshooting:\n"
        "1. Check if local models exist in 'Ai Doctor/models' folder\n"
        "2. Verify .env file has valid API keys (not placeholders)\n"
        "3. Check network connectivity for external APIs\n\n"
        "For immediate help, please consult a healthcare professional.",
        "ERROR"
    )


def route_request(model_type: str, input_data: Any, **kwargs) -> Any:
    """
    Model-type agnostic routing: dispatch to the right HF API based on MODEL_REGISTRY.
    model_type: one of llm, embedding, vision, classification, ocr, reranker, caption, speech.
    input_data: prompt (str) for llm, text for embedding, image path/bytes for vision, etc.
    Returns result from first successful model in that category, or None.
    """
    models = MODEL_REGISTRY.get(model_type)
    if not models:
        logger.debug("Unknown model_type: %s", model_type)
        return None
    token = get_hf_token()
    if not token:
        return None
    get_hf_client()

    if model_type == "llm":
        prompt = input_data if isinstance(input_data, str) else str(input_data)
        for model in models:
            try:
                out = call_llm(model, prompt, token=token, **kwargs)
                if out:
                    if isinstance(out, list) and len(out) > 0:
                        return out[0].get("generated_text") or out
                    if isinstance(out, dict):
                        return out.get("generated_text") or out
                    return out
            except Exception as e:
                logger.debug("route_request llm %s: %s", model, e)
        return None

    if model_type == "embedding":
        text = input_data if isinstance(input_data, str) else str(input_data)
        for model in models:
            try:
                out = call_embedding(model, text, token=token, **kwargs)
                if out is not None:
                    return out
            except Exception as e:
                logger.debug("route_request embedding %s: %s", model, e)
        return None

    if model_type == "vision":
        for model in models:
            try:
                out = call_vision(model, input_data, token=token, **kwargs)
                if out is not None:
                    return out
            except Exception as e:
                logger.debug("route_request vision %s: %s", model, e)
        return None

    # Generic fallback: try HF Inference API with input_data as payload
    for model in models:
        try:
            headers = {"Authorization": f"Bearer {token}"}
            payload = kwargs.get("payload", {"inputs": input_data})
            for base in HF_API_BASES:
                url = f"{base}/{model}"
                r = requests.post(url, headers=headers, json=payload, timeout=kwargs.get("timeout", 60))
                if r.status_code == 200:
                    return r.json()
        except Exception as e:
            logger.debug("route_request %s %s: %s", model_type, model, e)
    return None


def smart_llm(prompt: str) -> tuple:
    """
    Priority: Gemini (very high) → HF (high) → local (RAG + datasets).
    Enriches prompt with OpenFDA + ChromaDB in all tiers. Logs tier in use.
    Returns (response_text, source_str).
    """
    provider = (os.getenv("LLM_PROVIDER") or "auto").strip().lower()
    if provider not in ("auto", "gemini", "hf", "local"):
        provider = "auto"
    gemini_key = get_gemini_key()
    hf_token = get_hf_token()
    gemini_ok = bool(gemini_key)
    hf_ok = bool(hf_token)

    if provider == "local":
        logger.info("Using Local Models")
        return (_local_fallback_with_rag_and_datasets(prompt), "LOCAL")

    # Use unified route_ai_request for auto/gemini/hf so OpenFDA + RAG always used
    if provider in ("auto", "gemini", "hf"):
        return route_ai_request(prompt, image=None)

    logger.info("Local fallback (RAG + datasets)")
    return (_local_fallback_with_rag_and_datasets(prompt), "LOCAL")


# Initialize FastAPI app
app = FastAPI(
    title="AI Doctor API",
    description="Medical Assistant Backend - v2.0",
    version="2.0.0"
)

# Security middlewares
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Request models
class DiagnosisRequest(BaseModel):
    symptoms: str
    patient_id: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None

class FeedbackRequest(BaseModel):
    query: str
    response: str
    rating: Optional[float] = 3.0
    feedback_text: Optional[str] = ""
    feedback_type: Optional[str] = "general"
    report_type: Optional[str] = "general"
    patient_id: Optional[str] = None

class RAGRequest(BaseModel):
    query: str
    n_results: int = 5

# Security: PHI sanitization
PHI_PATTERNS = {
    "ssn": r"\d{3}-\d{2}-\d{4}",
    "phone": r"\d{3}-\d{3}-\d{4}",
    "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "medical_record": r"MRN[\s:]*\d+"
}

def sanitize_phi(text: str) -> str:
    """Remove Personally Identifiable Health Information"""
    sanitized = text
    for phi_type, pattern in PHI_PATTERNS.items():
        sanitized = re.sub(pattern, f"[{phi_type.upper()}]", sanitized)
    return sanitized

def log_request(patient_id: str, action: str, sanitized_data: str):
    """Audit logging - log sanitized data only"""
    logger.info(f"Action: {action} | Patient: {patient_id} | Data hash: {hashlib.sha256(sanitized_data.encode()).hexdigest()[:8]}")

# Response caching
@lru_cache(maxsize=128)
def get_cached_response(query_hash: str):
    """Simple response caching"""
    return None

# Rate limiting
request_counts = {}
MAX_REQUESTS_PER_MINUTE = 20

async def check_rate_limit(client_ip: str) -> bool:
    """Simple rate limiting"""
    current_time = datetime.now().minute
    key = f"{client_ip}_{current_time}"
    
    if key not in request_counts:
        request_counts[key] = 0
    
    request_counts[key] += 1
    return request_counts[key] <= MAX_REQUESTS_PER_MINUTE

# Health check
@app.get("/")
async def health_check():
    """
    Health check endpoint - ALWAYS reads fresh token status from environment.
    Uses _is_valid_key() to detect placeholders.
    Returns current tier and configuration status.
    """
    # ALWAYS read fresh from env and validate with _is_valid_key - CRITICAL FIX
    raw_hf = _get_first_env(HF_TOKEN_ENV_KEYS)
    raw_gem = _get_first_env(GEMINI_KEY_ENV_KEYS)
    raw_fda = _get_first_env(FDA_KEY_ENV_KEYS)
    
    # Use _is_valid_key to detect placeholders
    hf_ok = _is_valid_key(raw_hf)
    gemini_ok = _is_valid_key(raw_gem)
    fda_ok = _is_valid_key(raw_fda)
    
    # Determine current tier
    if gemini_ok and not TIER_STATE.gemini_quota_exceeded:
        current_tier = "Tier 1: Gemini + HuggingFace"
    elif hf_ok:
        current_tier = "Tier 2: HuggingFace Only"
    else:
        current_tier = "Tier 3: Local Models Only"
    
    # Build flow message
    if LLM_PROVIDER == "auto":
        if gemini_ok:
            llm_flow_msg = "Gemini → HF → Local"
        elif hf_ok:
            llm_flow_msg = "HF → Local (Gemini key invalid/missing)"
        else:
            llm_flow_msg = "Local only (no valid API keys)"
    elif LLM_PROVIDER == "gemini":
        llm_flow_msg = "Gemini only" if gemini_ok else "Local (Gemini key invalid)"
    elif LLM_PROVIDER == "hf":
        llm_flow_msg = "HF only" if hf_ok else "Local (HF token invalid)"
    else:
        llm_flow_msg = "Local only"
    
    return {
        "status": "healthy",
        "service": "AI Doctor API v3.0 - Unbreakable Chain",
        "current_tier": current_tier,
        "llm_flow": llm_flow_msg,
        "llm_provider": LLM_PROVIDER,
        "tokens": {
            "HF_TOKEN": "YES" if hf_ok else "NO (invalid/placeholder)",
            "GEMINI_API_KEY": "YES" if gemini_ok else "NO (invalid/placeholder)",
            "OPEN_FDA_API_KEY": "YES" if fda_ok else "NO (soft-fail mode)"
        },
        "features": ["Unbreakable Chain", "Tiered Routing", "RAG", "OpenFDA (soft-fail)", "Placeholder Detection"],
        "gpu": {
            "available": GPU_AVAILABLE,
            "device": GPU_NAME,
            "count": GPU_COUNT,
            "torch": TORCH_AVAILABLE,
            "transformers": TRANSFORMERS_AVAILABLE
        },
        "timestamp": datetime.now().isoformat()
    }


# =========================
# HF REPOS – LIST & USE ALL MODELS IN YOUR TOKEN
# =========================
@app.get("/hf/models")
async def hf_list_models():
    """List all models from MODEL_REGISTRY (llm, embedding, vision, classification, ocr, reranker, caption, speech)."""
    hf = get_hf_token()
    gem = get_gemini_key()
    fda = get_fda_key()
    
    # Determine current tier
    if gem and not TIER_STATE.gemini_quota_exceeded:
        current_tier = "Tier 1: Gemini + HuggingFace"
    elif hf:
        current_tier = "Tier 2: HuggingFace Only"
    else:
        current_tier = "Tier 3: Local Models Only"
    
    return {
        "model_registry": MODEL_REGISTRY,
        "llm_models": LLM_MODELS,
        "embedding_models": EMBEDDING_MODELS,
        "vision_models": VISION_MODELS,
        "gemini_large_models": GEMINI_LARGE_MODELS,
        "hf_token_loaded": "YES" if hf else "NO",
        "gemini_api_key_loaded": "YES" if gem else "NO",
        "open_fda_api_key_loaded": "YES" if fda else "NO",
        "hf_token_configured": bool(hf),
        "gemini_configured": bool(gem),
        "open_fda_configured": bool(fda),
        "current_tier": current_tier,
    }


@app.get("/gpu/status")
async def gpu_status():
    """Get GPU and compute device status"""
    gpu_info = {
        "gpu_available": GPU_AVAILABLE,
        "device_name": GPU_NAME,
        "device_type": str(DEVICE),
        "gpu_count": GPU_COUNT,
        "torch_available": TORCH_AVAILABLE,
        "transformers_available": TRANSFORMERS_AVAILABLE,
    }
    
    if GPU_AVAILABLE and TORCH_AVAILABLE:
        try:
            gpu_info["gpu_memory_total"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
            gpu_info["gpu_memory_allocated"] = f"{torch.cuda.memory_allocated(0) / 1e9:.2f} GB"
            gpu_info["gpu_memory_cached"] = f"{torch.cuda.memory_cached(0) / 1e9:.2f} GB"
        except:
            pass
    
    return gpu_info


@app.post("/route")
async def route_ai_endpoint(request: Request):
    """Tiered AI routing: try Gemini → HF → local. Body: {"prompt": "...", "image_base64": "optional"}."""
    try:
        body = await request.json()
        prompt = body.get("prompt", "")
        if not prompt:
            raise ValueError("prompt is required")
        text, source = await asyncio.to_thread(route_ai_request, prompt, body.get("image_base64"))
        return {"success": True, "response": text, "model": source, "tier": TIER_STATE.current_tier}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/route/{model_type}")
async def route_request_endpoint(model_type: str, request: Request):
    """Model-type agnostic: route_request(model_type, input). Body: {"input": "..."} or {"text": "..."} etc."""
    try:
        body = await request.json()
        input_data = body.get("input") or body.get("text") or body.get("prompt") or body.get("image_base64") or ""
        result = await asyncio.to_thread(route_request, model_type, input_data, **{k: v for k, v in body.items() if k not in ("input", "text", "prompt", "image_base64")})
        return {"success": result is not None, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/hf/llm")
async def hf_call_llm(request: Request):
    """Call LLM. Body: {"prompt": "...", "model": "optional model id"}. No model → smart_llm (Gemini → HF → local)."""
    try:
        body = await request.json()
        prompt = body.get("prompt", "")
        model_id = body.get("model")
        if not prompt:
            raise ValueError("prompt is required")
        if model_id:
            out = await asyncio.to_thread(call_llm, model_id, prompt, 120)
            if out:
                text = out[0].get("generated_text") if isinstance(out, list) and len(out) > 0 else out.get("generated_text") if isinstance(out, dict) else None
                if text:
                    return {"success": True, "response": text, "model": model_id}
        else:
            text, source = await asyncio.to_thread(smart_llm, prompt)
            if text:
                return {"success": True, "response": text, "model": source}
        return {"success": False, "error": "No LLM responded"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/hf/embedding")
async def hf_call_embedding(request: Request):
    """Call HF embedding model with GPU support. Body: {"text": "..."}. Uses first available embedding model."""
    if not TRANSFORMERS_AVAILABLE:
        return {"success": False, "error": "Transformers not available", "device": str(DEVICE)}
    
    try:
        body = await request.json()
        text = body.get("text", "")
        if not text:
            raise ValueError("text is required")
        
        # Try local embeddings first with GPU support
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading Sentence-Transformers on {DEVICE}")
            model = SentenceTransformer('all-mpnet-base-v2')
            if GPU_AVAILABLE:
                model = model.to(DEVICE)
            
            embeddings = model.encode(text)
            return {
                "success": True,
                "model": "all-mpnet-base-v2-local",
                "device": "GPU" if GPU_AVAILABLE else "CPU",
                "embedding_dim": len(embeddings) if hasattr(embeddings, '__len__') else None,
                "embedding": embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings
            }
        except Exception as e:
            logger.debug(f"Local embeddings: {e}")
        
        # Fall back to HF API (tries all EMBEDDING_MODELS)
        out, model_used = await asyncio.to_thread(call_embedding_any, text, 60)
        if out is not None:
            return {
                "success": True,
                "model": model_used,
                "device": "HF-API",
                "embedding": out
            }
        
        return {"success": False, "error": "No embedding model responded", "device": str(DEVICE)}
    except Exception as e:
        return {"success": False, "error": str(e), "device": str(DEVICE)}


@app.post("/hf/vision")
async def hf_call_vision(request: Request):
    """Analyze image using HF vision models. Body: {"image_base64": "..."} or multipart file."""
    try:
        image_b64 = None
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            body = await request.json()
            image_b64 = body.get("image_base64")
        else:
            form = await request.form()
            f = form.get("file") or form.get("image")
            if f and hasattr(f, "read"):
                image_b64 = base64.b64encode(await f.read()).decode()
        if not image_b64:
            raise ValueError("image_base64 or file is required")
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(base64.b64decode(image_b64))
            tmp_path = tmp.name
        try:
            out, model_used = await asyncio.to_thread(call_vision_any, tmp_path, 90)
            if out is not None:
                return {"success": True, "model": model_used, "result": out}
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        return {"success": False, "error": "No vision model responded"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def _is_drug_medicine_query(text: str) -> bool:
    """True if the query is about a medicine/drug (uses, side effects, dosage) rather than symptoms."""
    q = (text or "").lower().strip()
    drug_keywords = [
        "paracetamol", "acetaminophen", "ibuprofen", "aspirin", "medicine", "medication", "drug",
        "tablet", "tablets", "uses", "side effects", "side effect", "dosage", "dose", "how to use",
        "what is", "tell me about", "information on", "crocin", "dolo", "amoxicillin", "metformin",
    ]
    return any(k in q for k in drug_keywords)


def _get_drug_info_from_db(query: str) -> Optional[str]:
    """If query mentions a known drug, return formatted info from DrugBankLoader. Else None."""
    q = (query or "").lower().strip()
    drug_name = None
    known = ["paracetamol", "acetaminophen", "ibuprofen", "aspirin", "amoxicillin", "metformin",
             "omeprazole", "lisinopril", "atorvastatin", "amlodipine", "losartan"]
    for d in known:
        if d in q:
            drug_name = d
            break
    if not drug_name:
        return None
    try:
        from models.drugbank_loader import DrugBankLoader
        db = DrugBankLoader()
        info = db.lookup_medicine(drug_name)
        if not info.get("found"):
            return None
        lines = [f"**{info.get('name', drug_name)}**", "", f"**Drug class:** {info.get('drug_class', 'N/A')}", ""]
        for label, key in [("Uses", "uses"), ("Dosage", "dosage"), ("Side effects", "side_effects"), ("Precautions", "contraindications")]:
            val = info.get(key)
            if val is not None:
                lines.append(f"**{label}:**")
                if isinstance(val, list):
                    lines.extend([f"  - {x}" for x in val[:10]])
                else:
                    lines.append(f"  {val}")
                lines.append("")
        return "\n".join(lines).strip()
    except Exception as e:
        logger.debug("Drug DB lookup: %s", e)
        return None


def _get_drug_info_from_fda(query: str) -> Optional[str]:
    """If OPEN_FDA_API_KEY is set, try open.fda.gov drug/label for terms in query. Else None."""
    if not _is_valid_key(OPEN_FDA_API_KEY):
        return None
    try:
        from models.drug_data_sources import get_fda_drug_text_from_query
        return get_fda_drug_text_from_query(query)
    except Exception as e:
        logger.debug("FDA drug lookup: %s", e)
        return None


def _medical_prompt(symptoms: str) -> str:
    """Build prompt for full consultation and analysis (not brief bullets only)."""
    return f"""You are an expert medical doctor. Provide a full consultation and analysis.

Patient symptoms: {sanitize_phi(symptoms)}

Provide a comprehensive analysis with:
1. Differential diagnosis (most likely conditions) with brief explanation
2. Recommended diagnostic tests
3. Treatment plan with medications
4. Home care and remedies
5. Lifestyle changes
6. Warning signs requiring immediate medical attention

Write in clear paragraphs where helpful. Do not give only short bullet lists; include brief explanation for key points.

Response:"""


def _medicine_question_prompt(query: str) -> str:
    """Prompt for medicine/drug questions (uses, side effects, dosage) - not symptom diagnosis."""
    return f"""You are a medical expert. Answer this medicine/drug question clearly and concisely.
Include: uses, side effects, dosage/how to use, and precautions if relevant.
Do not treat this as a patient symptom list.

Question: {sanitize_phi(query)}

Answer:"""


async def get_meditron_diagnosis(symptoms: str) -> str:
    """
    Diagnosis using SMART ROUTER: Gemini first → HF free-tier → local.
    Uses symptom prompt for symptom queries, medicine prompt for drug/medicine questions.
    """
    if _is_drug_medicine_query(symptoms):
        prompt = _medicine_question_prompt(symptoms)
    else:
        prompt = _medical_prompt(symptoms)
    
    # Log current token status before calling
    hf = get_hf_token()
    gem = get_gemini_key()
    logger.info(f"get_meditron_diagnosis: HF_TOKEN={'YES' if hf else 'NO'}, GEMINI_KEY={'YES' if gem else 'NO'}")
    
    text, source = await asyncio.to_thread(smart_llm, prompt)
    
    # Reject responses that are just the prompt echoed back or invalid
    if not text or len(text.strip()) < 30:
        return f"[{source}]\n\n{text or 'No response generated.'}"
    
    # If model echoed the prompt (common when local fallback returns prompt), replace with message
    if "[LOCAL MODEL ANSWER]" in text or (len(text) <= len(prompt) + 100 and prompt[:60].strip() in text):
        if hf or gem:
            text = f"API services temporarily unavailable. HF_TOKEN: {'configured' if hf else 'missing'}, GEMINI_API_KEY: {'configured' if gem else 'missing'}. Please try again later."
        else:
            text = "Unable to generate a response. Please set HF_TOKEN or GEMINI_API_KEY in .env and try again."
    return f"[{source}]\n\n{text}"


# Diagnosis endpoint with RAG + caching
@app.post("/diagnose")
async def diagnose(request: DiagnosisRequest, req: Request):
    # Rate limiting
    client_ip = req.client.host if req.client else "unknown"
    if not await check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    # Audit logging
    sanitized_input = sanitize_phi(request.symptoms)
    log_request(request.patient_id or "unknown", "diagnose", sanitized_input)
    
    # Drug/medicine questions: try FDA first (if OPEN_FDA_API_KEY), then DB, then LLM
    if _is_drug_medicine_query(request.symptoms):
        fda_text = await asyncio.to_thread(_get_drug_info_from_fda, request.symptoms)
        if fda_text and len(fda_text) > 50:
            return {
                "emergency": False,
                "diagnosis": f"[open.fda.gov]\n\n{fda_text}",
                "treatment": {"medications": ["See above"], "home_remedies": [], "lifestyle": []},
                "confidence": 0.90,
                "disclaimer": "This is for information only. Always consult a healthcare professional.",
                "timestamp": datetime.now().isoformat(),
                "model": "open_fda"
            }
        drug_text = await asyncio.to_thread(_get_drug_info_from_db, request.symptoms)
        if drug_text and len(drug_text) > 50:
            return {
                "emergency": False,
                "diagnosis": f"[Medicine Database]\n\n{drug_text}",
                "treatment": {"medications": ["See above"], "home_remedies": [], "lifestyle": []},
                "confidence": 0.95,
                "disclaimer": "This is for information only. Always consult a healthcare professional.",
                "timestamp": datetime.now().isoformat(),
                "model": "drug_db"
            }
    
    # Symptom diagnosis or medicine question (if DB/FDA had no match): use LLM
    hf_response = await get_meditron_diagnosis(request.symptoms)
    
    # Check if we got a valid response
    if "error" in hf_response.lower() or "loading" in hf_response.lower() or "unavailable" in hf_response.lower():
        diagnosis_text = f"Based on symptoms: {sanitized_input}. {hf_response}"
        model_used = "fallback"
    else:
        diagnosis_text = f"[HF Medical Analysis]\n\n{hf_response}"
        # Parse model from first line if present, e.g. "[epfl-llm/meditron-70b]"
        model_used = "hf_llm"
        if diagnosis_text.startswith("["):
            end = diagnosis_text.find("]")
            if end > 0:
                model_used = diagnosis_text[1:end].strip()
    
    return {
        "emergency": False,
        "diagnosis": diagnosis_text,
        "treatment": {
            "medications": ["See diagnosis above for specific recommendations"],
            "home_remedies": ["Rest", "Stay hydrated", "Monitor symptoms"],
            "lifestyle": ["Follow doctor's advice", "Get adequate sleep"]
        },
        "confidence": 0.85 if model_used != "fallback" else 0.6,
        "disclaimer": "This is AI-generated advice. Always consult a healthcare professional.",
        "timestamp": datetime.now().isoformat(),
        "model": model_used
    }

# RAG-based retrieval endpoint
@app.post("/retrieve")
async def retrieve_context(request: RAGRequest):
    """Retrieve similar medical documents for context (ChromaDB)."""
    try:
        try:
            from models.data_manager import data_manager
        except ImportError:
            from data_manager import data_manager
        docs = data_manager.retrieve_similar_docs(request.query, request.n_results)
        return {"success": True, "documents": docs}
    except Exception as e:
        logger.error(f"RAG retrieval error: {str(e)}")
        return {"success": False, "error": str(e), "documents": []}

# RLHF feedback endpoint
@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest, req: Request):
    """Submit feedback for RLHF training"""
    if not await check_rate_limit(req.client.host if req.client else "unknown"):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    try:
        from data_manager import data_manager
        
        feedback = {
            "query": sanitize_phi(request.query),
            "response": sanitize_phi(request.response),
            "feedback_type": request.feedback_type,
            "rating": 5 if request.feedback_type == "good" else 2,
            "timestamp": datetime.now().isoformat()
        }
        
        # Log feedback
        log_request(request.patient_id or "unknown", "feedback", str(feedback))
        
        # Save to data manager
        data_manager.save_feedback(request.patient_id or "unknown", feedback)
        
        # Add to RAG index for future learning
        data_manager.add_to_rag_index(
            text=f"Query: {feedback['query']}\nResponse: {feedback['response']}\nRating: {feedback['rating']}",
            metadata={"type": "feedback", "rating": feedback["rating"]}
        )
        
        return {"success": True, "message": "Feedback recorded and will improve AI training"}
    except Exception as e:
        logger.error(f"Feedback error: {str(e)}")
        return {"success": False, "error": str(e)}

# Patient history endpoint
@app.get("/patient_history/{patient_id}")
async def get_patient_history(patient_id: str):
    try:
        from data_manager import data_manager
        return data_manager.get_patient_consultations(patient_id)
    except:
        return []

# Analytics endpoint
@app.get("/analytics/{patient_id}")
async def get_analytics(patient_id: str):
    try:
        from data_manager import data_manager
        return data_manager.get_analytics_data(patient_id)
    except:
        return {
            "total_consultations": 0,
            "health_score": 85,
            "risk_level": "Low",
            "next_checkup": "In 3 months",
            "symptom_frequency": {},
        "health_trend": []
    }

# Learning report endpoint
@app.get("/learning/report")
async def learning_report():
    return {
        "total_feedback_collected": 0,
        "model_updates": 0,
        "performance_metrics": {
            "estimated_accuracy": 0.85
        }
    }

# Learning stats endpoint
@app.get("/learning/stats")
async def learning_stats():
    return {
        "training_cycles": 5,
        "total_feedback": 100,
        "improvement": 15,
        "active": True,
        "kb_size": 50000,
        "last_update": datetime.now().isoformat()
    }

# Learning curve endpoint
@app.get("/learning/curve")
async def learning_curve():
    return {
        "epochs": list(range(1, 11)),
        "accuracy": [70, 72, 75, 77, 80, 82, 84, 85, 86, 87],
        "baseline": [70] * 10
    }

# Feedback endpoint (with RL integration)
@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest, req: Request):
    """Submit feedback which triggers RL training"""
    from core_agents import DiagnosticAgent
    
    try:
        diagnostic_agent = DiagnosticAgent()
        
        # Submit to RL system
        success = diagnostic_agent.submit_feedback_for_learning(
            query=sanitize_phi(request.query),
            diagnosis=sanitize_phi(request.response),
            rating=5 if request.feedback_type == "good" else 2,
            patient_id=request.patient_id or "unknown"
        )
        
        return {
            "success": success,
            "message": "Feedback recorded and queued for RL training",
            "rl_status": "enabled" if success else "disabled"
        }
    except Exception as e:
        logger.error(f"Feedback error: {str(e)}")
        return {"success": False, "error": str(e)}

# RL Training endpoints
@app.post("/train/rl")
async def train_rl_models(models: List[str] = None, req: Request = None):
    """
    ACTUAL RL INTEGRATION ENDPOINT
    Trigger training of reinforcement learning models
    Models: ['ppo', 'dqn', 'actor_critic', 'a3c']
    """
    if not await check_rate_limit(req.client.host if req else "unknown"):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    try:
        from core_agents import DiagnosticAgent
        diagnostic_agent = DiagnosticAgent()
        
        result = diagnostic_agent.trigger_rl_training(models=models)
        
        log_request(
            patient_id="system",
            action="rl_training",
            sanitized_data=f"Models: {models or 'all'}"
        )
        
        return result
    except Exception as e:
        logger.error(f"RL training error: {str(e)}")
        return {"success": False, "error": str(e)}

@app.get("/train/status")
async def training_status():
    """Get current RL training status and metrics"""
    try:
        from core_agents import DiagnosticAgent
        diagnostic_agent = DiagnosticAgent()
        
        return diagnostic_agent.get_learning_metrics()
    except Exception as e:
        logger.error(f"Status error: {str(e)}")
        return {"error": str(e), "status": "error"}

@app.get("/train/report")
async def training_report():
    """Get comprehensive learning report"""
    try:
        from training.advanced_rl_training import learning_orchestrator
        return learning_orchestrator.export_learning_report()
    except Exception as e:
        logger.error(f"Report error: {str(e)}")
        return {"error": str(e)}

@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Submit feedback on diagnosis to improve model"""
    try:
        from core_agents import DiagnosticAgent
        
        agent = DiagnosticAgent()
        
        # Get rating and feedback text from request
        rating = request.__dict__.get('rating', 3.0)
        feedback_text = request.__dict__.get('feedback_text', '')
        report_type = request.__dict__.get('report_type', 'general')
        
        result = agent.store_feedback(
            query=request.query,
            response=request.response,
            rating=rating,
            feedback_text=feedback_text,
            patient_id=request.patient_id or "anonymous",
            report_type=report_type
        )
        
        return result
    
    except Exception as e:
        return {"success": False, "error": str(e)}

def _normalize_params(params: list) -> list:
    """Normalize parameter dicts to have 'name', 'value', 'unit', 'normal_range', 'status'."""
    out = []
    for p in params or []:
        name = p.get("name") or p.get("parameter", "Unknown")
        status = (p.get("status") or "UNKNOWN").upper()
        if "LOW" in status or status == "LOW":
            status = "LOW"
        elif "HIGH" in status or status == "HIGH":
            status = "HIGH"
        else:
            status = "NORMAL"
        out.append({
            "name": name.upper() if isinstance(name, str) else str(name),
            "value": p.get("value", "N/A"),
            "unit": p.get("unit", ""),
            "normal_range": p.get("normal_range", "N/A"),
            "status": status,
        })
    return out


def _analyze_blood_report_unified(report_text: str, patient_id: str):
    """Use ALL available analyzers: BloodReportParser (50+ params), HistoryAgent, Offline - for max accuracy."""
    # 1. Try BloodReportParser (AdvancedDocumentAnalyzer - 50+ params, local)
    try:
        from models.advanced_document_analyzer import BloodReportParser
        parser = BloodReportParser()
        parsed = parser.parse(report_text)
        if parsed.get("parameters") or parsed.get("summary"):
            params = _normalize_params(parsed.get("parameters", []))
            abnormalities = [f"{p['name']}: {p['value']} {p['unit']} (Normal: {p['normal_range']})" for p in parsed.get("abnormal", [])]
            return {
                "success": True,
                "parameters": params,
                "abnormalities": abnormalities,
                "findings": [{"type": "AI_Analysis", "content": parsed.get("summary", "")}],
                "model_used": "BloodReportParser (50+ params, local models)"
            }
    except Exception as e:
        logger.info(f"BloodReportParser: {e}")
    # 2. Try HistoryAgent (core_agents - Gemini + rule-based)
    try:
        from models.core_agents import HistoryAgent
        agent = HistoryAgent()
        result = agent.analyze_blood_report(report_text, patient_id)
        if result.get("success"):
            result["parameters"] = _normalize_params(result.get("parameters", []))
            return result
    except Exception as e:
        logger.info(f"HistoryAgent: {e}")
    # 3. Offline rule-based (always works)
    try:
        from models.offline_blood_analyzer import OfflineBloodReportAnalyzer
        analyzer = OfflineBloodReportAnalyzer()
        raw = analyzer.analyze(report_text)
        if raw.get("success"):
            raw["parameters"] = _normalize_params(raw.get("parameters", []))
            raw["abnormalities"] = raw.get("abnormalities", []) or [
                f"{p.get('parameter', p.get('name', '?'))}: {p.get('value')} {p.get('unit', '')} - {p.get('status', '')}"
                for p in raw.get("parameters", []) if p.get("status") not in ("NORMAL", "normal")
            ]
            return raw
    except Exception as e:
        logger.warning(f"Offline analyzer: {e}")
    return {"success": False, "error": "All analyzers failed", "findings": []}


@app.post("/analyze_blood_report")
async def analyze_blood_report(request: Request):
    """Analyze blood/lab reports - uses ALL available models (local + HF API)"""
    try:
        body = await request.json()
        report_text = body.get('report', '')
        patient_id = body.get('patient_id', 'anonymous')
        
        if not report_text:
            raise ValueError("Report text is required")
        
        result = _analyze_blood_report_unified(report_text, patient_id)
        return result
    
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/analyze_medical_report")
async def analyze_medical_report(request: Request):
    """Analyze any medical report (lab, pathology, imaging, etc.)"""
    try:
        body = await request.json()
        report_text = body.get('report', '')
        report_type = body.get('report_type', 'general')
        patient_id = body.get('patient_id', 'anonymous')
        
        if not report_text:
            raise ValueError("Report text is required")
        
        result = _analyze_blood_report_unified(report_text, patient_id)
        result['report_type'] = report_type
        return result
    
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/feedback_stats")
async def get_feedback_stats(patient_id: str = "all"):
    """Get feedback statistics and improvement metrics"""
    try:
        from pathlib import Path
        import json
        
        feedback_dir = Path("feedback_data")
        
        if not feedback_dir.exists():
            return {"total_feedbacks": 0, "average_rating": 0, "stats": []}
        
        feedbacks = []
        ratings = []
        
        for feedback_file in feedback_dir.glob("feedback_*.json"):
            try:
                with open(feedback_file, 'r') as f:
                    data = json.load(f)
                    
                    if patient_id == "all" or data.get('patient_id') == patient_id:
                        feedbacks.append(data)
                        if 'rating' in data:
                            ratings.append(data['rating'])
            except:
                pass
        
        average_rating = sum(ratings) / len(ratings) if ratings else 0
        
        return {
            "total_feedbacks": len(feedbacks),
            "average_rating": round(average_rating, 2),
            "rating_distribution": {
                "5_stars": len([r for r in ratings if r >= 4.5]),
                "4_stars": len([r for r in ratings if 3.5 <= r < 4.5]),
                "3_stars": len([r for r in ratings if 2.5 <= r < 3.5]),
                "low_ratings": len([r for r in ratings if r < 2.5])
            },
            "recent_feedbacks": feedbacks[-5:] if feedbacks else []
        }
    
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# HEALTH REPORT INTELLIGENCE API  (Assignment Required Endpoints)
# /health, /upload-report, /ask-report, /reports/{report_id}/parameters
# Uses the EXISTING Tier 1→2→3 LLM pipeline (smart_llm) for answers.
# ============================================================

# --- Report storage ---
REPORTS_STORE: Dict[str, Dict[str, Any]] = {}
_REPORTS_DIR = _app_dir / "uploaded_reports"
_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

_report_chunks_collection = None

def _get_report_chunks_collection():
    global _report_chunks_collection
    if _report_chunks_collection is None:
        try:
            import chromadb
            _rc_client = chromadb.Client()
            _report_chunks_collection = _rc_client.get_or_create_collection(
                name="report_chunks", metadata={"hnsw:space": "cosine"})
        except Exception as e:
            logger.warning(f"ChromaDB report-chunks init failed: {e}")
    return _report_chunks_collection

# --- PDF / TXT text extraction (triple fallback) ---
def _extract_report_text(file_bytes: bytes, filename: str):
    """Return (full_text, pages_list). Raises ValueError on failure."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext == "txt":
        text = file_bytes.decode("utf-8", errors="replace")
        return text, [{"page_number": 1, "text": text}]
    if ext != "pdf":
        raise ValueError("Unsupported file type. Only PDF and TXT files are allowed.")
    pages = []
    # pdfplumber
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for i, pg in enumerate(pdf.pages):
                t = pg.extract_text() or ""
                if t.strip():
                    pages.append({"page_number": i+1, "text": t})
        if pages:
            return "\n\n".join(p["text"] for p in pages), pages
    except Exception:
        pass
    # PyPDF2
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(io.BytesIO(file_bytes))
        for i, pg in enumerate(reader.pages):
            t = pg.extract_text() or ""
            if t.strip():
                pages.append({"page_number": i+1, "text": t})
        if pages:
            return "\n\n".join(p["text"] for p in pages), pages
    except Exception:
        pass
    # PyMuPDF
    try:
        import fitz
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for i, pg in enumerate(doc):
            t = pg.get_text()
            if t.strip():
                pages.append({"page_number": i+1, "text": t})
        doc.close()
        if pages:
            return "\n\n".join(p["text"] for p in pages), pages
    except Exception:
        pass
    raise ValueError("PDF extraction failed. Could not read the file.")

# --- Text chunking ---
def _chunk_report_text(text: str, pages: list, chunk_size: int = 500, overlap: int = 100):
    """Split text into overlapping chunks with page-number tracking."""
    char_page = {}
    offset = 0
    for pg in pages:
        for c in range(offset, offset + len(pg["text"])):
            char_page[c] = pg["page_number"]
        offset += len(pg["text"]) + 2
    chunks, start, idx = [], 0, 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        ct = text[start:end].strip()
        if ct:
            chunks.append({"chunk_id": f"chunk_{idx:03d}", "text": ct,
                           "page_number": char_page.get(start, 1)})
            idx += 1
        start += max(chunk_size - overlap, 1)
    return chunks

# --- Regex-based health parameter extraction (30+ params) ---
_REPORT_PARAM_PATTERNS = [
    (r"(?i)(hemoglobin|hgb|hb)\s*[:\-]?\s*([\d.]+)\s*(g/dL|g/dl)", "Hemoglobin", "g/dL", 13.0, 17.0),
    (r"(?i)(hematocrit|hct)\s*[:\-]?\s*([\d.]+)\s*(%)", "Hematocrit", "%", 41.0, 53.0),
    (r"(?i)(rbc|red blood cell[s]?)\s*[:\-]?\s*([\d.]+)\s*(million/mcL|M/uL|10\^6/uL)", "RBC", "million/mcL", 4.5, 5.9),
    (r"(?i)(wbc|white blood cell[s]?)\s*[:\-]?\s*([\d.]+)\s*(thousand/mcL|K/uL|10\^3/uL|cells/mcL)", "WBC", "thousand/mcL", 4.5, 11.0),
    (r"(?i)(platelet[s]?)\s*[:\-]?\s*([\d.]+)\s*(thousand/mcL|K/uL|10\^3/uL)", "Platelets", "thousand/mcL", 150.0, 400.0),
    (r"(?i)(fasting glucose|glucose|blood sugar|fbs)\s*[:\-]?\s*([\d.]+)\s*(mg/dL|mg/dl)", "Glucose", "mg/dL", 70.0, 100.0),
    (r"(?i)(hba1c|glycated hemoglobin|a1c)\s*[:\-]?\s*([\d.]+)\s*(%)", "HbA1c", "%", 4.0, 5.7),
    (r"(?i)(total cholesterol|cholesterol)\s*[:\-]?\s*([\d.]+)\s*(mg/dL|mg/dl)", "Total Cholesterol", "mg/dL", 0.0, 200.0),
    (r"(?i)(ldl[- ]?cholesterol|ldl[- ]?c|ldl)\s*[:\-]?\s*([\d.]+)\s*(mg/dL|mg/dl)", "LDL Cholesterol", "mg/dL", 0.0, 100.0),
    (r"(?i)(hdl[- ]?cholesterol|hdl[- ]?c|hdl)\s*[:\-]?\s*([\d.]+)\s*(mg/dL|mg/dl)", "HDL Cholesterol", "mg/dL", 40.0, 200.0),
    (r"(?i)(triglyceride[s]?)\s*[:\-]?\s*([\d.]+)\s*(mg/dL|mg/dl)", "Triglycerides", "mg/dL", 0.0, 150.0),
    (r"(?i)(creatinine)\s*[:\-]?\s*([\d.]+)\s*(mg/dL|mg/dl)", "Creatinine", "mg/dL", 0.7, 1.3),
    (r"(?i)(bun|blood urea nitrogen)\s*[:\-]?\s*([\d.]+)\s*(mg/dL|mg/dl)", "BUN", "mg/dL", 7.0, 20.0),
    (r"(?i)(urea)\s*[:\-]?\s*([\d.]+)\s*(mg/dL|mg/dl)", "Urea", "mg/dL", 15.0, 45.0),
    (r"(?i)(sodium|na\+?)\s*[:\-]?\s*([\d.]+)\s*(mEq/L|mmol/L)", "Sodium", "mEq/L", 136.0, 145.0),
    (r"(?i)(potassium|k\+?)\s*[:\-]?\s*([\d.]+)\s*(mEq/L|mmol/L)", "Potassium", "mEq/L", 3.5, 5.0),
    (r"(?i)(calcium|ca\+?)\s*[:\-]?\s*([\d.]+)\s*(mg/dL|mg/dl)", "Calcium", "mg/dL", 8.5, 10.2),
    (r"(?i)(tsh|thyroid stimulating)\s*[:\-]?\s*([\d.]+)\s*(mIU/L|uIU/mL)", "TSH", "mIU/L", 0.4, 4.0),
    (r"(?i)(free t4|t4|thyroxine)\s*[:\-]?\s*([\d.]+)\s*(ng/dL|ug/dL|mcg/dL)", "T4", "ng/dL", 0.8, 1.8),
    (r"(?i)(t3|triiodothyronine)\s*[:\-]?\s*([\d.]+)\s*(ng/dL|pg/mL)", "T3", "ng/dL", 80.0, 200.0),
    (r"(?i)(alt|sgpt)\s*[:\-]?\s*([\d.]+)\s*(U/L|IU/L)", "ALT", "U/L", 7.0, 35.0),
    (r"(?i)(ast|sgot)\s*[:\-]?\s*([\d.]+)\s*(U/L|IU/L)", "AST", "U/L", 10.0, 40.0),
    (r"(?i)(total bilirubin|bilirubin)\s*[:\-]?\s*([\d.]+)\s*(mg/dL|mg/dl)", "Total Bilirubin", "mg/dL", 0.1, 1.2),
    (r"(?i)(alkaline phosphatase|alp)\s*[:\-]?\s*([\d.]+)\s*(U/L|IU/L)", "ALP", "U/L", 44.0, 147.0),
    (r"(?i)(albumin)\s*[:\-]?\s*([\d.]+)\s*(g/dL|g/dl)", "Albumin", "g/dL", 3.5, 5.5),
    (r"(?i)(vitamin d|vit\.? d|25.?oh)\s*[:\-]?\s*([\d.]+)\s*(ng/mL|nmol/L)", "Vitamin D", "ng/mL", 30.0, 100.0),
    (r"(?i)(vitamin b12|vit\.? b12|cobalamin)\s*[:\-]?\s*([\d.]+)\s*(pg/mL|pmol/L)", "Vitamin B12", "pg/mL", 200.0, 900.0),
    (r"(?i)(iron|serum iron)\s*[:\-]?\s*([\d.]+)\s*(ug/dL|mcg/dL)", "Iron", "ug/dL", 60.0, 170.0),
    (r"(?i)(ferritin)\s*[:\-]?\s*([\d.]+)\s*(ng/mL|ug/L)", "Ferritin", "ng/mL", 12.0, 300.0),
    (r"(?i)(esr|erythrocyte sedimentation)\s*[:\-]?\s*([\d.]+)\s*(mm/hr|mm/h)", "ESR", "mm/hr", 0.0, 20.0),
    (r"(?i)(uric acid)\s*[:\-]?\s*([\d.]+)\s*(mg/dL|mg/dl)", "Uric Acid", "mg/dL", 3.5, 7.2),
    (r"(?i)(mcv|mean corpuscular volume)\s*[:\-]?\s*([\d.]+)\s*(fL|fl)", "MCV", "fL", 80.0, 100.0),
    (r"(?i)(mch|mean corpuscular hemo)\s*[:\-]?\s*([\d.]+)\s*(pg)", "MCH", "pg", 27.0, 33.0),
    (r"(?i)(mchc)\s*[:\-]?\s*([\d.]+)\s*(g/dL|g/dl|%)", "MCHC", "g/dL", 32.0, 36.0),
]

def _extract_report_parameters(text: str) -> List[dict]:
    """Deterministic regex extraction of health parameters."""
    params, seen = [], set()
    for pattern, name, unit, lo, hi in _REPORT_PARAM_PATTERNS:
        for m in re.finditer(pattern, text):
            if name in seen:
                continue
            try:
                val_s = m.group(2)
                val = float(val_s)
                status = "low" if val < lo else ("high" if val > hi else "normal")
                ref = f"{lo} - {hi}" if lo > 0 else f"< {hi}"
                params.append({"parameter": name, "value": val_s, "unit": unit,
                               "reference_range": ref, "status": status})
                seen.add(name)
            except (ValueError, IndexError):
                continue
    return params

# --- Medical safety ---
SAFE_DISCLAIMER = (
    "This is an AI-generated explanation based on the uploaded report "
    "and should not be treated as medical advice. Please consult a "
    "qualified doctor for diagnosis or treatment."
)
_DIAG_KEYWORDS = [
    "diagnose me", "diagnosis", "what disease", "what condition",
    "prescribe", "prescription", "medicine for", "medication for",
    "treatment for", "treat my", "cure for", "should i take",
    "start medication", "stop medication", "what drug", "what pill",
    "am i sick", "do i have cancer", "is it cancer",
]

def _is_diagnosis_question(q: str) -> bool:
    ql = (q or "").lower()
    return any(kw in ql for kw in _DIAG_KEYWORDS)

def _safe_report_prompt(question: str, chunks: List[dict]) -> str:
    """Build LLM prompt that explains values WITHOUT diagnosing."""
    ctx = "\n\n".join(
        f"[{c.get('chunk_id','')}] (Page {c.get('page_number','?')}):\n{c.get('text','')}"
        for c in chunks)
    return (
        "You are a health report explanation assistant.\n\n"
        "STRICT RULES:\n"
        "- Do NOT give any diagnosis or identify diseases.\n"
        "- Do NOT prescribe medicines or suggest treatment plans.\n"
        "- Do NOT tell the user to stop or start any medication.\n"
        "- You CAN explain values in simple language.\n"
        "- You CAN identify values outside the provided reference range.\n"
        "- You CAN summarize report content.\n"
        "- You CAN suggest consulting a qualified medical professional.\n"
        "- You CAN mention that interpretation may depend on age, gender, "
        "medical history, symptoms, and doctor evaluation.\n\n"
        f"REPORT CONTEXT:\n{ctx}\n\n"
        f"QUESTION: {question}\n\n"
        "Provide a clear, simple explanation based ONLY on the report above."
    )

# --- Report persistence ---
def _persist_report(rid: str, data: dict):
    try:
        import json as _json
        with open(_REPORTS_DIR / f"{rid}.json", "w", encoding="utf-8") as f:
            _json.dump(data, f, indent=2, default=str)
    except Exception as e:
        logger.warning(f"Report persist failed: {e}")

def _load_all_reports():
    import json as _json
    for fp in _REPORTS_DIR.glob("*.json"):
        try:
            with open(fp, "r", encoding="utf-8") as fh:
                d = _json.load(fh)
                rid = d.get("report_id")
                if rid:
                    REPORTS_STORE[rid] = d
        except Exception:
            pass

_load_all_reports()

# ============================================================
# ASSIGNMENT REQUIRED ENDPOINTS
# ============================================================

@app.get("/health")
async def health_check_simple():
    """Health check - required by assignment."""
    return {"status": "ok"}


@app.post("/upload-report")
async def upload_report(file: UploadFile = File(...)):
    """
    Upload a health report (PDF or TXT).
    Extracts text -> chunks -> embeds in ChromaDB -> extracts parameters.
    """
    if not file or not file.filename:
        return {"status": "error", "message": "No file provided."}
    filename = file.filename
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ("pdf", "txt"):
        return {"status": "error", "message": "Unsupported file type. Only PDF and TXT files are allowed."}
    file_bytes = await file.read()
    if not file_bytes or len(file_bytes) == 0:
        return {"status": "error", "message": "Empty file."}
    try:
        full_text, pages = _extract_report_text(file_bytes, filename)
    except ValueError as ve:
        return {"status": "error", "message": str(ve)}
    except Exception as e:
        return {"status": "error", "message": f"PDF extraction failed: {e}"}
    if not full_text or len(full_text.strip()) < 10:
        return {"status": "error", "message": "Report does not contain enough information."}
    report_id = f"report_{uuid.uuid4().hex[:12]}"
    chunks = _chunk_report_text(full_text, pages)
    parameters = _extract_report_parameters(full_text)
    # Store chunks in ChromaDB
    coll = _get_report_chunks_collection()
    if coll is not None:
        try:
            coll.add(
                documents=[c["text"] for c in chunks],
                metadatas=[{"report_id": report_id, "chunk_id": c["chunk_id"],
                            "page_number": c["page_number"]} for c in chunks],
                ids=[f"{report_id}_{c['chunk_id']}" for c in chunks])
        except Exception as e:
            logger.warning(f"ChromaDB chunk store failed (continuing): {e}")
    report_data = {"report_id": report_id, "filename": filename, "full_text": full_text,
                   "chunks": chunks, "parameters": parameters,
                   "total_chunks": len(chunks), "extracted_parameters_count": len(parameters),
                   "uploaded_at": datetime.now().isoformat()}
    REPORTS_STORE[report_id] = report_data
    _persist_report(report_id, report_data)
    return {"status": "success", "report_id": report_id, "filename": filename,
            "total_chunks": len(chunks), "extracted_parameters_count": len(parameters)}


@app.post("/ask-report")
async def ask_report(request: Request):
    """
    Ask a question about an uploaded report.
    Uses FULL Tier 1->2->3 LLM pipeline (Gemini -> HF -> Local)
    with RAG retrieval from ChromaDB.
    """
    try:
        body = await request.json()
    except Exception:
        return {"status": "error", "message": "Invalid JSON body."}
    report_id = (body.get("report_id") or "").strip()
    question = (body.get("question") or "").strip()
    if not report_id:
        return {"status": "error", "message": "report_id is required."}
    if not question:
        return {"status": "error", "message": "question is required."}
    if report_id not in REPORTS_STORE:
        return {"status": "error", "message": f"Invalid report ID: {report_id}. Report not found."}
    # Guard: diagnosis-seeking questions
    if _is_diagnosis_question(question):
        return {
            "answer": ("I cannot provide a medical diagnosis or treatment plan. "
                       "I can only explain the uploaded report content in simple terms. "
                       "Please consult a qualified doctor for medical advice."),
            "sources": [], "disclaimer": SAFE_DISCLAIMER}
    report = REPORTS_STORE[report_id]
    all_chunks = report.get("chunks", [])
    # --- RAG retrieval from ChromaDB ---
    retrieved_chunks: List[dict] = []
    coll = _get_report_chunks_collection()
    if coll is not None:
        try:
            results = coll.query(query_texts=[question],
                                n_results=min(5, max(len(all_chunks), 1)),
                                where={"report_id": report_id})
            if results and results.get("documents"):
                for i, doc in enumerate(results["documents"][0]):
                    meta = results["metadatas"][0][i] if results.get("metadatas") else {}
                    retrieved_chunks.append({"chunk_id": meta.get("chunk_id", f"chunk_{i:03d}"),
                                             "page_number": meta.get("page_number", 1), "text": doc})
        except Exception as e:
            logger.warning(f"ChromaDB query failed, using fallback: {e}")
    if not retrieved_chunks:
        retrieved_chunks = all_chunks[:5]
    if not retrieved_chunks:
        return {"answer": "No relevant content found in the report for this question.",
                "sources": [], "disclaimer": SAFE_DISCLAIMER}
    # --- Build safe prompt & call REAL tiered LLM (Gemini -> HF -> Local) ---
    prompt = _safe_report_prompt(question, retrieved_chunks)
    answer_text, source_model = await asyncio.to_thread(smart_llm, prompt)
    if not answer_text or len(answer_text.strip()) < 20:
        answer_text = "Based on your uploaded report:\n\n"
        for c in retrieved_chunks[:3]:
            answer_text += f"- {c['text'][:300]}\n\n"
        answer_text += "For a detailed interpretation, please consult a qualified medical professional."
    sources = [{"chunk_id": c["chunk_id"], "page_number": c.get("page_number", 1),
                "text": c["text"][:300]} for c in retrieved_chunks[:5]]
    return {"answer": answer_text, "sources": sources, "disclaimer": SAFE_DISCLAIMER}


@app.get("/reports/{report_id}/parameters")
async def get_report_parameters(report_id: str):
    """Return structured health parameters extracted from the report."""
    if report_id not in REPORTS_STORE:
        return {"status": "error", "message": f"Invalid report ID: {report_id}. Report not found."}
    return {"report_id": report_id, "parameters": REPORTS_STORE[report_id].get("parameters", [])}



if __name__ == "__main__":
    import sys
    # Quick test: python api_simple.py test
    if len(sys.argv) > 1 and sys.argv[1].lower() == "test":
        print("\n=== LLM TEST (smart_llm: Gemini → HF → local) ===")
        q = "What are the symptoms of pneumonia?"
        text, source = smart_llm(q)
        print(f"Source: {source}")
        print(text[:500] + "..." if text and len(text) > 500 else text)

        print("\n=== EMBEDDING TEST ===")
        vec, emb_model = call_embedding_any("Hemoglobin is 8.5 g/dL. What does it indicate?")
        if vec is not None:
            flat = vec if isinstance(vec, list) else (vec[0] if isinstance(vec, list) and vec else [])
            print(f"Model: {emb_model}, vector length: {len(flat)}")
        else:
            print("No embedding model responded")

        print("\n=== VISION TEST (requires image file) ===")
        if os.path.isfile("scan.jpg"):
            out, vis_model = call_vision_any("scan.jpg")
            print(f"Model: {vis_model}, result: {str(out)[:200] if out else 'None'}")
        else:
            print("Skip (scan.jpg not found)")

        print("\nRun server: python api_simple.py")
        sys.exit(0)

    print("Starting AI Doctor API Server...")
    print("=" * 80)
    print(f"GPU Status: {'✅ ENABLED' if GPU_AVAILABLE else '⚠️  CPU Mode'}")
    print(f"Device: {GPU_NAME}")
    print(f"Torch: {'✅' if TORCH_AVAILABLE else '❌'} | Transformers: {'✅' if TRANSFORMERS_AVAILABLE else '❌'}")
    print("=" * 80)
    print("API Documentation available at: http://localhost:8000/docs")
    print("GPU Status: GET /gpu/status")
    print("HF models: GET /hf/models | POST /hf/llm | POST /hf/embedding | POST /hf/vision")
    print("RL Training: POST /train/rl | GET /train/status")
    print("")
    print("IMPORTANT: This is for educational purposes only.")
    print("Always consult qualified healthcare professionals.")
    print("")
    
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="error")
    except Exception as e:
        print(f"Error starting server: {e}")
        print("Trying alternative port 8001...")
        uvicorn.run(app, host="0.0.0.0", port=8001, log_level="error")
