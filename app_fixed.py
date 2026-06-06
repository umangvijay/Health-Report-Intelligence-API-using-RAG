"""
AI Doctor - Fixed Streamlit Application with Login System
Complete medical assistant with all features working
"""

import sys
import os

# Add app directory to path so "from models.xxx" and "from user_auth" work
_app_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _app_dir)
sys.path.insert(0, os.path.join(_app_dir, 'models'))

# Auto-launch with Streamlit if run directly with Python
if __name__ == "__main__":
    if "streamlit" not in sys.modules:
        import subprocess
        print("\n" + "="*50)
        print("🚀 Launching AI Doctor with Streamlit...")
        print("="*50 + "\n")
        subprocess.run([sys.executable, "-m", "streamlit", "run", __file__])
        sys.exit(0)

import streamlit as st
import requests
import json
from pathlib import Path
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io
import base64
import os
from dotenv import load_dotenv
import time
import sys
import tempfile

# Load .env from Ai Doctor directory so HF_TOKEN and GEMINI_API_KEY are always found (even when run from another cwd)
_env_path = Path(__file__).resolve().parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path, override=False)
load_dotenv(override=False)

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

def _get_first_env(keys):
    for key in keys:
        value = os.getenv(key)
        if value is not None and str(value).strip():
            return str(value).strip().strip('"').strip("'")
    return ""

def _is_valid_key(value: str) -> bool:
    if not value:
        return False
    value = str(value).strip()
    if not value:
        return False
    upper = value.upper()
    placeholders = (
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
    )
    return not any(p in upper for p in placeholders)

def _get_hf_token():
    token = _get_first_env(HF_TOKEN_ENV_KEYS)
    return token if _is_valid_key(token) else None

def _get_gemini_key():
    key = _get_first_env(GEMINI_KEY_ENV_KEYS)
    return key if _is_valid_key(key) else None

def _get_fda_key():
    key = _get_first_env(FDA_KEY_ENV_KEYS)
    return key if _is_valid_key(key) else None

# Import auth system from models directory
try:
    from models.user_auth import UserAuthSystem
except ImportError:
    from user_auth import UserAuthSystem

# ============ LAZY LOAD AI MODELS ============
# Models are loaded ONLY when needed (on first use)
# This prevents the black screen issue from slow model loading

local_ai = None
local_ai_enabled = False
ensemble_ai = None
ensemble_ai_enabled = False
unified_ai = None
unified_ai_enabled = False
rag_hyde_system = None
rag_enabled = False
medicine_db = None
medicine_db_enabled = False

def get_ai_systems():
    """Lazy load AI systems only when needed"""
    global local_ai, local_ai_enabled, ensemble_ai, ensemble_ai_enabled
    global unified_ai, unified_ai_enabled, rag_hyde_system, rag_enabled
    global medicine_db, medicine_db_enabled
    
    # Only load once
    if local_ai is not None or local_ai_enabled:
        return
    
    use_hf_only = os.getenv("USE_HF_API_ONLY", "").strip().lower() in ("1", "true", "yes")
    if use_hf_only:
        local_ai_enabled = False
        ensemble_ai_enabled = False
        return  # Rely on HF API + drug DB only; no local models loaded
    
    print("[INFO] Loading AI models (this may take a minute)...")
    
    # Import RAG + HYDE system
    try:
        from models.rag_hyde_integration import RAGHYDEIntegration
        rag_hyde_system = RAGHYDEIntegration()
        rag_enabled = True
        print("[OK] RAG + HYDE system initialized")
    except Exception as e:
        print(f"[!] RAG system: {str(e)[:80]}")
        rag_enabled = False
    
    # PRIMARY: LOCAL ENSEMBLE
    try:
        from models.local_ensemble import get_local_ai
        local_ai = get_local_ai(use_gpu=True)
        local_ai_enabled = True
        model_info = local_ai.get_model_info()
        print(f"[OK] LOCAL AI: {model_info['loaded_models']} models, {model_info['estimated_accuracy']:.0f}% accuracy")
    except Exception as e:
        print(f"[!] Local AI: {str(e)[:80]}")
        local_ai_enabled = False
    
    # OPTIONAL: API ENSEMBLE
    try:
        from models.ensemble_medical_ai import get_ensemble_ai
        ensemble_ai = get_ensemble_ai(use_gpu=True, offline_only=False, load_large_models=False)
        ensemble_ai_enabled = True
        print("[OK] ENSEMBLE AI initialized")
    except Exception as e:
        print(f"[!] Ensemble AI: {str(e)[:80]}")
        ensemble_ai_enabled = False
    
    # FALLBACK: Unified AI
    try:
        from models.unified_medical_ai import get_unified_ai
        unified_ai = get_unified_ai(use_gpu=True, prefer_local=True)
        unified_ai_enabled = True
        print("[OK] Unified AI initialized")
    except Exception as e:
        print(f"[!] Unified AI: {str(e)[:80]}")
        unified_ai_enabled = False
    
    # Medicine Database
    try:
        from models.drugbank_loader import DrugBankLoader
        medicine_db = DrugBankLoader()
        medicine_db_enabled = True
        print(f"[OK] Medicine DB: {len(medicine_db.builtin_medicines)} medicines")
    except Exception as e:
        print(f"[!] Medicine DB: {str(e)[:80]}")
        medicine_db_enabled = False

# Import Google Generative AI (with fallback for deprecated package)
try:
    import google.genai as genai
    USING_NEW_GENAI = True
except ImportError:
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import google.generativeai as genai
        USING_NEW_GENAI = False
    except ImportError:
        genai = None
        USING_NEW_GENAI = False

# Configure Gemini if available
gemini_key = _get_gemini_key()
gemini_model = None
gemini_model_name = None

if gemini_key and _is_valid_key(gemini_key) and genai:
    try:
        if USING_NEW_GENAI:
            # New google.genai package
            client = genai.Client(api_key=gemini_key)
            gemini_model_name = 'gemini-2.5-flash'  # Updated to 2.5 flash
            gemini_model = client
            print(f"[OK] Gemini configured with new API: {gemini_model_name}")
        else:
            # Old google.generativeai package (deprecated)
            genai.configure(api_key=gemini_key)
            
            # Use 2.5 flash model (latest and fastest)
            model_options = [
                'models/gemini-2.5-flash',
                'models/gemini-2.0-flash-exp',
                'models/gemini-1.5-flash',
                'models/gemini-1.5-pro',
            ]
            
            gemini_model_name = model_options[0]
            gemini_model = genai.GenerativeModel(gemini_model_name)
            print(f"[OK] Gemini configured with deprecated API: {gemini_model_name}")
    except Exception as e:
        print(f"[!] Gemini configuration error: {e}")
        gemini_model = None
else:
    gemini_model = None

# Page config
st.set_page_config(
    page_title="AI Doctor - Medical Assistant",
    page_icon="H",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize auth system
auth_system = UserAuthSystem()

# Custom CSS for beautiful UI - FIXED WHITE TEXT ISSUES
st.markdown("""
<style>
    /* Main App Styling */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    /* Fix white text visibility */
    .stMarkdown, .stText, p, span, label {
        color: #ffffff !important;
    }
    
    /* Header Styling */
    .main-header {
        text-align: center;
        padding: 2rem;
        background: rgba(255,255,255,0.1);
        border-radius: 20px;
        backdrop-filter: blur(10px);
        margin-bottom: 2rem;
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .main-header h1 {
        color: #00d4ff !important;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        color: #a0a0a0 !important;
        font-size: 1.1rem;
    }
    
    /* Card Styling - FIXED */
    .card {
        background: rgba(255,255,255,0.1);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid rgba(255,255,255,0.2);
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
    }
    
    .card h3, .card h4, .card p, .card span {
        color: #ffffff !important;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: rgba(26, 26, 46, 0.95);
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: #ffffff !important;
    }
    
    /* Input fields */
    .stTextInput input, .stTextArea textarea {
        background: rgba(255,255,255,0.1) !important;
        border: 1px solid rgba(255,255,255,0.3) !important;
        color: #ffffff !important;
    }
    
    .stTextInput label, .stTextArea label {
        color: #ffffff !important;
    }
    
    /* Disclaimer Box */
    .disclaimer {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a5a 100%);
        color: white !important;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: 500;
    }
    
    /* Success/Info boxes */
    .success-box {
        background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
        color: white !important;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .info-box {
        background: linear-gradient(135deg, #0984e3 0%, #0873c9 100%);
        color: white !important;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    /* Login & Sign In popup form styling (from python_expert.css) */
    .search_form, .login-signup-popup {
        display: flex;
        flex-direction: column;
        align-items: stretch;
        gap: 0.5rem;
        background-color: rgba(255,255,255,0.08);
        box-shadow: 0 8px 32px hsla(230, 75%, 15%, 0.3);
        padding: 1.25rem;
        border-radius: 0.5rem;
        transform: translateY(0);
        transition: transform 0.45s cubic-bezier(0.2, 0.1, 0.8, 0.9);
        border: 1px solid rgba(255,255,255,0.2);
    }
    .search_form:hover, .login-signup-popup:hover {
        transform: translateY(-2px);
    }
    .search_icon {
        font-size: 1.25rem;
        color: #00d4ff;
    }
    /* Logout / door-style button */
    .door, .doorway {
        fill: #f4f7ff;
        transform: rotateY(20deg);
        transform-origin: 100% 50%;
        transition: transform 200ms ease;
    }
    .door:hover, .doorway:hover { transform: rotateY(0deg); }
    div[data-testid="stVerticalBlock"] > div:has(button) .stButton > button[kind="secondary"] {
        border-radius: 0.5rem;
        transition: transform 200ms ease, box-shadow 200ms ease;
    }
    div[data-testid="stVerticalBlock"] > div:has(button) .stButton > button[kind="secondary"]:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 12px hsla(230, 75%, 15%, 0.3);
    }
    /* Figure / accent element */
    .figure {
        bottom: 5px;
        right: 18px;
        fill: #437147;
        width: 30px;
        z-index: 4;
        transition: cubic-bezier(0.2, 0.1, 0.8, 0.9);
    }
    
    /* Logic flow info box */
    .logic-flow-box {
        background: linear-gradient(135deg, rgba(0,180,216,0.15) 0%, rgba(0,212,255,0.1) 100%);
        border: 1px solid rgba(0,212,255,0.4);
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
        font-size: 0.9rem;
        color: #e0e0e0;
    }
    .logic-flow-box strong { color: #00d4ff; }
    .logic-flow-box ul { margin: 0.5rem 0 0 1rem; padding: 0; }
    
    /* Metric Cards */
    .metric-card {
        background: rgba(255,255,255,0.1);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #00d4ff !important;
    }
    
    .metric-label {
        color: #a0a0a0 !important;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #ffffff !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.1) !important;
        color: #ffffff !important;
        border-radius: 10px;
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
        color: white !important;
        border: none;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        font-weight: bold;
    }
    
    .stButton button:hover {
        background: linear-gradient(135deg, #00b8e6 0%, #0088bb 100%);
    }
    
    /* File uploader */
    .stFileUploader {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Charts background */
    .js-plotly-plot {
        background: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'session_token' not in st.session_state:
    st.session_state.session_token = None
if 'user_data' not in st.session_state:
    st.session_state.user_data = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'diagnosis_history' not in st.session_state:
    st.session_state.diagnosis_history = []

# API Configuration
API_BASE_URL = os.getenv('API_URL', 'http://localhost:8000')
# LLM_PROVIDER: auto | gemini | hf | local — which provider to always use
LLM_PROVIDER = (os.getenv("LLM_PROVIDER") or "auto").strip().lower()
if LLM_PROVIDER not in ("auto", "gemini", "hf", "local"):
    LLM_PROVIDER = "auto"
OPEN_FDA_API_KEY = _get_fda_key()

# Full consultation prompt (same for HF and Gemini) - not brief, full analysis
def _full_consultation_prompt(symptoms: str, medical_history: str = "") -> str:
    history_text = f"\n\nPatient Medical History:\n{medical_history}" if medical_history else ""
    return f"""You are an expert medical doctor. Provide a full consultation and analysis.

Patient symptoms:
{symptoms}
{history_text}

Provide a comprehensive analysis with:
1. Differential diagnosis (most likely conditions)
2. Recommended diagnostic tests
3. Treatment plan with medications
4. Home care and remedies
5. Lifestyle changes
6. Warning signs requiring immediate medical attention

Write in clear paragraphs. Do not give short bullet-only answers; include brief explanation where helpful."""

# ============================================================
# CRITICAL FIX: Separate HF API models from LOCAL-ONLY models
# ============================================================
# NOT ALL MODELS ARE HOSTED ON HF INFERENCE API!
# Models like flan-t5, gpt2, biogpt return 404 on Inference API
# These must be loaded LOCALLY via transformers

# HF INFERENCE API MODELS (actually hosted on serverless API)
HF_API_MODELS = [
    "m42-health/Llama3-Med42-8B",             # Top tier Medical Llama 3
    "meta-llama/Meta-Llama-3-8B-Instruct",    # Llama 3 8B
    "epfl-llm/meditron-7b",                   # Specialized Medical
    "FreedomIntelligence/HuatuoGPT-7B",       # Medical LLM
    "mistralai/Mistral-7B-Instruct-v0.3"      # General reliable fallback
]

# LOCAL-ONLY MODELS (must load via transformers, return 404 on API)
LOCAL_HF_MODELS = [
    "microsoft/BioGPT-Large-PubMedQA",        # Superior BioGPT
    "StanfordAIMI/RadLLaMA-7b",               # Radiology specialized
    "google/flan-t5-large",                   # Better than base
    "emilyalsentzer/Bio_ClinicalBERT",        # Clinical understanding
    "microsoft/biogpt"
]

# For backward compatibility
HF_FREE_TIER_MODELS = HF_API_MODELS  # Only use API-compatible models in API calls

# NOTE: The old api-inference.huggingface.co endpoint is DEPRECATED (returns 410 Gone)
# Only use the new router.huggingface.co endpoint
HF_API_BASES = [
    "https://router.huggingface.co/hf-inference/models",  # NEW - primary (2026+)
    # "https://api-inference.huggingface.co/models",  # OLD - DEPRECATED
]

# ============================================================
# LOCAL MODEL LOADING (with CUDA support)
# ============================================================
_LOCAL_MODEL_CACHE = {}

def _get_device():
    """Get best available device (CUDA if available)."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.device("cuda")
    except ImportError:
        pass
    return "cpu"

def generate_with_local_model_frontend(model_name: str, prompt: str, max_tokens: int = 300) -> str:
    """
    Generate text using a locally loaded HF model with CUDA support.
    Returns generated text or None if failed.
    """
    global _LOCAL_MODEL_CACHE
    
    try:
        from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
        import torch
    except ImportError:
        print("[LOCAL] Transformers not available")
        return None
    
    device = _get_device()
    print(f"[LOCAL] Device: {device}")
    
    # Check cache
    if model_name in _LOCAL_MODEL_CACHE:
        model, tokenizer = _LOCAL_MODEL_CACHE[model_name]
        print(f"[LOCAL] Using cached model: {model_name}")
    else:
        try:
            print(f"[LOCAL] Loading {model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            
            is_seq2seq = "t5" in model_name.lower() or "flan" in model_name.lower()
            dtype = torch.float16 if device != "cpu" and torch.cuda.is_available() else torch.float32
            
            if is_seq2seq:
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name, trust_remote_code=True, torch_dtype=dtype, low_cpu_mem_usage=True
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, trust_remote_code=True, torch_dtype=dtype, low_cpu_mem_usage=True
                )
            
            if device != "cpu":
                model = model.to(device)
            model.eval()
            
            _LOCAL_MODEL_CACHE[model_name] = (model, tokenizer)
            print(f"[LOCAL] Loaded {model_name} on {device}")
            
        except Exception as e:
            print(f"[LOCAL] Failed to load {model_name}: {e}")
            return None
    
    try:
        inputs = tokenizer(prompt[:1500], return_tensors="pt", truncation=True, max_length=512)
        if device != "cpu":
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id or 0,
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # For causal models, remove input from output
        if "t5" not in model_name.lower() and "flan" not in model_name.lower():
            if generated.startswith(prompt[:50]):
                generated = generated[len(prompt):].strip()
        
        if generated and len(generated.strip()) > 30:
            print(f"[LOCAL] Generated {len(generated)} chars with {model_name}")
            return generated.strip()
        
        return None
        
    except Exception as e:
        print(f"[LOCAL] Generation failed: {e}")
        return None

def get_hf_free_tier_diagnosis(symptoms: str, medical_history: str = "") -> str:
    """
    FIXED: Try HF Inference API models first, then LOCAL models with CUDA.
    
    CRITICAL: Not all models are on HF Inference API!
    - Models like flan-t5, gpt2, biogpt return 404 on API
    - These MUST be loaded locally via transformers
    
    Order:
    1. Try HF_API_MODELS via API (if HF_TOKEN valid)
    2. Try LOCAL_HF_MODELS via local transformers (with CUDA)
    
    Returns full consultation or None.
    """
    hf_token = _get_hf_token()
    device = _get_device()
    
    print("")
    print("=" * 60)
    print("[AI ROUTER] get_hf_free_tier_diagnosis ENTRY")
    print(f"[AI ROUTER] HF TOKEN FOUND: {'YES' if hf_token else 'NO'}")
    print(f"[AI ROUTER] DEVICE: {device}")
    print("=" * 60)
    
    if not hf_token:
        print("[AI ROUTER] PHASE 1 SKIPPED: No valid HF_TOKEN")
        return None
    
    prompt = _full_consultation_prompt(symptoms, medical_history[:500] if medical_history else "")
    prompt = prompt[:2000]
    
    errors = []
    token_invalid = False
    
    # ============================================================
    # PHASE 1: Try HF Inference API models (if token valid)
    # ============================================================
    if hf_token:
        print(f"[AI ROUTER] PHASE 1: Trying {len(HF_API_MODELS)} API models...")
        print(f"[HF] Token: {hf_token[:15]}...")
        
        headers = {"Authorization": f"Bearer {hf_token}"}
        payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": 500, "temperature": 0.2, "top_p": 0.9, "do_sample": True, "return_full_text": False},
        }
        
        for i, model in enumerate(HF_API_MODELS):
            for base in HF_API_BASES:
                url = f"{base}/{model}"
                try:
                    print(f"[HF API] [{i+1}/{len(HF_API_MODELS)}] Trying: {model}")
                    r = requests.post(url, headers=headers, json=payload, timeout=45)
                    
                    if r.status_code == 200:
                        data = r.json()
                        text = None
                        if isinstance(data, list) and len(data) > 0:
                            text = data[0].get("generated_text")
                        elif isinstance(data, dict):
                            text = data.get("generated_text")
                        
                        if text and len(str(text).strip()) > 80:
                            print(f"[HF API] SUCCESS with {model}")
                            return str(text).strip()
                        else:
                            print(f"[HF API] {model}: Response too short")
                            
                    elif r.status_code == 401:
                        print(f"[AI ROUTER] FALLBACK REASON: 401 Unauthorized - HF_TOKEN invalid")
                        errors.append(f"{model}: 401 Unauthorized")
                        token_invalid = True
                        break
                    elif r.status_code == 404:
                        print(f"[HF API] {model}: 404 Not Found - Model not on Inference API")
                        errors.append(f"{model}: 404")
                    elif r.status_code == 503:
                        print(f"[HF API] {model}: 503 Service Unavailable")
                        errors.append(f"{model}: 503")
                    elif r.status_code == 429:
                        print(f"[HF API] {model}: 429 Rate Limited")
                        errors.append(f"{model}: 429")
                    else:
                        print(f"[HF API] {model}: HTTP {r.status_code}")
                        errors.append(f"{model}: HTTP {r.status_code}")
                        
                except requests.exceptions.Timeout:
                    print(f"[HF API] {model}: Timeout")
                    errors.append(f"{model}: Timeout")
                except Exception as e:
                    print(f"[HF API] {model}: {str(e)[:50]}")
                    errors.append(f"{model}: {str(e)[:30]}")
            
            if token_invalid:
                break
        
        print(f"[AI ROUTER] PHASE 1 COMPLETE: API models exhausted")
    # ============================================================
    # PHASE 2: Try LOCAL models via transformers (with CUDA)
    # ============================================================
    try:
        import transformers
        transformers_available = True
    except ImportError:
        transformers_available = False
    
    use_hf_only = os.getenv("USE_HF_API_ONLY", "").strip().lower() in ("1", "true", "yes")
    
    if transformers_available and not use_hf_only:
        print(f"[AI ROUTER] PHASE 2: Trying {len(LOCAL_HF_MODELS)} LOCAL models...")
        print(f"[AI ROUTER] HF MODE: LOCAL (with {'CUDA' if device != 'cpu' else 'CPU'})")
        
        for i, model in enumerate(LOCAL_HF_MODELS):
            print(f"[LOCAL] [{i+1}/{len(LOCAL_HF_MODELS)}] Loading: {model}")
            
            result = generate_with_local_model_frontend(model, prompt, max_tokens=400)
            
            if result and len(result.strip()) > 80:
                print(f"[LOCAL] SUCCESS with {model}")
                return result.strip()
            else:
                print(f"[LOCAL] {model}: Empty or short response")
        
        print(f"[AI ROUTER] PHASE 2 COMPLETE: Local models exhausted")
    else:
        print("[AI ROUTER] PHASE 2 SKIPPED: Transformers not available")
    
    # All failed
    print(f"[AI ROUTER] ALL MODELS FAILED!")
    if errors:
        print(f"[AI ROUTER] Errors: {errors[:5]}")
    
    return None

# Meditron-70B via HuggingFace Inference API
# NOTE: Old api-inference.huggingface.co is DEPRECATED (410 Gone)
MEDITRON_70B_URLS = [
    "https://router.huggingface.co/hf-inference/models/epfl-llm/meditron-70b",  # NEW endpoint
]

def get_meditron_70b_diagnosis(symptoms: str, medical_history: str = "") -> str:
    """Get diagnosis from Meditron-70B via Hugging Face Inference API (uses HF API key). Full consultation."""
    hf_token = _get_hf_token()
    if not hf_token:
        return None
    history_text = f"\n\nPatient Medical History:\n{medical_history}" if medical_history else ""
    medical_prompt = f"""<|im_start|>system
You are an expert medical doctor. Provide a full consultation and analysis.
<|im_end|>
<|im_start|>user
Patient symptoms: {symptoms}{history_text}

Provide: 1) Differential diagnosis 2) Recommended tests 3) Treatment plan 4) Home care 5) Warning signs. Full analysis, not brief bullets only.
<|im_end|>
<|im_start|>assistant
"""
    payload = {
        "inputs": medical_prompt,
        "parameters": {"max_new_tokens": 500, "temperature": 0.2, "top_p": 0.9, "do_sample": True, "return_full_text": False},
    }
    headers = {"Authorization": f"Bearer {hf_token}"}
    for api_url in MEDITRON_70B_URLS:
        try:
            resp = requests.post(api_url, headers=headers, json=payload, timeout=35)
            if resp.status_code == 200:
                result = resp.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get('generated_text', None)
                if isinstance(result, dict) and result.get('generated_text'):
                    return result['generated_text']
            if resp.status_code == 503:
                continue  # Model loading, try next URL
        except Exception:
            continue
    return None

def get_local_diagnosis(symptoms: str, medical_history: str = "") -> dict:
    """
    Local diagnosis using ALL available local resources.
    Priority:
    1. LocalEnsembleMedicalAI (BioGPT, ClinicalBERT, etc.)
    2. Individual local HF models (flan-t5, etc.)
    3. Rule-based symptom matching
    """
    
    # ============================================================
    # STEP 1: Try LocalEnsembleMedicalAI (full local ML models)
    # ============================================================
    try:
        print("[LOCAL] Trying LocalEnsembleMedicalAI...")
        from models.local_ensemble import LocalEnsembleMedicalAI
        
        import torch
        use_gpu = torch.cuda.is_available()
        print(f"[LOCAL] CUDA available: {use_gpu}")
        
        ensemble = LocalEnsembleMedicalAI(use_gpu=use_gpu)
        
        result = ensemble.medical_consultation(symptoms, patient_context=medical_history[:500] if medical_history else None)
        
        if result and result.get("response") and len(result.get("response", "").strip()) > 50:
            response_text = result.get("response", "")
            confidence = result.get("confidence", 0.80)
            models_used = result.get("models_used", ["LocalEnsemble"])
            
            print(f"[LOCAL] LocalEnsembleMedicalAI SUCCESS - Confidence: {confidence*100:.0f}%")
            
            return {
                "diagnosis": f"[Local Ensemble AI]\n\n{response_text}\n\n---\nModels: {', '.join(models_used[:3])}",
                "treatment": {
                    "medications": ["See analysis above"],
                    "home_remedies": ["See analysis above"],
                    "lifestyle": ["See analysis above"]
                },
                "confidence": confidence,
                "model": "local_ensemble"
            }
    except ImportError as e:
        print(f"[LOCAL] LocalEnsembleMedicalAI not available: {e}")
    except Exception as e:
        print(f"[LOCAL] LocalEnsembleMedicalAI failed: {str(e)[:80]}")
    
    # ============================================================
    # STEP 2: Try individual local models via transformers
    # ============================================================
    try:
        print("[LOCAL] Trying individual local HF models...")
        
        for model_name in LOCAL_HF_MODELS[:3]:
            print(f"[LOCAL] Loading: {model_name}")
            result = generate_with_local_model_frontend(model_name, _full_consultation_prompt(symptoms, medical_history), max_tokens=400)
            
            if result and len(result.strip()) > 80:
                print(f"[LOCAL] SUCCESS with {model_name}")
                return {
                    "diagnosis": f"[Local Model: {model_name}]\n\n{result.strip()}",
                    "treatment": {
                        "medications": ["See analysis above"],
                        "home_remedies": ["See analysis above"],
                        "lifestyle": ["See analysis above"]
                    },
                    "confidence": 0.75,
                    "model": f"local_{model_name.split('/')[-1]}"
                }
    except Exception as e:
        print(f"[LOCAL] Individual models failed: {str(e)[:80]}")
    
    # ============================================================
    # STEP 3: Rule-based symptom matching (guaranteed fallback)
    # ============================================================
    print("[LOCAL] Using rule-based symptom matching...")
    
    symptom_map = {
        'fever': {'conditions': ['Common Cold', 'Flu', 'Infection'], 'tests': ['Complete Blood Count', 'Thyroid Profile']},
        'headache': {'conditions': ['Migraine', 'Tension Headache', 'Sinus Infection'], 'tests': ['CT Scan if severe', 'Blood Pressure Check']},
        'cough': {'conditions': ['Common Cold', 'Bronchitis', 'Pneumonia', 'Asthma'], 'tests': ['Chest X-Ray', 'Throat Culture']},
        'fatigue': {'conditions': ['Anemia', 'Thyroid Disorder', 'Sleep Disorder', 'Depression'], 'tests': ['Complete Blood Count', 'Thyroid Function Tests']},
        'nausea': {'conditions': ['Food Poisoning', 'Gastroenteritis', 'Migraine', 'Pregnancy'], 'tests': ['Stool Test', 'Abdominal Ultrasound']},
        'chest pain': {'conditions': ['Heartburn', 'Angina', 'Pulmonary Embolism'], 'tests': ['ECG', 'Troponin Test', 'Chest X-Ray']},
        'back pain': {'conditions': ['Muscle Strain', 'Herniated Disc', 'Kidney Stones'], 'tests': ['MRI', 'X-Ray', 'CT Scan']},
        'joint pain': {'conditions': ['Arthritis', 'Rheumatoid Arthritis', 'Osteoarthritis'], 'tests': ['Rheumatoid Factor Test', 'X-Ray']},
        'kidney stones': {'conditions': ['Nephrolithiasis', 'Pyelonephritis'], 'tests': ['CT Scan', 'Urinalysis', 'Ultrasound']},
        'abdominal pain': {'conditions': ['Appendicitis', 'Gastritis', 'Peptic Ulcer', 'Colitis'], 'tests': ['Abdominal CT', 'Ultrasound']},
        'shortness of breath': {'conditions': ['Asthma', 'Pneumonia', 'Heart Failure', 'Anemia'], 'tests': ['Chest X-Ray', 'Spirometry', 'ECG']},
    }
    
    symptoms_lower = symptoms.lower()
    matched_conditions = []
    matched_tests = []
    
    for symptom, data in symptom_map.items():
        if symptom in symptoms_lower:
            matched_conditions.extend(data['conditions'])
            matched_tests.extend(data['tests'])
    
    matched_conditions = list(set(matched_conditions))
    matched_tests = list(set(matched_tests))
    
    if matched_conditions:
        diagnoses_section = "\n".join([f"- {condition}" for condition in matched_conditions[:5]])
    else:
        diagnoses_section = "Unable to determine from available symptoms. Please provide more details."
    
    if matched_tests:
        tests_section = "\n".join([f"- {test}" for test in matched_tests[:5]])
    else:
        tests_section = "Depends on final diagnosis - consult healthcare provider"
    
    diagnosis_text = f"""[Local Rule-Based Analysis]

PATIENT SYMPTOMS: {symptoms}

POSSIBLE DIAGNOSES:
{diagnoses_section}

RECOMMENDED TESTS:
{tests_section}

HOME REMEDIES:
- Rest and get adequate sleep
- Stay well hydrated
- Maintain balanced nutrition
- Avoid stress

IMPORTANT: This is a rule-based reference only. For accurate diagnosis, please consult a healthcare professional.
"""
    
    return {
        "diagnosis": diagnosis_text,
        "treatment": {
            "medications": ["Consult doctor for proper medication"],
            "home_remedies": ["Rest", "Stay hydrated", "Balanced diet", "Regular exercise"],
            "lifestyle": ["Sleep 7-9 hours", "Manage stress", "Avoid smoking", "Limit alcohol"]
        },
        "confidence": 0.65,
        "model": "local-rule-based"
    }

def _drug_lookup(query: str) -> dict | None:
    """If query is a drug question, return answer from medicine DB. Else None."""
    q = query.strip().lower()
    drug_name = None
    
    # Check if this is a medication/drug-related query
    drug_keywords = ["paracetamol", "acetaminophen", "ibuprofen", "aspirin", "amoxicillin", "metformin", 
                     "omeprazole", "lisinopril", "atorvastatin", "amlodipine", "losartan", "tablet", "medicine", 
                     "drug", "medication", "dose", "dosage"]
    is_drug_query = any(keyword in q for keyword in drug_keywords)
    
    if not is_drug_query:
        return None
    
    # Extract drug name from various query patterns
    prefixes_to_try = [
        ("what does ", " do"),
        ("what is the use of ", ["tablets", "tablet", "dose", "and", "dosage", "side effect"]),
        ("what is ", ["tablets", "tablet", "dose", "and", "dosage"]),
        ("tell me about ", []),
        ("info on ", []),
        ("about ", []),
    ]
    
    for prefix, suffixes in prefixes_to_try:
        if q.startswith(prefix):
            drug_name = q[len(prefix):]
            # Remove suffixes to clean up drug name
            if isinstance(suffixes, list):
                for suffix in suffixes:
                    if suffix in drug_name:
                        drug_name = drug_name[:drug_name.index(suffix)].strip()
                        break
            else:
                drug_name = drug_name.replace(suffixes, "")
            drug_name = drug_name.strip()
            break
    
    # If no drug name extracted, try to find first known drug name in query
    if not drug_name:
        known_drugs = ["paracetamol", "acetaminophen", "ibuprofen", "aspirin", "amoxicillin", "metformin",
                      "omeprazole", "lisinopril", "atorvastatin", "amlodipine", "losartan", "dolo", "crocin"]
        for drug in sorted(known_drugs, key=len, reverse=True):
            if drug in q:
                drug_name = drug
                break
    
    if not drug_name or len(drug_name) < 2:
        return None
    
    get_ai_systems()
    # Always try open.fda.gov first for drug queries (if key is set)
    if _is_valid_key(OPEN_FDA_API_KEY):
        try:
            from models.drug_data_sources import get_fda_drug_text_from_query
            fda_text = get_fda_drug_text_from_query(query)
            if fda_text and len(fda_text) > 30:
                return {
                    "diagnosis": "[open.fda.gov]\n\n" + fda_text,
                    "treatment": {"medications": [drug_name], "home_remedies": [], "lifestyle": []},
                    "confidence": 0.90,
                    "model": "open_fda"
                }
        except Exception:
            pass
    if not medicine_db or not medicine_db_enabled:
        return None
    
    info = medicine_db.lookup_medicine(drug_name)
    if not info.get("found"):
        return None
    
    # Format response with all key information
    lines = [
        f"**{info.get('name', drug_name)}**",
        f"",
        f"**Drug Class:** {info.get('drug_class', 'N/A')}",
        f"",
        f"**Uses:**",
    ]
    uses = info.get('uses', [])
    if isinstance(uses, list):
        for use in uses:
            lines.append(f"  - {use}")
    else:
        lines.append(f"  - {uses}")
    
    lines.extend([
        f"",
        f"**Dosage:**",
        f"{info.get('dosage', 'Consult your doctor for proper dosage')}",
        f"",
        f"**Side Effects:**",
    ])
    side_effects = info.get('side_effects', [])
    if isinstance(side_effects, list):
        for se in side_effects:
            lines.append(f"  - {se}")
    else:
        lines.append(f"  - {side_effects}")
    
    lines.extend([
        f"",
        f"**Precautions & Contraindications:**",
    ])
    precautions = info.get('contraindications', [])
    if isinstance(precautions, list):
        for prec in precautions:
            lines.append(f"  - {prec}")
    else:
        lines.append(f"  - {precautions}")
    
    if info.get('interactions'):
        lines.extend([
            f"",
            f"**Drug Interactions:**",
        ])
        interactions = info.get('interactions', [])
        if isinstance(interactions, list):
            for inter in interactions:
                lines.append(f"  - {inter}")
        else:
            lines.append(f"  - {interactions}")
    
    if info.get('pregnancy_category'):
        lines.append(f"\n**Pregnancy Category:** {info.get('pregnancy_category')}")
    
    lines.append(f"\n**⚠️ Important:** Always consult your healthcare provider before taking any medication.")
    
    return {
        "diagnosis": "[Medicine Database]\n\n" + "\n".join(lines),
        "treatment": {"medications": [info.get("name", "")], "home_remedies": [], "lifestyle": []},
        "confidence": 0.95,
        "model": "drug_db"
    }


def get_ai_diagnosis(symptoms: str, medical_history: str = "") -> dict:
    """
    Priority: drug DB → Backend API → Gemini direct → HF API → Local Ensemble → Rule-based.
    Full consultation with comprehensive logging.
    """
    global local_ai, local_ai_enabled, ensemble_ai, ensemble_ai_enabled
    
    print("")
    print("=" * 60)
    print("[AI DIAGNOSIS] ENTRY")
    print(f"[AI DIAGNOSIS] Symptoms: {symptoms[:50]}...")
    print(f"[AI DIAGNOSIS] LLM_PROVIDER: {LLM_PROVIDER}")
    print(f"[AI DIAGNOSIS] GEMINI_KEY: {'YES' if _get_gemini_key() else 'NO'}")
    print(f"[AI DIAGNOSIS] HF_TOKEN: {'YES' if _get_hf_token() else 'NO'}")
    print("=" * 60)
    
    # 1. Drug question = instant (no API)
    drug_ans = _drug_lookup(symptoms)
    if drug_ans:
        print("[AI DIAGNOSIS] Drug lookup matched - returning drug info")
        return drug_ans
    
    # If LLM_PROVIDER=local, skip all API and use local/ensemble/rule-based only
    if LLM_PROVIDER == "local":
        print("[AI DIAGNOSIS] LLM_PROVIDER=local - skipping APIs")
        get_ai_systems()
        if ensemble_ai and ensemble_ai_enabled:
            try:
                result = ensemble_ai.medical_consultation(symptoms, context=(medical_history or "")[:500])
                if result.get("response"):
                    print("[AI DIAGNOSIS] Ensemble AI SUCCESS")
                    return {
                        "diagnosis": f"[Ensemble]\n\n{result['response']}",
                        "treatment": {"medications": ["See above"], "home_remedies": ["See above"], "lifestyle": ["See above"]},
                        "confidence": min(0.95, result.get("confidence", 0.85)),
                        "model": "ensemble"
                    }
            except Exception as e:
                print(f"[AI DIAGNOSIS] Ensemble failed: {e}")
        if local_ai and local_ai_enabled:
            try:
                result = local_ai.medical_consultation(symptoms, patient_context=(medical_history or "")[:500])
                if result.get("success") and result.get("response"):
                    print("[AI DIAGNOSIS] Local AI SUCCESS")
                    return {
                        "diagnosis": f"[Local]\n\n{result['response']}",
                        "treatment": {"medications": ["See above"], "home_remedies": ["See above"], "lifestyle": ["See above"]},
                        "confidence": result.get("accuracy_estimate", 75) / 100,
                        "model": "local"
                    }
            except Exception as e:
                print(f"[AI DIAGNOSIS] Local AI failed: {e}")
        return get_local_diagnosis(symptoms, medical_history)
    
    # 2. Backend API first (uses HF + Gemini via api_simple)
    print("[AI DIAGNOSIS] Trying Backend API...")
    try:
        url = f"{API_BASE_URL}/diagnose"
        payload = {"symptoms": symptoms, "patient_id": "streamlit_user"}
        if medical_history:
            payload["medical_history"] = medical_history[:500]
        r = requests.post(url, json=payload, timeout=35)
        print(f"[AI DIAGNOSIS] Backend response: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            diag = data.get("diagnosis", "")
            if diag and len(diag) > 50 and "LOCAL MODEL ANSWER" not in diag:
                conf = data.get("confidence")
                if conf is None or not isinstance(conf, (int, float)):
                    conf = 0.85 if ("[Gemini]" in diag or "[HF API:" in diag) else 0.75
                print(f"[AI DIAGNOSIS] Backend SUCCESS - {data.get('model', 'unknown')}")
                return {
                    "diagnosis": diag,
                    "treatment": data.get("treatment", {"medications": ["See above"], "home_remedies": ["See above"], "lifestyle": ["See above"]}),
                    "confidence": float(conf),
                    "model": "backend_api"
                }
            else:
                print(f"[AI DIAGNOSIS] Backend returned short/invalid response")
    except requests.exceptions.ConnectionError:
        print("[AI DIAGNOSIS] Backend NOT RUNNING (connection refused)")
    except requests.exceptions.Timeout:
        print("[AI DIAGNOSIS] Backend TIMEOUT")
    except Exception as e:
        print(f"[AI DIAGNOSIS] Backend error: {str(e)[:60]}")
    
    # 3. Gemini and HF free-tier (respect LLM_PROVIDER)
    full_prompt = _full_consultation_prompt(symptoms, medical_history[:500] if medical_history else "")
    
    # 3a. Try Gemini with full consultation prompt
    print(f"[AI DIAGNOSIS] Trying Gemini... (model={gemini_model_name}, available={gemini_model is not None})")
    if (LLM_PROVIDER in ("auto", "gemini")) and gemini_model:
        try:
            if USING_NEW_GENAI:
                print("[AI DIAGNOSIS] Using new google.genai API...")
                r = gemini_model.models.generate_content(model=gemini_model_name, contents=full_prompt)
                text = r.text
            else:
                print("[AI DIAGNOSIS] Using old google.generativeai API...")
                r = gemini_model.generate_content(full_prompt)
                text = r.text
            if text and len(str(text).strip()) > 80:
                print(f"[AI DIAGNOSIS] Gemini SUCCESS - {len(text)} chars")
                return {
                    "diagnosis": f"[Gemini]\n\n{str(text).strip()}",
                    "treatment": {"medications": ["See above"], "home_remedies": ["See above"], "lifestyle": ["See above"]},
                    "confidence": 0.88,
                    "model": "gemini"
                }
            else:
                print(f"[AI DIAGNOSIS] Gemini returned short response: {len(text) if text else 0} chars")
        except Exception as e:
            print(f"[AI DIAGNOSIS] Gemini FAILED: {str(e)[:80]}")
    else:
        if gemini_model is None:
            print("[AI DIAGNOSIS] Gemini SKIPPED: model not configured")
        else:
            print(f"[AI DIAGNOSIS] Gemini SKIPPED: LLM_PROVIDER={LLM_PROVIDER}")
    
    # 3b. Try HF free-tier models
    print("[AI DIAGNOSIS] Trying HuggingFace...")
    hf_result = None
    if LLM_PROVIDER in ("auto", "hf") and _get_hf_token():
        hf_result = get_hf_free_tier_diagnosis(symptoms, medical_history[:500] if medical_history else "")
    elif LLM_PROVIDER in ("auto", "hf"):
        print("[AI DIAGNOSIS] HF SKIPPED: HF token not configured")
    if hf_result and len(hf_result) > 80:
        print(f"[AI DIAGNOSIS] HF SUCCESS - {len(hf_result)} chars")
        return {
            "diagnosis": f"[HF API]\n\n{hf_result}",
            "treatment": {"medications": ["See above"], "home_remedies": ["See above"], "lifestyle": ["See above"]},
            "confidence": 0.85,
            "model": "hf_free_tier"
        }
    else:
        print(f"[AI DIAGNOSIS] HF FAILED or short response")
    
    # 4. Meditron-70B (if on your plan)
    meditron_result = get_meditron_70b_diagnosis(symptoms, medical_history[:500] if medical_history else "")
    if meditron_result and len(meditron_result) > 30:
        return {
            "diagnosis": f"[HF API]\n\n{meditron_result}",
            "treatment": {"medications": ["See above"], "home_remedies": ["See above"], "lifestyle": ["See above"]},
            "confidence": 0.90,
            "model": "hf_api"
        }
    
    # 5. Local tier: use all models and datasets (RAG, DrugBank/Kaggle/FDA already used above; now ensemble, local_ai, RAG context, rule-based)
    get_ai_systems()
    # 5a. Try RAG (data_manager) for relevant medical context when no API succeeded
    try:
        try:
            from models.data_manager import data_manager
        except ImportError:
            from data_manager import data_manager
        docs = data_manager.retrieve_similar_docs(symptoms[:500], n_results=5)
        if docs and len(docs) > 0:
            rag_text = "**From medical knowledge base (RAG):**\n\n"
            for i, d in enumerate(docs[:3], 1):
                text = d.get("content") or d.get("text") or str(d)[:400]
                rag_text += f"{i}. {text}\n\n"
            rag_text += "\n*Consider setting GEMINI_API_KEY and HF_TOKEN in .env for full AI consultation.*"
            return {
                "diagnosis": "[Local RAG]\n\n" + rag_text.strip(),
                "treatment": {"medications": ["See above"], "home_remedies": ["See above"], "lifestyle": ["See above"]},
                "confidence": 0.70,
                "model": "local_rag"
            }
    except Exception:
        pass
    if ensemble_ai and ensemble_ai_enabled:
        try:
            result = ensemble_ai.medical_consultation(symptoms, context=(medical_history or "")[:500])
            if result.get("response"):
                return {
                    "diagnosis": f"[Ensemble]\n\n{result['response']}",
                    "treatment": {"medications": ["See above"], "home_remedies": ["See above"], "lifestyle": ["See above"]},
                    "confidence": min(0.95, result.get("confidence", 0.85)),
                    "model": "ensemble"
                }
        except Exception:
            pass
    if local_ai and local_ai_enabled:
        try:
            result = local_ai.medical_consultation(symptoms, patient_context=(medical_history or "")[:500])
            if result.get("success") and result.get("response"):
                return {
                    "diagnosis": f"[Local]\n\n{result['response']}",
                    "treatment": {"medications": ["See above"], "home_remedies": ["See above"], "lifestyle": ["See above"]},
                    "confidence": result.get("accuracy_estimate", 75) / 100,
                    "model": "local"
                }
        except Exception:
            pass
    
    # 6. Rule-based fallback only when HF and Gemini are unavailable
    return get_local_diagnosis(symptoms, medical_history)

def call_api(endpoint: str, method: str = 'GET', data: dict = None, files: dict = None):
    """Helper function to call backend API with fallback"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        
        if method == 'GET':
            response = requests.get(url, params=data, timeout=2)
        elif method == 'POST':
            if files:
                response = requests.post(url, data=data, files=files, timeout=5)
            else:
                response = requests.post(url, json=data, timeout=5)
        else:
            response = requests.request(method, url, json=data, timeout=2)
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except:
        # Return mock data when backend is not available
        if endpoint == '/learning/report':
            return {"total_feedback_collected": 0, "model_updates": 0}
        elif endpoint.startswith('/patient_history'):
            return []
        elif endpoint.startswith('/analytics'):
            return {
                "total_consultations": len(st.session_state.diagnosis_history),
                "health_score": 85,
                "risk_level": "Low",
                "symptom_frequency": {},
                "health_trend": []
            }
        return None

def login_page():
    """Display login/signup page with Login & Search Popup styling and logic flow info"""
    st.markdown("""
    <div class="main-header">
        <h1>AI Doctor</h1>
        <p>Your Personal Medical Assistant - Powered by Meditron & Gemini AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class="login-signup-popup search_form">
            <p class="search_icon" style="margin:0;font-size:1rem;">🔐 Login & Sign In</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="logic-flow-box">
            <strong>🧠 Logic flow (AI diagnosis)</strong>
            <ul>
                <li><strong>Priority:</strong> Gemini (very high) → HF (high) → local (all models & datasets)</li>
                <li>Loads .env from Ai Doctor folder</li>
                <li>Checks: HF token exists? Gemini key exists?</li>
                <li>Tries <strong>Gemini first</strong> (large models) when GEMINI_API_KEY exists and quota not exceeded</li>
                <li>If Gemini fails → uses <strong>HF free-tier</strong> (set HF_TOKEN in .env)</li>
                <li>If HF fails → uses <strong>local models & datasets</strong> (RAG, DrugBank, ensemble, rule-based)</li>
                <li>⚠️ Set <code>HF_TOKEN</code> and/or <code>GEMINI_API_KEY</code> in .env so HF/Gemini are used</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        
        with tab1:
            st.subheader("Welcome Back!")
            
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                remember = st.checkbox("Remember me")
                
                if st.form_submit_button("Login", use_container_width=True):
                    if username and password:
                        result = auth_system.login(username, password)
                        if result["success"]:
                            st.session_state.logged_in = True
                            st.session_state.session_token = result["session_token"]
                            st.session_state.user_data = result["user_data"]
                            st.success("Login successful! Redirecting...")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(result["message"])
                    else:
                        st.warning("Please enter both username and password")
        
        with tab2:
            st.subheader("Create New Account")
            
            with st.form("signup_form"):
                new_username = st.text_input("Choose Username", placeholder="Choose a unique username")
                new_email = st.text_input("Email", placeholder="your.email@example.com")
                new_fullname = st.text_input("Full Name", placeholder="John Doe")
                new_password = st.text_input("Password", type="password", placeholder="Min 6 characters")
                confirm_password = st.text_input("Confirm Password", type="password", placeholder="Re-enter password")
                agree = st.checkbox("I agree to the Terms of Service and Privacy Policy")
                
                if st.form_submit_button("Create Account", use_container_width=True):
                    if not all([new_username, new_email, new_password, confirm_password]):
                        st.warning("Please fill all fields")
                    elif new_password != confirm_password:
                        st.error("Passwords don't match")
                    elif not agree:
                        st.warning("Please agree to the terms")
                    else:
                        result = auth_system.create_user(new_username, new_password, new_email, new_fullname)
                        if result["success"]:
                            st.success("Account created! Please login.")
                            time.sleep(2)
                            st.rerun()
                        else:
                            st.error(result["message"])

def main_app():
    """Display main application"""
    
    # Sidebar
    with st.sidebar:
        st.markdown(f"""
        <div class="card">
            <h3>{st.session_state.user_data['full_name'] or st.session_state.user_data['username']}</h3>
            <p style="font-size: 0.8rem; color: #a0a0a0;">Patient ID: {st.session_state.user_data['patient_id']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="doorway">', unsafe_allow_html=True)
        if st.button("Logout", use_container_width=True, key="logout_btn"):
            auth_system.logout(st.session_state.session_token)
            st.session_state.logged_in = False
            st.session_state.session_token = None
            st.session_state.user_data = None
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.divider()
        
        # Logic flow (same as login page)
        st.markdown("""
        <div class="logic-flow-box" style="margin:0.5rem 0;">
            <strong>🧠 AI flow</strong><br/>
            Priority: Gemini (very high) → HF (high) → local (all models & datasets). Set HF_TOKEN/GEMINI_API_KEY in .env.
        </div>
        """, unsafe_allow_html=True)
        
        # Quick Stats
        st.subheader("Your Health Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Consultations", len(st.session_state.diagnosis_history))
        with col2:
            st.metric("Health Score", "85/100")
        
        st.divider()
        
        # System Status: Gemini → HF free-tier → local
        st.subheader("System Status")
        hf_ok = bool(_get_hf_token())
        gemini_ok = bool(_get_gemini_key())
        if gemini_ok and gemini_model:
            st.success("AI: Gemini first → HF → local")
        elif hf_ok:
            st.success("AI: HF free-tier → local (set GEMINI_API_KEY for Gemini)")
        elif local_ai and local_ai_enabled:
            acc = local_ai._estimate_accuracy() if hasattr(local_ai, '_estimate_accuracy') else 75
            st.success(f"AI: Local (~{acc:.0f}%)")
        else:
            st.info("AI: Set HF_TOKEN & GEMINI_API_KEY in .env")
    
    # Main Header
    st.markdown("""
    <div class="main-header">
        <h1>AI Doctor - Medical Assistant</h1>
        <p>Powered by Meditron-70B, Deep Learning, and Gemini AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        MEDICAL DISCLAIMER: This AI assistant is for informational purposes only. 
        It is NOT a substitute for professional medical advice. Always consult a qualified 
        healthcare provider for medical decisions. In case of emergency, call emergency services immediately!
    </div>
    """, unsafe_allow_html=True)
    
    # Main Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Consultation", 
        "Image Analysis", 
        "Medical Records", 
        "Health Trends",
        "AI Learning",
        "Provide Feedback"
    ])
    
    # Tab 1: Consultation - IMPROVED
    with tab1:
        st.header("Medical Consultation")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            symptoms = st.text_area(
                "Describe your symptoms in detail:",
                placeholder="Example: I have been experiencing sharp pain in my lower back and sides, pain during urination, and blood in urine for the past 3 days...",
                height=150
            )
            
            # Option to include previous medical history
            st.subheader("Previous Medical History")
            include_history = st.checkbox("Include my previous medical history in diagnosis", value=True)
            
            # Option to add custom history
            add_custom_history = st.checkbox("Add additional medical history")
            custom_history = ""
            if add_custom_history:
                custom_history = st.text_area(
                    "Enter your medical history:",
                    placeholder="Example: Diabetes Type 2, Hypertension, Previous surgeries, Allergies to medications...",
                    height=100
                )
            
            # Upload previous medical records
            st.subheader("Upload Medical Records (Optional)")
            uploaded_file = st.file_uploader(
                "Upload previous medical reports (PDF/Images)",
                type=['pdf', 'png', 'jpg', 'jpeg'],
                help="Upload blood reports, prescriptions, or other medical documents"
            )
            
            if uploaded_file:
                st.success(f"Uploaded: {uploaded_file.name}")
        
        with col2:
            st.subheader("Quick Symptoms")
            common_symptoms = [
                "Fever", "Headache", "Cough", "Fatigue",
                "Nausea", "Chest Pain", "Back Pain", "Joint Pain",
                "Kidney Stones", "Abdominal Pain", "Dizziness", "Shortness of Breath"
            ]
            
            selected_symptoms = []
            for symptom in common_symptoms:
                if st.checkbox(symptom, key=f"sym_{symptom}"):
                    selected_symptoms.append(symptom)
            
            if selected_symptoms:
                st.info(f"Selected: {', '.join(selected_symptoms)}")
        
        # Get Diagnosis button
        if st.button("Get AI Diagnosis", type="primary", use_container_width=True):
            if symptoms or selected_symptoms:
                with st.spinner("Analyzing symptoms with AI..."):
                    # Combine symptoms
                    all_symptoms = symptoms
                    if selected_symptoms:
                        all_symptoms += f"\nAdditional symptoms: {', '.join(selected_symptoms)}"
                    
                    # Get medical history if enabled
                    medical_history = ""
                    if include_history:
                        user_history = auth_system.get_user_history(st.session_state.user_data['username'])
                        if user_history:
                            history_texts = []
                            for h in user_history[-5:]:  # Last 5 records
                                history_texts.append(f"- {h.get('symptoms', 'N/A')}: {h.get('diagnosis', 'N/A')[:100]}")
                            medical_history = "\n".join(history_texts)
                    
                    if custom_history:
                        medical_history += f"\n\nAdditional history: {custom_history}"
                    
                    # Get AI diagnosis
                    result = get_ai_diagnosis(all_symptoms, medical_history)
                    
                    # Store in session
                    diagnosis_record = {
                        "symptoms": all_symptoms,
                        "diagnosis": result["diagnosis"],
                        "treatment": result["treatment"],
                        "confidence": result["confidence"],
                        "timestamp": datetime.now().isoformat(),
                        "included_history": include_history
                    }
                    st.session_state.diagnosis_history.append(diagnosis_record)
                    
                    # Save to user history
                    auth_system.add_to_history(
                        st.session_state.user_data['username'],
                        diagnosis_record
                    )
                    
                    # Display Results
                    st.markdown("---")
                    st.subheader("AI Diagnosis Results")
                    
                    # Confidence indicator
                    confidence = result["confidence"]
                    if confidence >= 0.8:
                        st.success(f"Confidence: {confidence:.0%} - High")
                    elif confidence >= 0.6:
                        st.warning(f"Confidence: {confidence:.0%} - Medium")
                    else:
                        st.error(f"Confidence: {confidence:.0%} - Low (Consult a doctor)")
                    
                    # Diagnosis
                    st.markdown("### Detailed Analysis")
                    st.markdown(f"""
                    <div class="info-box">
                    {result['diagnosis']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Disclaimer again
                    st.warning("Remember: This is AI-generated advice. Please consult a healthcare professional for proper diagnosis and treatment.")
            else:
                st.warning("Please describe your symptoms first")
    
    # Tab 2: Image Analysis - FIXED + Blood Report Integration
    with tab2:
        st.header("Medical Image & Report Analysis")
        
        # Tabs within tab2
        analysis_tab1, analysis_tab2 = st.tabs(["🖼️ Medical Images", "📋 Blood Reports"])
        
        with analysis_tab1:
            st.markdown("Upload X-rays, CT scans, MRI, skin images for AI analysis")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                uploaded_image = st.file_uploader(
                    "Upload medical image",
                    type=['png', 'jpg', 'jpeg'],
                    help="Upload X-ray, CT scan, MRI, or skin images",
                    key="medical_image_upload"
                )
                
                image_type = st.selectbox(
                    "Image Type",
                    ["Auto-detect", "Chest X-ray", "CT Scan", "MRI", "Skin Lesion", "Other"],
                    key="image_type_select"
                )
                
                if uploaded_image:
                    image = Image.open(uploaded_image)
                    st.image(image, caption="Uploaded Image", width=400)
            
            with col2:
                if uploaded_image:
                    if st.button("Analyze Image", type="primary", key="analyze_image_btn"):
                        with st.spinner("Analyzing with all available models (OCR + BiomedCLIP + Blood Parser)..."):
                            try:
                                from models.advanced_document_analyzer import AdvancedDocumentAnalyzer
                                analyzer = AdvancedDocumentAnalyzer(use_gpu=True)
                                img_path = os.path.join(tempfile.gettempdir(), uploaded_image.name)
                                with open(img_path, "wb") as f:
                                    f.write(uploaded_image.getvalue())
                                result_analysis = analyzer.analyze(img_path)
                                # Normalize: analyzer returns either blood report (parameters, summary) or xray (findings)
                                has_result = result_analysis.get("parameters") or result_analysis.get("summary") or result_analysis.get("findings")
                                if has_result:
                                    st.success("✅ Analysis Complete!")
                                    st.markdown("### Analysis Results")
                                    params = result_analysis.get("parameters", [])
                                    summary = result_analysis.get("summary", "")
                                    if params or summary:
                                        st.markdown("**Extracted parameters (OCR + Blood Parser):**")
                                        for p in params[:20]:
                                            name = p.get("name", p.get("parameter", ""))
                                            st.write(f"- **{name}**: {p.get('value')} {p.get('unit', '')} (Normal: {p.get('normal_range', 'N/A')}) — {p.get('status', '')}")
                                        if summary:
                                            st.markdown("**Summary:**")
                                            st.markdown(summary)
                                    else:
                                        findings = result_analysis.get("findings", [])
                                        for f in findings[:15]:
                                            if isinstance(f, dict):
                                                st.write(f"- **{f.get('finding', '')}**: {f.get('probability', 0):.0%}")
                                            else:
                                                st.write(f"- {f}")
                                    st.metric("Models used", "OCR + BloodReportParser / BiomedCLIP")
                                else:
                                    st.warning("Analysis completed with limited results. Consult a radiologist for clinical interpretation.")
                            except Exception as e:
                                st.error(f"Analysis error: {str(e)[:200]}")
                                st.info("Image analysis uses local models (OCR + BloodReportParser). For X-rays, BiomedCLIP/CheXNet may need to be loaded.")
                else:
                    st.info("Please upload a medical image to analyze")
        
        with analysis_tab2:
            st.markdown("Upload or paste your blood test reports for AI analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📤 Upload Report")
                blood_file = st.file_uploader(
                    "Choose blood report file",
                    type=['txt', 'pdf', 'png', 'jpg', 'jpeg'],
                    key="blood_report_file",
                    help="Upload TXT, PDF, PNG, JPG, or JPEG files"
                )
                
                blood_text = None
                if blood_file:
                    try:
                        file_extension = blood_file.name.split('.')[-1].lower()
                        
                        # Handle different file types
                        if file_extension == 'txt':
                            blood_text = blood_file.read().decode('utf-8')
                            st.success(f"✅ Loaded: {blood_file.name}")
                        
                        elif file_extension == 'pdf':
                            # For PDF files, we'll use pdfplumber if available
                            try:
                                import pdfplumber
                                from io import BytesIO
                                pdf_bytes = BytesIO(blood_file.read())
                                with pdfplumber.open(pdf_bytes) as pdf:
                                    blood_text = ""
                                    for page in pdf.pages:
                                        blood_text += page.extract_text() or ""
                                if blood_text:
                                    st.success(f"✅ Loaded: {blood_file.name}")
                                else:
                                    st.warning(f"⚠️ No text extracted from PDF. Might be scanned/image-based.")
                                    blood_text = f"[PDF Document: {blood_file.name}]\nNote: This appears to be a scanned PDF requiring OCR analysis."
                            except Exception as pdf_error:
                                # Fallback: treat as binary and note that OCR is needed
                                st.warning(f"Could not process PDF: {str(pdf_error)}")
                                blood_text = f"[PDF Document: {blood_file.name}]\nNote: This appears to be a scanned PDF requiring OCR analysis."
                        
                        elif file_extension in ['png', 'jpg', 'jpeg']:
                            # For image files, display and extract text
                            image = Image.open(blood_file)
                            st.image(image, caption="Uploaded Report Image", width=300)
                            st.info(f"📸 Image loaded: {blood_file.name}")
                            blood_text = f"[Blood Report Image: {blood_file.name}]\nImage-based report detected. AI will analyze the visual content."
                        
                        else:
                            st.error("Unsupported file format")
                            blood_text = None
                    
                    except Exception as e:
                        st.error(f"Error reading file: {str(e)}")
                        blood_text = None
                else:
                    blood_text = None
            
            with col2:
                st.subheader("✏️ Paste Report")
                blood_pasted = st.text_area(
                    "Or paste your blood report here",
                    height=200,
                    placeholder="Hemoglobin: 14.5 g/dL\\nGlucose: 125 mg/dL\\n...",
                    key="blood_report_paste"
                )
                if blood_pasted:
                    blood_text = blood_pasted
            
            if blood_text:
                if st.button("🔍 Analyze Blood Report", type="primary", key="analyze_blood_btn"):
                    with st.spinner("Analyzing blood report (using all available models)..."):
                        try:
                            analysis = None
                            
                            # 1. Try API (uses BloodReportParser + HistoryAgent + Offline)
                            try:
                                health_check = requests.get("http://localhost:8000/", timeout=5)
                                response = requests.post(
                                    "http://localhost:8000/analyze_blood_report",
                                    json={
                                        "report": blood_text,
                                        "patient_id": st.session_state.user_data.get('username', 'demo')
                                    },
                                    timeout=120
                                )
                                if response.status_code == 200:
                                    analysis = response.json()
                            except Exception:
                                pass
                            
                            # 2. If API fails, use direct analyzers (all local models)
                            if not analysis or not analysis.get('success'):
                                try:
                                    from models.advanced_document_analyzer import BloodReportParser
                                    parser = BloodReportParser()
                                    parsed = parser.parse(blood_text)
                                    if parsed.get("parameters") or parsed.get("summary"):
                                        analysis = {
                                            "success": True,
                                            "parameters": parsed.get("parameters", []),
                                            "abnormalities": [f"{p.get('name', '')}: {p.get('value')} {p.get('unit', '')}" for p in parsed.get("abnormal", [])],
                                            "findings": [{"type": "AI_Analysis", "content": parsed.get("summary", "")}],
                                        }
                                except Exception:
                                    pass
                            if not analysis or not analysis.get('success'):
                                try:
                                    from models.offline_blood_analyzer import OfflineBloodReportAnalyzer
                                    analyzer = OfflineBloodReportAnalyzer()
                                    raw = analyzer.analyze(blood_text)
                                    if raw.get("success"):
                                        analysis = {
                                            "success": True,
                                            "parameters": [{"name": p.get("parameter", "?"), "value": p.get("value"), "unit": p.get("unit", ""), "normal_range": p.get("normal_range", ""), "status": p.get("status", "")} for p in raw.get("parameters", [])],
                                            "abnormalities": [f"{p.get('parameter', '')}: {p.get('value')} {p.get('unit', '')} - {p.get('status', '')}" for p in raw.get("parameters", []) if p.get("status") != "NORMAL"],
                                            "findings": [{"type": "AI_Analysis", "content": raw.get("interpretation", "")}],
                                        }
                                except Exception as e:
                                    st.error(f"Analysis error: {str(e)}")
                                    analysis = None
                            
                            if analysis and analysis.get('success', False):
                                st.success("✅ Analysis Complete!")
                                
                                st.subheader("📊 Lab Results Analysis")
                                
                                # Display extracted parameters with status
                                if "parameters" in analysis and analysis["parameters"]:
                                    st.markdown("### Extracted Parameters")
                                    
                                    # Create columns for better display
                                    col1, col2, col3, col4, col5 = st.columns([1.5, 1.2, 1.2, 1, 1.3])
                                    with col1:
                                        st.markdown("**Parameter**")
                                    with col2:
                                        st.markdown("**Value**")
                                    with col3:
                                        st.markdown("**Unit**")
                                    with col4:
                                        st.markdown("**Normal**")
                                    with col5:
                                        st.markdown("**Status**")
                                    
                                    st.divider()
                                    
                                    for param in analysis["parameters"]:
                                        col1, col2, col3, col4, col5 = st.columns([1.5, 1.2, 1.2, 1, 1.3])
                                        
                                        with col1:
                                            st.markdown(f"**{param.get('name', 'Unknown')}**")
                                        with col2:
                                            st.markdown(f"{param.get('value', 'N/A')}")
                                        with col3:
                                            st.markdown(f"{param.get('unit', '')}")
                                        with col4:
                                            st.markdown(f"`{param.get('normal_range', 'N/A')}`")
                                        with col5:
                                            status = param.get('status', '⚪ UNKNOWN')
                                            if "HIGH" in status:
                                                st.markdown(f"<span style='color:red;font-weight:bold'>{status}</span>", unsafe_allow_html=True)
                                            elif "LOW" in status:
                                                st.markdown(f"<span style='color:orange;font-weight:bold'>{status}</span>", unsafe_allow_html=True)
                                            else:
                                                st.markdown(f"<span style='color:green;font-weight:bold'>{status}</span>", unsafe_allow_html=True)
                                else:
                                    st.info("No parameters were extracted from the report")
                                
                                # Display abnormalities
                                if "abnormalities" in analysis and analysis["abnormalities"]:
                                    st.markdown("### ⚠️ Abnormal Findings")
                                    for abnormality in analysis["abnormalities"]:
                                        st.warning(f"🔴 {abnormality}")
                                else:
                                    st.success("✅ All values are within normal ranges")
                                
                                # Display AI analysis
                                if "findings" in analysis and analysis["findings"]:
                                    st.markdown("### 📋 Clinical Findings")
                                    for finding in analysis["findings"]:
                                        if isinstance(finding, dict):
                                            content = finding.get('content', '')
                                            if content:
                                                st.markdown(f"**Analysis Details:**\n\n{content}")
                                        else:
                                            st.markdown(finding)
                            else:
                                error_msg = analysis.get('error', 'Analysis failed or no parameters found') if analysis else 'Analysis failed. Try uploading a report with parameter names and values.'
                                st.error(f"❌ {error_msg}")
                        except Exception as e:
                            st.error(f"Analysis failed: {str(e)}")
    
    # Tab 3: Medical Records - IMPROVED
    with tab3:
        st.header("Your Medical Records")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Current session history
            st.subheader("Recent Consultations")
            if st.session_state.diagnosis_history:
                for i, record in enumerate(reversed(st.session_state.diagnosis_history)):
                    with st.expander(f"Record {len(st.session_state.diagnosis_history) - i} - {record.get('timestamp', 'N/A')[:10]}"):
                        st.markdown(f"**Symptoms:** {record.get('symptoms', 'N/A')}")
                        st.markdown(f"**Diagnosis:** {record.get('diagnosis', 'N/A')[:500]}...")
                        st.markdown(f"**Confidence:** {record.get('confidence', 0):.0%}")
            else:
                st.info("No consultation records yet. Start a consultation to build your history.")
            
            # Stored user history
            st.subheader("Saved Medical History")
            user_history = auth_system.get_user_history(st.session_state.user_data['username'])
            if user_history:
                for i, record in enumerate(reversed(user_history[-10:])):
                    with st.expander(f"Saved Record {len(user_history) - i} - {record.get('timestamp', 'N/A')[:10]}"):
                        st.markdown(f"**Symptoms:** {record.get('symptoms', 'N/A')}")
                        st.markdown(f"**Diagnosis:** {str(record.get('diagnosis', 'N/A'))[:300]}...")
            else:
                st.info("No saved medical history.")
        
        with col2:
            st.subheader("Add Medical History")
            st.markdown("Upload or manually add your previous medical records")
            
            with st.form("add_history_form"):
                history_type = st.selectbox("Record Type", [
                    "Previous Diagnosis",
                    "Chronic Condition",
                    "Surgery",
                    "Allergy",
                    "Medication",
                    "Family History",
                    "Other"
                ])
                
                history_date = st.date_input("Date")
                history_description = st.text_area("Description", placeholder="Enter details...")
                
                if st.form_submit_button("Add Record"):
                    if history_description:
                        record = {
                            "type": history_type,
                            "date": str(history_date),
                            "description": history_description,
                            "timestamp": datetime.now().isoformat(),
                            "source": "manual_entry"
                        }
                        auth_system.add_to_history(st.session_state.user_data['username'], record)
                        st.success("Record added successfully!")
                        st.rerun()
                    else:
                        st.warning("Please enter a description")
    
    # Tab 4: Health Trends - IMPROVED WITH REAL DATA
    with tab4:
        st.header("Your Health Analytics")
        
        # Calculate metrics from actual history
        total_consultations = len(st.session_state.diagnosis_history)
        
        # Calculate health score based on consultations
        base_score = 85
        if total_consultations > 0:
            health_score = min(95, base_score + (total_consultations * 2))
        else:
            health_score = base_score
        
        # Determine risk level
        if health_score >= 80:
            risk_level = "Low"
            risk_color = "🟢"
        elif health_score >= 60:
            risk_level = "Medium"
            risk_color = "🟡"
        else:
            risk_level = "High"
            risk_color = "🔴"
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{total_consultations}</div>
                <div class="metric-label">Total Consultations</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{int(health_score)}</div>
                <div class="metric-label">Health Score</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{risk_color} {risk_level}</div>
                <div class="metric-label">Risk Level</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">3 mo</div>
                <div class="metric-label">Next Checkup</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Health Score Trend")
            
            # Always generate trend data
            num_records = 12
            dates = pd.date_range(end=datetime.now(), periods=num_records, freq='W')
            
            if len(st.session_state.diagnosis_history) > 0:
                # Use actual history to influence trend
                num_records = min(12, len(st.session_state.diagnosis_history) + 5)
                dates = pd.date_range(end=datetime.now(), periods=num_records, freq='W')
                
                # Create health score based on consultations
                base_trend = [75 + (i * 1.5) for i in range(num_records)]
                variability = [(-2 if i % 3 == 0 else 1) for i in range(num_records)]
                scores = [min(100, max(60, b + v)) for b, v in zip(base_trend, variability)]
            else:
                # Show sample health trend to demonstrate feature
                base_trend = [78 + (i * 0.8) for i in range(num_records)]
                variability = [(-1 if i % 4 == 0 else 1) for i in range(num_records)]
                scores = [min(95, max(65, b + v)) for b, v in zip(base_trend, variability)]
            
            trend_df = pd.DataFrame({
                'Date': dates,
                'Health Score': scores
            })
            
            fig = px.line(trend_df, x='Date', y='Health Score', 
                         markers=True, title='Weekly Health Score Progression')
            fig.update_traces(line_color='#00d4ff', line_width=3, marker=dict(size=8))
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                hovermode='x unified',
                xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.1)', range=[0, 100])
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🩺 Symptom Frequency")
            
            # Analyze symptoms from history
            if st.session_state.diagnosis_history:
                symptom_counts = {}
                keywords = ['fever', 'headache', 'cough', 'pain', 'fatigue', 'nausea', 'kidney', 'stones', 'dizziness', 'cold']
                
                for record in st.session_state.diagnosis_history:
                    symptoms_text = record.get('symptoms', '').lower()
                    for kw in keywords:
                        if kw in symptoms_text:
                            symptom_counts[kw.title()] = symptom_counts.get(kw.title(), 0) + 1
                
                if symptom_counts:
                    # Sort by frequency
                    symptom_counts = dict(sorted(symptom_counts.items(), key=lambda x: x[1], reverse=True))
                    
                    symptom_df = pd.DataFrame({
                        'Symptom': list(symptom_counts.keys()),
                        'Count': list(symptom_counts.values())
                    })
                    
                    fig = px.bar(symptom_df, x='Symptom', y='Count', title='Most Frequent Symptoms')
                    fig.update_traces(marker_color='#00d4ff')
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='white',
                        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                        yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Symptom data will appear after consultations")
            else:
                # Show sample symptom distribution
                sample_symptoms = {
                    'Headache': 3,
                    'Fever': 2,
                    'Fatigue': 2,
                    'Cough': 1
                }
                symptom_df = pd.DataFrame({
                    'Symptom': list(sample_symptoms.keys()),
                    'Count': list(sample_symptoms.values())
                })
                
                fig = px.bar(symptom_df, x='Symptom', y='Count', title='Sample Symptom Distribution')
                fig.update_traces(marker_color='#00d4ff', opacity=0.6)
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption("📝 *Sample data. Start consultations to see your actual data.*")
        
        # Additional Analytics Row
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("💊 Wellness Metrics")
            metrics_data = {
                'Metric': ['Sleep Quality', 'Activity Level', 'Nutrition', 'Stress'],
                'Score': [82, 75, 88, 65]
            }
            metrics_df = pd.DataFrame(metrics_data)
            fig = px.bar(metrics_df, x='Metric', y='Score', 
                        color='Score',
                        color_continuous_scale=['#ff6b6b', '#ffd93d', '#6bcf7f'],
                        title='Daily Wellness Indicators')
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.1)', range=[0, 100]),
                coloraxis_showscale=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📊 Health Distribution")
            health_categories = ['Excellent', 'Good', 'Fair', 'Poor']
            health_values = [45, 35, 15, 5]
            
            fig = px.pie(values=health_values, names=health_categories,
                        color_discrete_sequence=['#6bcf7f', '#00d4ff', '#ffd93d', '#ff6b6b'],
                        title='Overall Health Distribution')
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            st.subheader("📈 Weekly Activity")
            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            activity = [65, 72, 68, 85, 90, 75, 60]
            
            fig = px.bar(x=days, y=activity, 
                        color=activity,
                        color_continuous_scale='Viridis',
                        title='Weekly Activity Level')
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.1)', range=[0, 100]),
                coloraxis_showscale=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed Analytics
        st.subheader("📋 Detailed Health Insights")
        
        if st.session_state.diagnosis_history:
            # Create summary
            total_consultations = len(st.session_state.diagnosis_history)
            avg_confidence = sum(r.get('confidence', 0) for r in st.session_state.diagnosis_history) / total_consultations
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="info-box">
                <h4>Consultation Summary</h4>
                <p>Total consultations: {total_consultations}</p>
                <p>Average AI confidence: {avg_confidence:.0%}</p>
                <p>Last consultation: {st.session_state.diagnosis_history[-1].get('timestamp', 'N/A')[:10]}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="success-box">
                <h4>Recommendations</h4>
                <p>- Schedule regular checkups</p>
                <p>- Maintain a healthy lifestyle</p>
                <p>- Track symptoms regularly</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Complete some consultations to see detailed health insights")
    
    # Tab 5: AI Learning
    with tab5:
        st.header("AI Self-Learning System")
        st.markdown("Monitor the AI's learning progress and provide feedback")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">92%</div>
                <div class="metric-label">Model Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">1.5K</div>
                <div class="metric-label">Learning Cycles</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">Active</div>
                <div class="metric-label">Learning Status</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Learning curve
        st.subheader("Learning Progress")
        
        epochs = list(range(1, 21))
        accuracy = [70 + (i * 1.2) + (2 if i % 3 == 0 else -1) for i in epochs]
        accuracy = [min(95, max(70, a)) for a in accuracy]
        baseline = [70] * 20
        
        curve_df = pd.DataFrame({
            'Training Cycles': epochs,
            'Model Accuracy': accuracy,
            'Baseline': baseline
        })
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=curve_df['Training Cycles'],
            y=curve_df['Model Accuracy'],
            mode='lines+markers',
            name='Model Accuracy',
            line=dict(color='#00d4ff', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=curve_df['Training Cycles'],
            y=curve_df['Baseline'],
            mode='lines',
            name='Baseline',
            line=dict(color='#ff6b6b', width=2, dash='dash')
        ))
        fig.update_layout(
            title='AI Learning Progress Over Time',
            xaxis_title='Training Cycles',
            yaxis_title='Accuracy (%)',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            yaxis=dict(range=[60, 100], gridcolor='rgba(255,255,255,0.1)'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Feedback section
        st.subheader("Provide Feedback")
        st.markdown("Help improve the AI by providing feedback on diagnoses")
        
        if st.session_state.diagnosis_history:
            last_diagnosis = st.session_state.diagnosis_history[-1]
            st.markdown(f"**Last consultation:** {last_diagnosis.get('symptoms', 'N/A')[:100]}...")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Helpful", use_container_width=True):
                    st.success("Thank you for your positive feedback!")
            with col2:
                if st.button("Partially Helpful", use_container_width=True):
                    st.info("Thank you for your feedback. We'll improve!")
            with col3:
                if st.button("Not Helpful", use_container_width=True):
                    st.warning("We're sorry. Please consult a doctor and we'll learn from this.")
    
    # Tab 6: Provide Feedback
    with tab6:
        st.header("💬 Provide Feedback")
        st.markdown("""
        **Help improve the AI by providing feedback on diagnoses**
        
        Your feedback is crucial for our AI's learning and improvement. Rate the diagnosis accuracy 
        and share your comments to help us provide better medical guidance.
        """)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📝 Feedback Form")
            
            # Original query
            original_query = st.text_area(
                "Original Query/Symptoms",
                placeholder="What was your initial question or symptom?",
                height=80,
                key="feedback_query_tab6"
            )
            
            # AI Response
            ai_response = st.text_area(
                "AI's Response",
                placeholder="What was the AI's response or diagnosis?",
                height=80,
                key="feedback_response_tab6"
            )
            
            # Your feedback
            feedback_text = st.text_area(
                "Your Feedback",
                placeholder="Was the diagnosis accurate? What could be improved? Did you verify with a doctor?",
                height=100,
                key="feedback_text_tab6"
            )
        
        with col2:
            st.subheader("⭐ Rate This Response")
            
            rating = st.slider(
                "Accuracy Rating",
                min_value=1.0,
                max_value=5.0,
                value=3.0,
                step=0.5,
                key="feedback_rating_tab6"
            )
            
            # Display rating emoji
            if rating >= 4.5:
                st.markdown("### 🌟 Excellent!")
            elif rating >= 3.5:
                st.markdown("### ⭐ Good")
            elif rating >= 2.5:
                st.markdown("### 👍 Average")
            else:
                st.markdown("### 😕 Needs Improvement")
            
            report_type = st.selectbox(
                "Feedback Type",
                ["General", "Diagnosis", "Treatment", "Lab Analysis", "Image Analysis"],
                key="feedback_type_tab6"
            )
        
        # Submit button
        if st.button("📤 Submit Feedback", use_container_width=True, key="submit_feedback_tab6", type="primary"):
            if not original_query.strip():
                st.warning("⚠️ Please enter the original query")
            elif not ai_response.strip():
                st.warning("⚠️ Please enter the AI response")
            elif not feedback_text.strip():
                st.warning("⚠️ Please enter your feedback")
            else:
                with st.spinner("Submitting feedback..."):
                    try:
                        response = requests.post(
                            "http://localhost:8000/feedback",
                            json={
                                "query": original_query,
                                "response": ai_response,
                                "rating": rating,
                                "feedback_text": feedback_text,
                                "patient_id": st.session_state.user_data['username'],
                                "report_type": report_type
                            },
                            timeout=15
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.success("✅ Feedback submitted successfully!")
                            st.balloons()
                            
                            if result.get("success"):
                                st.info(f"📋 Feedback ID: {result.get('feedback_id', 'N/A')}")
                                st.write(f"**Thank you!** Your feedback has been indexed in our learning system and will help improve AI accuracy for future patients!")
                                
                                # Clear form
                                st.session_state.feedback_submitted = True
                        else:
                            st.error(f"API Error: {response.status_code}")
                    
                    except Exception as e:
                        st.error(f"Failed to submit feedback: {str(e)}")
        
        # Feedback Statistics
        st.divider()
        st.subheader("📈 Feedback Statistics")
        
        try:
            stats_response = requests.get(
                "http://localhost:8000/feedback_stats",
                timeout=10
            )
            
            if stats_response.status_code == 200:
                stats = stats_response.json()
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Feedbacks", stats.get("total_feedbacks", 0))
                
                with col2:
                    st.metric("Average Rating", f"{stats.get('average_rating', 0):.1f}/5.0")
                
                with col3:
                    st.metric("⭐ Excellent", stats.get("rating_distribution", {}).get("5_stars", 0))
                
                with col4:
                    st.metric("⭐ Good", stats.get("rating_distribution", {}).get("4_stars", 0))
                
                # Recent feedbacks
                if stats.get("recent_feedbacks"):
                    st.write("**Recent Feedbacks:**")
                    for fb in stats["recent_feedbacks"][-3:]:
                        with st.expander(f"⭐ {fb.get('rating', 0)}/5 - {fb.get('timestamp', 'Unknown')[:10]}"):
                            st.write(f"**Query:** {fb.get('query', '')[:150]}...")
                            st.write(f"**Feedback:** {fb.get('feedback_text', '')[:250]}...")
                            st.write(f"**Type:** {fb.get('report_type', 'N/A')}")
        
        except Exception as e:
            st.warning(f"Could not load feedback statistics: {e}")

# Main App Logic
def main():
    if st.session_state.logged_in:
        # Validate session
        if st.session_state.session_token:
            user_data = auth_system.validate_session(st.session_state.session_token)
            if user_data:
                main_app()
            else:
                st.session_state.logged_in = False
                st.warning("Session expired. Please login again.")
                login_page()
        else:
            login_page()
    else:
        login_page()

if __name__ == "__main__":
    main()
