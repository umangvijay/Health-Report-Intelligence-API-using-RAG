"""
All-in-One launcher for AI Doctor
Runs both backend and frontend in the same process
"""

import os
import sys
import threading
import time
import webbrowser
import subprocess
import signal
from pathlib import Path
from dotenv import load_dotenv

# Fix Windows encoding issues
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Set up paths: use absolute path and set cwd so imports and relative paths work
current_dir = Path(__file__).resolve().parent
os.chdir(current_dir)
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / "models"))  # so "from user_auth" finds models/user_auth.py

# Load environment
env_file = current_dir / '.env'
if env_file.exists():
    load_dotenv(env_file, override=False)

def _get_first_env(names):
    for name in names:
        val = os.getenv(name)
        if val is not None and str(val).strip():
            return str(val).strip().strip('"').strip("'")
    return ""

def _is_placeholder(value: str) -> bool:
    if not value:
        return True
    upper = value.upper()
    return any(tag in upper for tag in (
        "YOUR_HF_TOKEN",
        "YOUR_NEW_HF_TOKEN_HERE",
        "YOUR_GEMINI",
        "YOUR_NEW_GEMINI_KEY_HERE",
        "PLACEHOLDER",
        "INSERT_",
        "ADD_YOUR",
        "PUT_YOUR",
        "ENTER_YOUR",
        "REPLACE_",
        "<YOUR",
        "CHANGEME",
        "TODO",
    ))

def _kill_port(port: int):
    """Kill any process using the given port (Windows only)."""
    if sys.platform != 'win32':
        return
    try:
        result = subprocess.run(
            ["netstat", "-ano", "-p", "TCP"],
            capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.splitlines():
            if f":{port}" in line and "LISTENING" in line:
                parts = line.split()
                pid = int(parts[-1])
                if pid > 0:
                    try:
                        os.kill(pid, signal.SIGTERM)
                        print(f"  [cleanup] Killed old process on port {port} (PID {pid})")
                    except (OSError, ProcessLookupError):
                        pass
    except Exception:
        pass

print("""
============================================================
      AI DOCTOR - Advanced Medical AI System
      Uses LOCAL + HuggingFace Models
============================================================
""")

# ============================================================
# STEP 0: Kill old processes on ports 8000 and 8502
# ============================================================
print("Cleaning up old processes...")
_kill_port(8000)
_kill_port(8502)
time.sleep(1)

# Check environment
print("Checking environment...")

hf_token = _get_first_env([
    "HF_TOKEN",
    "HUGGINGFACE_TOKEN",
    "HUGGINGFACEHUB_API_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HF_API_KEY",
])
gemini_key = _get_first_env([
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GOOGLE_GENAI_API_KEY",
])

# Check token validity
if hf_token and not _is_placeholder(hf_token):
    print(f"[OK] HuggingFace token found ({hf_token[:15]}...)")
    print("     Used for: Downloading models + Inference API (if permitted)")
else:
    print("[!] HuggingFace token not set or placeholder")
    print("     Get token at: https://huggingface.co/settings/tokens")
    print("     Add permissions: Read + Inference API")

if gemini_key and not _is_placeholder(gemini_key):
    print("[OK] Gemini API key found (optional backup)")
else:
    print("[i] Gemini API key not set (optional - system works without it)")

print("\n" + "="*60)
print("MODELS - EQUAL PRIORITY: LOCAL + HF API")
print("="*60)
print("""
LOCAL MODELS (50% weight - run on your computer):
  [+] BioGPT (25%) - Primary text generation
  [+] ClinicalBERT (15%) - Clinical understanding
  [+] BioBERT (12%) - Medical NER
  [+] SciBERT (10%) - Scientific text
  [+] PubMedBERT (15%) - PubMed knowledge
  [+] BioLinkBERT (15%) - Biomedical relations
  [+] BiomedCLIP (8%) - Medical images

HF INFERENCE API (45% weight - runs on HuggingFace servers):
  [*] Meditron-70B (25%) - 70B medical specialist
  [*] Other HF models (20%) - Ensemble boost

OPTIONAL (5% weight):
  [*] Gemini API - Google backup

DATASETS (cached locally):
  [+] MedQA, MedMCQA, DrugBank vocabulary
""")
print("="*60)

# Create demo account
print("\nSetting up demo account...")
try:
    try:
        from user_auth import UserAuthSystem  # models/ on path
    except ImportError:
        from models.user_auth import UserAuthSystem
    auth = UserAuthSystem()
    result = auth.create_user("demo", "demo123", "demo@example.com", "Demo User")
    if result["success"]:
        print("[OK] Demo account created")
    else:
        print("[i] Demo account exists")
except Exception as e:
    print(f"[!] Auth system issue: {e}")

# ============================================================
# STEP 1: Start backend via subprocess (uvicorn)
# ============================================================
print("\nStarting backend API...")
backend_process = None
try:
    backend_env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(current_dir), str(current_dir / "models")])}
    # Ensure child gets .env vars so HF_TOKEN and GEMINI_API_KEY are always passed
    if env_file.exists():
        try:
            from dotenv import dotenv_values
            for k, v in (dotenv_values(env_file) or {}).items():
                current_val = str(backend_env.get(k, "")).strip()
                if v is not None and k and str(v).strip() and not current_val:
                    backend_env[k] = str(v).strip()
        except Exception:
            pass
    backend_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api_simple:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd=str(current_dir),
        env=backend_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )
    print("[OK] Backend process started (uvicorn api_simple:app)")
except Exception as e:
    print(f"[!] Could not start backend: {e}")
    import traceback
    traceback.print_exc()

# Wait for backend to start (up to 30 seconds)
print("Waiting for backend...")
import requests
backend_ok = False
for i in range(30):
    time.sleep(1)
    if backend_process and backend_process.poll() is not None:
        err = (backend_process.stderr and backend_process.stderr.read()) or b""
        if err:
            print("[!] Backend exited. Stderr:", err.decode("utf-8", errors="replace")[:500])
        break
    try:
        r = requests.get("http://localhost:8000/", timeout=2)
        if r.status_code == 200:
            print("[OK] Backend is running!")
            backend_ok = True
            break
    except Exception:
        pass
if not backend_ok and backend_process and backend_process.stderr:
    try:
        err = backend_process.stderr.read()
        if err:
            print("[!] Backend stderr:", err.decode("utf-8", errors="replace")[:800])
    except Exception:
        pass
if not backend_ok:
    print("[!] Backend may not be responding. Start manually: python api_simple.py")

# ============================================================
# STEP 2: Start frontend (Streamlit)
# ============================================================
print("\nStarting frontend...")

# Determine which app to use - use FULL PATH
if (current_dir / 'app_fixed.py').exists():
    app_file = str(current_dir / "app_fixed.py")
    print(f"Using app_fixed.py (with login)")
else:
    app_file = str(current_dir / "app.py")
    print(f"Using app.py")

print(f"App path: {app_file}")

# Open browser
print("\nOpening browser...")
time.sleep(2)
webbrowser.open("http://localhost:8502")

print("\n" + "="*60)
print("[OK] AI DOCTOR IS STARTING!")
print("="*60)
print("""
Access at: http://localhost:8502
Login: demo / demo123

ACTIVE MODELS (Equal Priority):
   * LOCAL: BioGPT, ClinicalBERT, BioBERT, SciBERT, PubMedBERT
   * HF API: Meditron-70B (FREE on HuggingFace servers)
   * Anti-hallucination systems
   * RLHF continuous learning

Medical Disclaimer:
   For educational purposes only.
   Not a substitute for professional medical advice.

Press Ctrl+C to stop
""")

print("="*60 + "\n")

# Change to correct directory first
os.chdir(current_dir)

# Run Streamlit in main thread
import streamlit.web.cli as stcli
sys.argv = ["streamlit", "run", app_file, 
            "--server.port=8502",
            "--server.address=localhost",
            "--server.headless=true",
            "--server.fileWatcherType=none"]

try:
    stcli.main()
except KeyboardInterrupt:
    print("\n\n[OK] Stopped AI Doctor")
except Exception as e:
    print(f"\n[X] Error: {e}")
    print("\nTry running components separately:")
    print(f'Terminal 1: cd "{current_dir}" && python api_simple.py')
    print(f'Terminal 2: cd "{current_dir}" && streamlit run app_fixed.py --server.port 8502')
