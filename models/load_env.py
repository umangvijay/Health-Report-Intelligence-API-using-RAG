"""
Load environment variables from .env file
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Get the directory of this script
current_dir = Path(__file__).parent

# Try multiple possible .env locations
env_paths = [
    current_dir / '.env',
    current_dir.parent / '.env',
    Path.cwd() / '.env'
]

# Load .env file
for env_path in env_paths:
    if env_path.exists():
        load_dotenv(env_path, override=True)
        print(f"✅ Loaded .env from: {env_path}")
        break
else:
    print("⚠️ No .env file found")

# Print loaded keys (masked)
if os.getenv('GEMINI_API_KEY'):
    key = os.getenv('GEMINI_API_KEY')
    print(f"✅ GEMINI_API_KEY loaded: {'*' * 20}{key[-4:]}")
else:
    print("❌ GEMINI_API_KEY not found")

if os.getenv('HF_TOKEN'):
    token = os.getenv('HF_TOKEN')
    print(f"✅ HF_TOKEN loaded: {'*' * 20}{token[-4:]}")
else:
    print("❌ HF_TOKEN not found")