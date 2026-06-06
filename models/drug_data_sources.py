"""
Drug data sources: Kaggle drug datasets (kagglehub) + open.fda.gov API.
- Kaggle: shaiksha19/drug-dataset, prathamtripathi/drug-classification
- FDA: https://open.fda.gov/apis/authentication/ (OPEN_FDA_API_KEY in .env)
"""
import os
from pathlib import Path
from typing import Optional, Dict, Any, List

# Optional: load .env so OPEN_FDA_API_KEY is available
try:
    from dotenv import load_dotenv
    _env = Path(__file__).resolve().parent.parent / ".env"
    if _env.exists():
        load_dotenv(_env, override=True)
except Exception:
    pass

OPEN_FDA_API_KEY = os.getenv("OPEN_FDA_API_KEY") or os.getenv("FDA_API_KEY")
FDA_BASE = "https://api.fda.gov"

# Common drug synonyms for FDA search (generic/brand)
DRUG_SYNONYMS = {
    "paracetamol": ["paracetamol", "acetaminophen"],
    "acetaminophen": ["acetaminophen", "paracetamol"],
    "crocin": ["acetaminophen", "paracetamol"],
    "dolo": ["acetaminophen", "paracetamol"],
    "ibuprofen": ["ibuprofen", "advil", "motrin"],
    "aspirin": ["aspirin", "acetylsalicylic acid"],
    "amoxicillin": ["amoxicillin"],
    "metformin": ["metformin"],
    "omeprazole": ["omeprazole"],
    "lisinopril": ["lisinopril"],
    "atorvastatin": ["atorvastatin", "lipitor"],
    "amlodipine": ["amlodipine"],
    "losartan": ["losartan"],
}


def extract_drug_terms_from_query(query: str) -> List[str]:
    """Extract likely drug terms from a user query and expand with synonyms."""
    q = (query or "").lower()
    terms: List[str] = []
    # Direct known match
    for key in DRUG_SYNONYMS.keys():
        if key in q:
            terms.extend(DRUG_SYNONYMS.get(key, [key]))
    if terms:
        return list(dict.fromkeys(terms))
    # Pattern-based extraction (fallback)
    import re
    patterns = [
        r"(?:use of|about|info on|information on|tell me about)\s+([a-z0-9\- ]{2,40})",
        r"(?:what is|what does)\s+([a-z0-9\- ]{2,40})",
    ]
    for pat in patterns:
        m = re.search(pat, q)
        if m:
            raw = m.group(1).strip()
            raw = re.sub(r"\b(tablet|tablets|uses|use|side effects|dosage|dose)\b", "", raw).strip()
            if raw:
                terms.append(raw)
                break
    return list(dict.fromkeys(terms))


def get_kaggle_drug_dataset_path() -> Optional[str]:
    """Download latest shaiksha19/drug-dataset via kagglehub; return path to dataset files."""
    try:
        import kagglehub
        path = kagglehub.dataset_download("shaiksha19/drug-dataset")
        print("Path to dataset files (drug-dataset):", path)
        return path
    except Exception as e:
        print("Kaggle drug-dataset download failed:", e)
        return None


def get_kaggle_drug_classification_path() -> Optional[str]:
    """Download latest prathamtripathi/drug-classification via kagglehub; return path to dataset files."""
    try:
        import kagglehub
        path = kagglehub.dataset_download("prathamtripathi/drug-classification")
        print("Path to dataset files (drug-classification):", path)
        return path
    except Exception as e:
        print("Kaggle drug-classification download failed:", e)
        return None


def download_all_drug_datasets() -> Dict[str, Optional[str]]:
    """Download both drug datasets; return dict with keys 'drug_dataset' and 'drug_classification'."""
    return {
        "drug_dataset": get_kaggle_drug_dataset_path(),
        "drug_classification": get_kaggle_drug_classification_path(),
    }


def fetch_fda_drug_info(drug_terms: List[str], api_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Fetch drug label info from open.fda.gov (drug/label.json).
    Uses OPEN_FDA_API_KEY from env if api_key not passed.
    Returns best matching result or None.
    """
    from urllib.parse import urlencode
    key = api_key or OPEN_FDA_API_KEY
    if not key or "YOUR" in str(key):
        return None
    terms = [t.strip().lower() for t in (drug_terms or []) if t and t.strip()]
    if not terms:
        return None
    fields = ["openfda.generic_name", "openfda.brand_name", "openfda.substance_name"]
    search_terms = []
    for term in terms:
        for field in fields:
            search_terms.append(f'{field}:"{term}"')
    search_query = " OR ".join(search_terms)
    params = {"api_key": key, "search": search_query, "limit": 5}
    url = f"{FDA_BASE}/drug/label.json?{urlencode(params)}"
    try:
        import urllib.request
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read().decode()
    except Exception:
        try:
            import requests
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            data = r.text
        except Exception:
            return None
    try:
        import json
        out = json.loads(data)
        results = out.get("results")
        if not results or len(results) == 0:
            return None
        # pick best match (generic/brand/substance contains a term)
        first = None
        for r in results:
            openfda = r.get("openfda") or {}
            names = []
            for k in ("generic_name", "brand_name", "substance_name"):
                vals = openfda.get(k) or []
                names.extend([v.lower() for v in vals if isinstance(v, str)])
            if any(term in names for term in terms):
                first = r
                break
        if first is None:
            return None
        openfda = first.get("openfda") or {}
        indications = first.get("indications_and_usage")
        if isinstance(indications, list):
            indications = indications[0] if indications else ""
        adverse = first.get("adverse_reactions")
        if isinstance(adverse, list):
            adverse = adverse[0] if adverse else ""
        dosage = first.get("dosage_and_administration")
        if isinstance(dosage, list):
            dosage = dosage[0] if dosage else ""
        warnings = first.get("warnings")
        if isinstance(warnings, list):
            warnings = warnings[0] if warnings else ""
        return {
            "brand_name": (openfda.get("brand_name") or [None])[0],
            "generic_name": (openfda.get("generic_name") or [None])[0],
            "substance_name": (openfda.get("substance_name") or [None])[0],
            "indications_and_usage": indications,
            "adverse_reactions": adverse,
            "dosage_and_administration": dosage,
            "warnings": warnings,
            "source": "open.fda.gov",
        }
    except Exception:
        return None


def get_fda_drug_text(drug_terms: List[str]) -> Optional[str]:
    """
    Get formatted drug info string from FDA for display.
    Returns None if no API key or no result.
    """
    info = fetch_fda_drug_info(drug_terms)
    if not info:
        return None
    name = info.get("generic_name") or info.get("brand_name") or info.get("substance_name") or (drug_terms[0] if drug_terms else "medicine")
    lines = [f"**{name}** (FDA)", ""]
    if info.get("indications_and_usage"):
        lines.append("**Uses:**")
        lines.append(info["indications_and_usage"][:1500])
        lines.append("")
    if info.get("dosage_and_administration"):
        lines.append("**Dosage:**")
        lines.append(info["dosage_and_administration"][:800])
        lines.append("")
    if info.get("adverse_reactions"):
        lines.append("**Adverse reactions:**")
        lines.append(info["adverse_reactions"][:800])
        lines.append("")
    if info.get("warnings"):
        lines.append("**Warnings:**")
        lines.append(info["warnings"][:800])
    return "\n".join(lines).strip()


def get_fda_drug_text_from_query(query: str) -> Optional[str]:
    """Extract drug terms from a query and return FDA text if found."""
    terms = extract_drug_terms_from_query(query)
    if not terms:
        return None
    return get_fda_drug_text(terms)
