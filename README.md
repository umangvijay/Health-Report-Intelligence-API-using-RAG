# Health Report Intelligence API

> **ML Engineer Technical Assignment** — AI-powered RAG pipeline to upload, process, retrieve, and explain health report data.

---

## 1. Project Overview

A **FastAPI backend** that allows users to upload health reports (PDF/TXT), extracts text and structured health parameters, stores content in a vector database for semantic retrieval, and answers questions about the report using a Retrieval-Augmented Generation (RAG) pipeline — all while maintaining strict medical safety guardrails.

The system also includes an advanced AI Doctor module with ensemble models (Meditron, Mistral, BioGPT), image analysis, drug lookup, and RLHF training for continuous learning.

---

## 2. Problem Statement

Patients often receive health/blood test reports with medical terminology they don't understand. This API:

1. Accepts health report uploads (PDF or plain text).
2. Extracts structured parameters (e.g., Hemoglobin: 13.5 g/dL — Normal).
3. Answers natural-language questions grounded in the uploaded report.
4. Returns source chunks so the user can verify what the answer is based on.
5. **Never** diagnoses, prescribes medicine, or suggests treatment.

---

## 3. Tech Stack

| Layer | Technology | Reason |
|-------|-----------|--------|
| Language | **Python 3.10+** | Mandatory per assignment |
| Backend | **FastAPI** | Async, auto-docs, Pydantic validation |
| PDF Parsing | **pdfplumber** / PyPDF2 / PyMuPDF | Triple fallback for maximum compatibility |
| Embeddings | **sentence-transformers** (`all-MiniLM-L6-v2`) | Lightweight, good quality, runs on CPU |
| Vector DB | **ChromaDB** | Easy setup, metadata filtering, cosine similarity |
| LLM | **Gemini 1.5 Flash** / HuggingFace Inference API / Local models | Tiered fallback: Gemini → HF → Local |
| Parameter Extraction | Regex-based (30+ patterns) | Deterministic, no API dependency |
| Optional | Docker, Streamlit UI, OCR (pytesseract), SQLite | Bonus features |

---

## 4. Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                  HEALTH REPORT INTELLIGENCE API              │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│   POST /upload-report                                        │
│   ┌──────────┐   ┌───────────┐   ┌──────────┐   ┌────────┐ │
│   │ PDF/TXT  │──▸│  Extract  │──▸│  Chunk   │──▸│ Embed  │ │
│   │ Upload   │   │  Text     │   │  (500c/  │   │ Store  │ │
│   │          │   │ (3 libs)  │   │  100 ovl)│   │ Chroma │ │
│   └──────────┘   └───────────┘   └──────────┘   └────────┘ │
│         │                                            │       │
│         ▼                                            │       │
│   ┌──────────────┐    ┌─────────────────────────┐    │       │
│   │  Parameter   │    │  ChromaDB Collection    │◂───┘       │
│   │  Extraction  │    │  (report_chunks)        │            │
│   │  (30+ regex) │    │  - Cosine similarity    │            │
│   └──────────────┘    │  - Metadata filtering   │            │
│                       └─────────────────────────┘            │
│                                   │                          │
│   POST /ask-report                │                          │
│   ┌──────────┐   ┌───────────┐   │   ┌──────────────────┐  │
│   │ Question │──▸│ Retrieve  │◂──┘──▸│  LLM (Safe      │  │
│   │ + Report │   │ Top-K     │       │  Prompt - NO     │  │
│   │   ID     │   │ Chunks    │       │  diagnosis)      │  │
│   └──────────┘   └───────────┘       └──────────────────┘  │
│                                             │                │
│                                             ▼                │
│                                    ┌────────────────┐        │
│                                    │ Answer +       │        │
│                                    │ Sources +      │        │
│                                    │ Disclaimer     │        │
│                                    └────────────────┘        │
└──────────────────────────────────────────────────────────────┘
```

---

## 5. Setup Instructions

### Prerequisites
- Python 3.10 or higher
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/ai-doctor.git
cd ai-doctor

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Linux/Mac
# or
venv\Scripts\activate      # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and add your API keys

# 5. Run the server
python api_simple.py
```

Server starts at: `http://localhost:8000`  
API docs at: `http://localhost:8000/docs`

### Docker

```bash
docker-compose up --build
```

---

## 6. Environment Variable Setup

Copy `.env.example` to `.env` and configure:

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | Recommended | HuggingFace token ([get here](https://huggingface.co/settings/tokens)) |
| `GEMINI_API_KEY` | Optional | Google Gemini key ([get here](https://aistudio.google.com/apikey)) |
| `OPEN_FDA_API_KEY` | Optional | OpenFDA drug data key |
| `LLM_PROVIDER` | Optional | `auto` (default) / `gemini` / `hf` / `local` |

The system works in three tiers:
- **Tier 1:** Gemini + HuggingFace (best quality)
- **Tier 2:** HuggingFace only
- **Tier 3:** Local models only (always works, no API keys needed)

---

## 7. API Documentation

### 7.1 Health Check
```
GET /health
```
**Response:**
```json
{"status": "ok"}
```

### 7.2 Upload Report
```
POST /upload-report
Content-Type: multipart/form-data
```
**Request:** Form-data with `file` field (PDF or TXT).

```bash
curl -X POST http://localhost:8000/upload-report \
  -F "file=@sample_report.pdf"
```

**Response:**
```json
{
  "status": "success",
  "report_id": "report_a1b2c3d4e5f6",
  "filename": "sample_report.pdf",
  "total_chunks": 15,
  "extracted_parameters_count": 12
}
```

**Error responses:**
```json
{"status": "error", "message": "Unsupported file type. Only PDF and TXT files are allowed."}
{"status": "error", "message": "Empty file."}
{"status": "error", "message": "PDF extraction failed: ..."}
```

### 7.3 Ask Report
```
POST /ask-report
Content-Type: application/json
```
**Request:**
```json
{
  "report_id": "report_a1b2c3d4e5f6",
  "question": "Summarize my report in simple words."
}
```

```bash
curl -X POST http://localhost:8000/ask-report \
  -H "Content-Type: application/json" \
  -d '{"report_id": "report_a1b2c3d4e5f6", "question": "Summarize my report in simple words."}'
```

**Response:**
```json
{
  "answer": "Your report contains blood sugar, cholesterol, and thyroid-related values. Based on the uploaded report, some values appear to be outside the reference range...",
  "sources": [
    {
      "chunk_id": "chunk_002",
      "page_number": 1,
      "text": "Relevant report text used for the answer."
    }
  ],
  "disclaimer": "This is an AI-generated explanation based on the uploaded report and should not be treated as medical advice. Please consult a qualified doctor for diagnosis or treatment."
}
```

**Diagnosis guard response:**
```json
{
  "answer": "I cannot provide a medical diagnosis or treatment plan. I can only explain the uploaded report content in simple terms. Please consult a qualified doctor for medical advice.",
  "sources": [],
  "disclaimer": "This is an AI-generated explanation..."
}
```

### 7.4 Extracted Parameters
```
GET /reports/{report_id}/parameters
```

```bash
curl http://localhost:8000/reports/report_a1b2c3d4e5f6/parameters
```

**Response:**
```json
{
  "report_id": "report_a1b2c3d4e5f6",
  "parameters": [
    {
      "parameter": "Total Cholesterol",
      "value": "245",
      "unit": "mg/dL",
      "reference_range": "< 200.0",
      "status": "high"
    },
    {
      "parameter": "Hemoglobin",
      "value": "13.5",
      "unit": "g/dL",
      "reference_range": "13.0 - 17.0",
      "status": "normal"
    }
  ]
}
```

---

## 8. Sample API Requests and Responses

See section 9 (Sample Questions) for full end-to-end examples with evaluation.

---

## 9. Chunking Strategy

- **Method:** Fixed-size overlapping chunks
- **Chunk size:** 500 characters
- **Overlap:** 100 characters
- **Page tracking:** Each chunk carries its source `page_number`
- **Rationale:** Health reports are typically 1–5 pages. 500-char chunks ensure each chunk captures ~2–3 complete lab results with their reference ranges, providing enough context for the LLM without exceeding token limits. The 100-char overlap prevents values from being split across chunk boundaries.

---

## 10. Embedding Model Choice

- **Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Dimension:** 384
- **Why:** Lightweight (~80MB), fast on CPU, good semantic quality for short medical text. It outperforms larger models on short passages typical of lab reports.
- **Alternative tried:** `all-mpnet-base-v2` (768-dim) — better quality but 2x slower; overkill for this use case.

---

## 11. Vector Database Choice

- **Database:** ChromaDB
- **Why:**
  1. Zero-config — no external server needed (runs in-process)
  2. Built-in metadata filtering (`where={"report_id": "..."}`) — perfect for per-report retrieval
  3. Cosine similarity search out of the box
  4. Python-native with good FastAPI compatibility
- **Alternative considered:** FAISS — faster for large-scale, but no built-in metadata filtering.

---

## 12. Health Parameter Extraction Approach

- **Method:** Deterministic regex patterns (30+ patterns covering CBC, metabolic panel, lipid panel, thyroid, liver, vitamins, minerals)
- **Each pattern extracts:** parameter name, value, unit
- **Status determination:** Compared against hardcoded reference ranges → `high` / `low` / `normal`
- **Why regex over LLM?** Deterministic, reproducible, works offline, no API latency. LLM-based extraction is used as optional enhancement.
- **Covered parameters:** Hemoglobin, Hematocrit, RBC, WBC, Platelets, Glucose, HbA1c, Cholesterol (Total/LDL/HDL), Triglycerides, Creatinine, BUN, Urea, Sodium, Potassium, Calcium, TSH, T3, T4, ALT, AST, Bilirubin, ALP, Albumin, Vitamin D, B12, Iron, Ferritin, ESR, Uric Acid, MCV, MCH, MCHC

---

## 13. Medical Safety Handling

The system follows strict safety guidelines:

### What the system DOES NOT do:
- ❌ Give final diagnoses
- ❌ Prescribe medicines
- ❌ Suggest treatment plans
- ❌ Tell users to start/stop medication
- ❌ Make emergency medical decisions

### What the system CAN do:
- ✅ Explain values in simple language
- ✅ Identify values outside reference range
- ✅ Summarize report content
- ✅ Suggest consulting a qualified medical professional
- ✅ Note that interpretation depends on age, gender, medical history

### Implementation:
1. **Diagnosis guard:** `_is_diagnosis_seeking()` detects keywords like "diagnose", "prescribe", "treatment for" and returns a safe refusal.
2. **Safe system prompt:** The LLM prompt explicitly forbids diagnosis/prescription in its instructions.
3. **Mandatory disclaimer:** Every `/ask-report` response includes: *"This is an AI-generated explanation based on the uploaded report and should not be treated as medical advice. Please consult a qualified doctor for diagnosis or treatment."*

---

## 14. Sample Questions to Test

### Q1: Summarize this health report in simple language
- **Retrieved chunks:** chunk_000 (CBC section), chunk_001 (metabolic panel), chunk_002 (lipid panel)
- **Expected answer:** Overview of all parameters with flagged abnormals
- **Correctness:** ✅ Correct — accurately summarizes all sections
- **Limitation:** Very long reports may not be fully covered by 5 chunks

### Q2: Which values are outside the normal range?
- **Retrieved chunks:** All chunks containing out-of-range values
- **Expected answer:** Lists parameters with status "high" or "low"
- **Correctness:** ✅ Correct — regex extraction is deterministic
- **Limitation:** Only detects parameters matching known regex patterns

### Q3: Is my cholesterol normal according to this report?
- **Retrieved chunks:** Lipid panel chunk with Total Cholesterol, LDL, HDL
- **Expected answer:** "Your Total Cholesterol is 245 mg/dL, which is above the reference range of < 200 mg/dL."
- **Correctness:** ✅ Correct
- **Limitation:** Cannot interpret borderline values in clinical context

### Q4: Explain my thyroid results
- **Retrieved chunks:** Thyroid panel chunk (TSH, T3, T4)
- **Expected answer:** Explains each thyroid parameter and whether in range
- **Correctness:** ✅ Correct if thyroid values are in the report
- **Limitation:** If TSH/T3/T4 not in report, may respond with "insufficient data"

### Q5: What does my HbA1c value mean in this report?
- **Retrieved chunks:** Metabolic/diabetes section
- **Expected answer:** "HbA1c of 6.2% is slightly above the normal range of 4.0-5.7%, indicating elevated average blood sugar over 2-3 months"
- **Correctness:** ✅ Correct
- **Limitation:** Cannot correlate with fasting glucose or other diabetes markers

### Q6: Are there any vitamin deficiencies shown in this report?
- **Retrieved chunks:** Vitamin D, B12, Iron panel chunks
- **Expected answer:** Lists any vitamin/mineral values below reference range
- **Correctness:** ⚠️ Partially correct — only detects vitamins matching regex patterns
- **Limitation:** Uncommon vitamins (B6, folate, zinc) may not be extracted

### Q7: Which parameters are marked high?
- **Retrieved chunks:** All parameter chunks
- **Expected answer:** Lists all parameters with status "high"
- **Correctness:** ✅ Correct — uses deterministic extraction
- **Limitation:** Relies on regex matching; unusual formats may be missed

### Q8: Which parameters are marked low?
- **Retrieved chunks:** All parameter chunks
- **Expected answer:** Lists all parameters with status "low"
- **Correctness:** ✅ Correct
- **Limitation:** Same as Q7

---

## 15. Error Handling

| Error Case | HTTP Status | Response |
|---|---|---|
| Unsupported file type | 200 | `{"status": "error", "message": "Unsupported file type. Only PDF and TXT files are allowed."}` |
| Empty file | 200 | `{"status": "error", "message": "Empty file."}` |
| PDF extraction failure | 200 | `{"status": "error", "message": "PDF extraction failed: ..."}` |
| Invalid report ID | 200 | `{"status": "error", "message": "Invalid report ID: ... Report not found."}` |
| Empty question | 200 | `{"status": "error", "message": "question is required."}` |
| No relevant chunks | 200 | `{"answer": "No relevant content found...", "sources": [], "disclaimer": "..."}` |
| LLM API failure | 200 | Falls back through tiers (Gemini → HF → Local → rule-based) |
| Vector DB failure | 200 | Falls back to first N chunks from stored report |
| Not enough info | 200 | `{"status": "error", "message": "Report does not contain enough information."}` |
| Diagnosis-seeking | 200 | `{"answer": "I cannot provide a medical diagnosis...", "disclaimer": "..."}` |

---

## 16. Limitations

1. **Regex-based extraction** only covers ~30 common parameters; unusual test names or non-standard formatting may be missed.
2. **No OCR** on scanned/image-based PDFs by default (pytesseract available as optional).
3. **Chunk-boundary splitting** — a parameter value may be split across chunks despite overlap.
4. **Single-language** — optimized for English-language reports.
5. **No multi-report comparison** — each report is queried independently.
6. **LLM quality** depends on configured API keys; Tier 3 (local) produces shorter answers.

---

## 17. Future Improvements

1. **LLM-enhanced extraction** — use LLM as secondary extractor for parameters regex misses.
2. **OCR pipeline** — integrate pytesseract/easyocr for scanned report images.
3. **Multi-report comparison** — compare parameters across multiple uploads over time.
4. **Conversation memory** — follow-up questions with context from previous Q&A.
5. **Confidence scoring** — indicate how certain the answer is based on chunk relevance.
6. **Multi-language support** — Hindi, Spanish, etc.
7. **Cloud deployment** — Deploy on Render / Railway / EC2 with persistent storage.

---

## 18. Advanced Features (Bonus)

Beyond the core assignment, this project includes:

- **Docker support** — `Dockerfile` + `docker-compose.yml`
- **Streamlit UI** — Full frontend with login, chat, image upload
- **User authentication** — Login/signup system with `users_db`
- **Feedback & RLHF** — User feedback collection + reinforcement learning pipeline
- **Drug lookup** — 50+ medicines via DrugBank + OpenFDA integration
- **Multi-model ensemble** — Tiered routing through Gemini, HuggingFace (Mistral, BLOOM, Falcon), and local models (flan-t5, BioGPT)
- **GPU acceleration** — CUDA support with automatic device detection
- **Medical image analysis** — X-ray classification, BiomedCLIP integration
- **PHI sanitization** — Removes SSN, phone, email, MRN from logs

---

## Medical Disclaimer

This AI system is for **EDUCATIONAL AND INFORMATIONAL PURPOSES ONLY**.
- NOT a substitute for professional medical advice.
- Always consult qualified healthcare providers.
- Do not use for diagnosis or treatment decisions.
- If you have a medical emergency, call emergency services immediately.
