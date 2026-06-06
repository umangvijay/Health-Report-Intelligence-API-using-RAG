"""
Unified Medical AI System
Uses ALL HuggingFace models for comprehensive medical analysis
Works OFFLINE without Gemini API - Uses local LLMs (Mistral-7B, BioGPT)
Integrates: SciBERT, BioBERT, ClinicalBERT, BioGPT, BiomedCLIP, Mistral-7B
"""

import os
import torch
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from PIL import Image
import json

logger = logging.getLogger(__name__)

# Get HuggingFace token
HF_TOKEN = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_TOKEN')


class UnifiedMedicalAI:
    """
    Unified system that uses ALL available HuggingFace models
    Primary LLM: BioGPT (lightweight) or Mistral-7B (powerful)
    Fallback: Gemini API (if available)
    """
    
    def __init__(self, use_gpu: bool = True, prefer_local: bool = True):
        """
        Initialize unified medical AI system
        
        Args:
            use_gpu: Use GPU if available
            prefer_local: Prefer local models over API calls
        """
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.prefer_local = prefer_local
        self.models = {}
        self.pipelines = {}
        
        logger.info(f"Initializing Unified Medical AI on {self.device}")
        logger.info(f"HuggingFace token: {'Found' if HF_TOKEN else 'Not found'}")
        
        # Model status tracking
        self.model_status = {
            "biogpt": False,
            "mistral_7b": False,
            "clinical_bert": False,
            "biobert": False,
            "scibert": False,
            "biomedclip": False,
            "gemini": False
        }
        
        # Initialize models
        self._init_all_models()
    
    def _init_all_models(self):
        """Initialize all available models"""
        
        # 1. BioGPT - Lightweight text generation (PRIMARY for offline)
        self._init_biogpt()
        
        # 2. Clinical-BERT - Clinical text understanding
        self._init_clinical_bert()
        
        # 3. BioBERT - Medical entity extraction
        self._init_biobert()
        
        # 4. SciBERT - Scientific text
        self._init_scibert()
        
        # 5. BiomedCLIP - Medical image understanding
        self._init_biomedclip()
        
        # 6. Mistral-7B - Powerful LLM (optional, requires more resources)
        # Only load if explicitly requested due to size
        
        # 7. Check Gemini availability
        self._check_gemini()
        
        self._print_status()
    
    def _init_biogpt(self):
        """Initialize BioGPT for medical text generation"""
        try:
            # Check sacremoses requirement first
            try:
                import sacremoses
            except ImportError:
                logger.warning("BioGPT requires sacremoses: pip install sacremoses")
                self.model_status["biogpt"] = False
                return
            
            from transformers import BioGptTokenizer, BioGptForCausalLM
            
            logger.info("Loading BioGPT...")
            model_id = "microsoft/biogpt"
            
            self.models["biogpt_tokenizer"] = BioGptTokenizer.from_pretrained(
                model_id, token=HF_TOKEN
            )
            self.models["biogpt"] = BioGptForCausalLM.from_pretrained(
                model_id, 
                token=HF_TOKEN,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            if self.device == "cuda":
                self.models["biogpt"] = self.models["biogpt"].to(self.device)
            
            self.models["biogpt"].eval()
            self.model_status["biogpt"] = True
            logger.info("✅ BioGPT loaded successfully")
            
        except Exception as e:
            logger.warning(f"Could not load BioGPT: {e}")
            self.model_status["biogpt"] = False
    
    def _init_clinical_bert(self):
        """Initialize Clinical-BERT for clinical text understanding"""
        try:
            from transformers import AutoTokenizer, AutoModel
            
            logger.info("Loading Clinical-BERT...")
            model_id = "medicalai/ClinicalBERT"
            
            self.models["clinical_bert_tokenizer"] = AutoTokenizer.from_pretrained(
                model_id, token=HF_TOKEN
            )
            self.models["clinical_bert"] = AutoModel.from_pretrained(
                model_id, token=HF_TOKEN
            )
            
            if self.device == "cuda":
                self.models["clinical_bert"] = self.models["clinical_bert"].to(self.device)
            
            self.model_status["clinical_bert"] = True
            logger.info("✅ Clinical-BERT loaded successfully")
            
        except Exception as e:
            logger.warning(f"Could not load Clinical-BERT: {e}")
            self.model_status["clinical_bert"] = False
    
    def _init_biobert(self):
        """Initialize BioBERT for NER"""
        try:
            from transformers import pipeline
            
            logger.info("Loading BioBERT...")
            self.pipelines["biobert_ner"] = pipeline(
                "ner",
                model="dmis-lab/biobert-base-cased-v1.2",
                tokenizer="dmis-lab/biobert-base-cased-v1.2",
                device=0 if self.device == "cuda" else -1,
                aggregation_strategy="simple",
                token=HF_TOKEN
            )
            
            self.model_status["biobert"] = True
            logger.info("✅ BioBERT NER loaded successfully")
            
        except Exception as e:
            logger.warning(f"Could not load BioBERT: {e}")
            self.model_status["biobert"] = False
    
    def _init_scibert(self):
        """Initialize SciBERT for scientific text"""
        try:
            from transformers import AutoTokenizer, AutoModel
            
            logger.info("Loading SciBERT...")
            model_id = "allenai/scibert_scivocab_cased"
            
            self.models["scibert_tokenizer"] = AutoTokenizer.from_pretrained(
                model_id, token=HF_TOKEN
            )
            self.models["scibert"] = AutoModel.from_pretrained(
                model_id, token=HF_TOKEN
            )
            
            if self.device == "cuda":
                self.models["scibert"] = self.models["scibert"].to(self.device)
            
            self.model_status["scibert"] = True
            logger.info("✅ SciBERT loaded successfully")
            
        except Exception as e:
            logger.warning(f"Could not load SciBERT: {e}")
            self.model_status["scibert"] = False
    
    def _init_biomedclip(self):
        """Initialize BiomedCLIP for medical image understanding"""
        try:
            # Try open_clip first (better compatibility)
            try:
                import open_clip
                logger.info("Loading BiomedCLIP via open_clip...")
                
                model, _, preprocess = open_clip.create_model_and_transforms(
                    'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
                )
                tokenizer = open_clip.get_tokenizer(
                    'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
                )
                
                self.models["biomedclip"] = model
                self.models["biomedclip_preprocess"] = preprocess
                self.models["biomedclip_tokenizer"] = tokenizer
                
                if self.device == "cuda":
                    self.models["biomedclip"] = self.models["biomedclip"].to(self.device)
                
                self.model_status["biomedclip"] = True
                logger.info("✅ BiomedCLIP loaded via open_clip")
                return
                
            except ImportError:
                logger.info("open_clip not installed, trying transformers...")
            
            # Fallback to transformers
            from transformers import AutoProcessor, AutoModel
            
            logger.info("Loading BiomedCLIP via transformers...")
            model_id = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            
            self.models["biomedclip_processor"] = AutoProcessor.from_pretrained(
                model_id, token=HF_TOKEN, trust_remote_code=True
            )
            self.models["biomedclip"] = AutoModel.from_pretrained(
                model_id, token=HF_TOKEN, trust_remote_code=True
            )
            
            if self.device == "cuda":
                self.models["biomedclip"] = self.models["biomedclip"].to(self.device)
            
            self.model_status["biomedclip"] = True
            logger.info("✅ BiomedCLIP loaded successfully")
            
        except Exception as e:
            logger.warning(f"Could not load BiomedCLIP: {e}")
            self.model_status["biomedclip"] = False
    
    def _check_gemini(self):
        """Check if Gemini API is available"""
        try:
            gemini_key = os.getenv('GEMINI_API_KEY')
            if gemini_key and 'your' not in gemini_key.lower():
                import google.generativeai as genai
                genai.configure(api_key=gemini_key)
                self.models["gemini"] = genai.GenerativeModel('models/gemini-2.0-flash')
                self.model_status["gemini"] = True
                logger.info("✅ Gemini API available (backup)")
            else:
                self.model_status["gemini"] = False
        except Exception as e:
            logger.warning(f"Gemini not available: {e}")
            self.model_status["gemini"] = False
    
    def load_mistral_7b(self):
        """Load Mistral-7B for powerful inference (call explicitly due to size)"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            
            logger.info("Loading Mistral-7B (this may take a while)...")
            model_id = "mistralai/Mistral-7B-Instruct-v0.3"
            
            self.models["mistral_tokenizer"] = AutoTokenizer.from_pretrained(
                model_id, token=HF_TOKEN
            )
            
            # Use 8-bit quantization to reduce memory
            if self.device == "cuda":
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                self.models["mistral"] = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    token=HF_TOKEN,
                    quantization_config=quantization_config,
                    device_map="auto"
                )
            else:
                self.models["mistral"] = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    token=HF_TOKEN,
                    torch_dtype=torch.float32
                )
            
            self.model_status["mistral_7b"] = True
            logger.info("✅ Mistral-7B loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Could not load Mistral-7B: {e}")
            self.model_status["mistral_7b"] = False
            return False
    
    def _print_status(self):
        """Print model loading status"""
        print("\n" + "="*60)
        print("UNIFIED MEDICAL AI - Model Status")
        print("="*60)
        for model, status in self.model_status.items():
            icon = "✅" if status else "❌"
            print(f"  {icon} {model}")
        print("="*60 + "\n")
    
    # ============= MAIN INFERENCE METHODS =============
    
    def generate_response(self, query: str, context: str = None) -> Dict[str, Any]:
        """
        Generate medical response using best available model
        Priority: BioGPT → Mistral → Gemini
        
        Args:
            query: User's medical question
            context: Additional context (patient history, etc.)
        
        Returns:
            Dict with response and metadata
        """
        # Build prompt
        if context:
            full_prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        else:
            full_prompt = f"Question: {query}\n\nAnswer:"
        
        # Try BioGPT first (lightweight, medical-specific)
        if self.model_status["biogpt"]:
            try:
                response = self._generate_biogpt(full_prompt)
                return {
                    "response": response,
                    "model": "BioGPT",
                    "offline": True,
                    "success": True
                }
            except Exception as e:
                logger.warning(f"BioGPT generation failed: {e}")
        
        # Try Mistral-7B (powerful, but larger)
        if self.model_status["mistral_7b"]:
            try:
                response = self._generate_mistral(full_prompt)
                return {
                    "response": response,
                    "model": "Mistral-7B",
                    "offline": True,
                    "success": True
                }
            except Exception as e:
                logger.warning(f"Mistral generation failed: {e}")
        
        # Fallback to Gemini API
        if self.model_status["gemini"] and not self.prefer_local:
            try:
                response = self._generate_gemini(full_prompt)
                return {
                    "response": response,
                    "model": "Gemini",
                    "offline": False,
                    "success": True
                }
            except Exception as e:
                logger.warning(f"Gemini generation failed: {e}")
        
        return {
            "response": "No model available for text generation. Please check model status.",
            "model": None,
            "offline": True,
            "success": False
        }
    
    def _generate_biogpt(self, prompt: str, max_length: int = 256) -> str:
        """Generate text using BioGPT"""
        tokenizer = self.models["biogpt_tokenizer"]
        model = self.models["biogpt"]
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from response
        if prompt in response:
            response = response.replace(prompt, "").strip()
        
        return response
    
    def _generate_mistral(self, prompt: str, max_length: int = 512) -> str:
        """Generate text using Mistral-7B"""
        tokenizer = self.models["mistral_tokenizer"]
        model = self.models["mistral"]
        
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
        
        if self.device == "cuda":
            inputs = inputs.to(self.device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "[/INST]" in response:
            response = response.split("[/INST]")[-1].strip()
        
        return response
    
    def _generate_gemini(self, prompt: str) -> str:
        """Generate text using Gemini API (fallback)"""
        model = self.models["gemini"]
        response = model.generate_content(prompt)
        return response.text
    
    def extract_medical_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract medical entities using BioBERT
        
        Args:
            text: Medical text to analyze
        
        Returns:
            Dict with categorized entities
        """
        if not self.model_status["biobert"]:
            return {"error": "BioBERT not loaded"}
        
        try:
            entities = self.pipelines["biobert_ner"](text)
            
            organized = {
                "drugs": [],
                "diseases": [],
                "symptoms": [],
                "procedures": [],
                "anatomy": [],
                "other": []
            }
            
            for entity in entities:
                word = entity.get("word", "").replace("##", "")
                label = entity.get("entity_group", "").lower()
                
                if any(x in label for x in ["drug", "medication", "chemical"]):
                    if word not in organized["drugs"]:
                        organized["drugs"].append(word)
                elif any(x in label for x in ["disease", "disorder", "condition"]):
                    if word not in organized["diseases"]:
                        organized["diseases"].append(word)
                elif any(x in label for x in ["symptom", "sign"]):
                    if word not in organized["symptoms"]:
                        organized["symptoms"].append(word)
                else:
                    if word not in organized["other"]:
                        organized["other"].append(word)
            
            return organized
            
        except Exception as e:
            return {"error": str(e)}
    
    def analyze_medical_image(self, image_path: str, text_query: str = None) -> Dict[str, Any]:
        """
        Analyze medical image using BiomedCLIP
        
        Args:
            image_path: Path to medical image
            text_query: Optional text query for image-text matching
        
        Returns:
            Dict with analysis results
        """
        if not self.model_status["biomedclip"]:
            return {"error": "BiomedCLIP not loaded", "success": False}
        
        try:
            processor = self.models["biomedclip_processor"]
            model = self.models["biomedclip"]
            
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Default medical queries if none provided
            if not text_query:
                queries = [
                    "normal chest x-ray",
                    "pneumonia",
                    "tuberculosis",
                    "lung cancer",
                    "pleural effusion",
                    "cardiomegaly",
                    "healthy tissue",
                    "abnormal finding"
                ]
            else:
                queries = [text_query]
            
            # Process
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
            
            # Get similarities
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
    
    def analyze_blood_report(self, report_text: str) -> Dict[str, Any]:
        """
        Analyze blood report using local models
        
        Args:
            report_text: Blood report text
        
        Returns:
            Dict with analysis
        """
        prompt = f"""Analyze this blood report and identify:
1. All parameters with values
2. Which are normal/abnormal
3. Possible conditions indicated
4. Recommendations

Blood Report:
{report_text}

Analysis:"""
        
        result = self.generate_response(prompt)
        
        # Also extract entities
        entities = self.extract_medical_entities(report_text)
        
        return {
            "analysis": result.get("response", ""),
            "model_used": result.get("model", "Unknown"),
            "entities": entities,
            "offline": result.get("offline", True),
            "success": result.get("success", False)
        }
    
    def diagnose_symptoms(self, symptoms: str, patient_info: str = None) -> Dict[str, Any]:
        """
        Diagnose based on symptoms using local models
        
        Args:
            symptoms: Patient symptoms
            patient_info: Optional patient information
        
        Returns:
            Dict with diagnosis
        """
        context = f"Patient Info: {patient_info}" if patient_info else ""
        
        prompt = f"""As a medical AI, analyze these symptoms and provide:
1. Top 5 possible conditions (ranked by likelihood)
2. Recommended diagnostic tests
3. Urgency level (Low/Medium/High/Emergency)
4. When to seek immediate medical attention

Symptoms: {symptoms}
{context}

Differential Diagnosis:"""
        
        result = self.generate_response(prompt)
        
        # Extract entities from symptoms
        entities = self.extract_medical_entities(symptoms)
        
        return {
            "diagnosis": result.get("response", ""),
            "model_used": result.get("model", "Unknown"),
            "extracted_symptoms": entities.get("symptoms", []),
            "mentioned_conditions": entities.get("diseases", []),
            "offline": result.get("offline", True),
            "success": result.get("success", False)
        }
    
    def get_medicine_info(self, medicine_name: str) -> Dict[str, Any]:
        """
        Get medicine information using local LLM + database
        
        Args:
            medicine_name: Name of medicine
        
        Returns:
            Dict with medicine details
        """
        # First try built-in database
        try:
            from models.drugbank_loader import DrugBankLoader
            db = DrugBankLoader()
            info = db.lookup_medicine(medicine_name)
            if info.get("found"):
                return info
        except:
            pass
        
        # Fall back to LLM
        prompt = f"""Provide detailed information about the medicine: {medicine_name}

Include:
1. Drug class
2. Common uses
3. Recommended dosage
4. Side effects
5. Contraindications
6. Drug interactions
7. Pregnancy category

Medicine Information:"""
        
        result = self.generate_response(prompt)
        
        return {
            "medicine": medicine_name,
            "information": result.get("response", ""),
            "model_used": result.get("model", "Unknown"),
            "source": "AI Generated",
            "offline": result.get("offline", True),
            "success": result.get("success", False)
        }
    
    def check_drug_interactions(self, drugs: List[str]) -> Dict[str, Any]:
        """
        Check drug interactions
        
        Args:
            drugs: List of drug names
        
        Returns:
            Dict with interaction analysis
        """
        drugs_str = ", ".join(drugs)
        
        prompt = f"""Check for interactions between these medications: {drugs_str}

For each potential interaction, provide:
1. Severity (Major/Moderate/Minor)
2. Effect
3. Mechanism
4. Recommendation

Drug Interaction Analysis:"""
        
        result = self.generate_response(prompt)
        
        return {
            "drugs": drugs,
            "interactions": result.get("response", ""),
            "model_used": result.get("model", "Unknown"),
            "offline": result.get("offline", True),
            "success": result.get("success", False)
        }


# Singleton instance
_unified_ai = None

def get_unified_ai(use_gpu: bool = True, prefer_local: bool = True) -> UnifiedMedicalAI:
    """Get or create unified AI instance"""
    global _unified_ai
    if _unified_ai is None:
        _unified_ai = UnifiedMedicalAI(use_gpu=use_gpu, prefer_local=prefer_local)
    return _unified_ai


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Unified Medical AI...")
    ai = UnifiedMedicalAI(use_gpu=True, prefer_local=True)
    
    # Test text generation
    print("\n1. Testing medical question...")
    result = ai.generate_response("What are the symptoms of diabetes?")
    print(f"Model: {result['model']}")
    print(f"Offline: {result['offline']}")
    print(f"Response: {result['response'][:200]}...")
    
    # Test entity extraction
    print("\n2. Testing entity extraction...")
    entities = ai.extract_medical_entities(
        "Patient has diabetes and takes metformin 500mg twice daily"
    )
    print(f"Entities: {entities}")
    
    # Test medicine lookup
    print("\n3. Testing medicine lookup...")
    med_info = ai.get_medicine_info("paracetamol")
    print(f"Found: {med_info.get('found', False)}")
    
    print("\n✅ All tests completed!")
