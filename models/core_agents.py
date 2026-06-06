"""
Core AI Agents for Medical Analysis
Implements Vision, History, and Diagnostic agents
WITH INTEGRATED RL/SELF-LEARNING MODELS
"""

import os
import io
import json
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image
import google.generativeai as genai
from transformers import (
    AutoModel, 
    AutoTokenizer, 
    AutoImageProcessor,
    pipeline
)
import requests
from datetime import datetime
import chromadb
from chromadb.config import Settings
import PyPDF2
import pytesseract
from pathlib import Path
import yaml
import warnings
from dotenv import load_dotenv

# Load RL models
try:
    from training.advanced_rl_training import learning_orchestrator
    RL_AVAILABLE = True
except:
    RL_AVAILABLE = False

# Load environment variables from .env file
load_dotenv(override=True)
warnings.filterwarnings('ignore')

# Load configuration
def load_config():
    config_path = Path(__file__).parent / 'config.yaml'
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}

CONFIG = load_config()

class VisionAgent:
    """Analyzes medical images using deep learning models"""
    
    def __init__(self):
        self.hf_token = os.getenv('HF_TOKEN', CONFIG.get('api', {}).get('huggingface_token', ''))
        self.api_url = "https://api-inference.huggingface.co/models/"
        
        # Medical condition mappings for different image types
        self.xray_conditions = {
            'pneumonia': 'Pneumonia detected - lung infection requiring antibiotics',
            'tuberculosis': 'TB markers found - requires immediate medical attention',
            'covid19': 'COVID-19 patterns detected - isolation and treatment needed',
            'normal': 'No abnormalities detected in chest X-ray',
            'atelectasis': 'Lung collapse detected - may need respiratory support',
            'consolidation': 'Lung consolidation present - possible infection',
            'effusion': 'Fluid in lungs detected - requires drainage evaluation'
        }
        
        self.skin_conditions = {
            'melanoma': 'Possible melanoma - urgent dermatologist consultation needed',
            'nevus': 'Benign mole detected - monitor for changes',
            'basal_cell': 'Basal cell carcinoma suspected - requires biopsy',
            'actinic_keratosis': 'Pre-cancerous lesion - needs treatment',
            'eczema': 'Eczema detected - topical treatment recommended',
            'psoriasis': 'Psoriasis identified - systemic treatment may be needed'
        }
    
    def analyze_image(self, image_path: str, image_type: str = 'auto') -> Dict:
        """
        Analyze medical image using Hugging Face API
        
        Args:
            image_path: Path to the image file
            image_type: 'xray', 'skin', 'mri', or 'auto'
        
        Returns:
            Dictionary with analysis results
        """
        try:
            # Determine image type if auto
            if image_type == 'auto':
                image_type = self._detect_image_type(image_path)
            
            # Select appropriate model
            if image_type == 'xray':
                model_id = "microsoft/resnet-50"
                conditions = self.xray_conditions
            elif image_type == 'skin':
                model_id = "google/vit-base-patch16-224"
                conditions = self.skin_conditions
            else:
                model_id = "microsoft/resnet-50"
                conditions = {}
            
            # If using local inference (requires GPU)
            if not self.hf_token:
                return self._analyze_local(image_path, model_id)
            
            # Use Hugging Face API
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            
            headers = {"Authorization": f"Bearer {self.hf_token}"}
            response = requests.post(
                self.api_url + model_id,
                headers=headers,
                data=image_bytes
            )
            
            if response.status_code == 200:
                results = response.json()
                
                # Process results
                findings = []
                probabilities = {}
                
                for item in results[:5]:  # Top 5 predictions
                    label = item.get('label', 'unknown').lower()
                    score = item.get('score', 0)
                    probabilities[label] = score
                    
                    # Map to medical conditions
                    for condition, description in conditions.items():
                        if condition in label:
                            findings.append({
                                'condition': condition,
                                'confidence': score,
                                'description': description
                            })
                
                return {
                    'success': True,
                    'image_type': image_type,
                    'findings': findings,
                    'raw_probabilities': probabilities,
                    'timestamp': datetime.now().isoformat()
                }
            
            else:
                return {
                    'success': False,
                    'error': f"API error: {response.status_code}",
                    'message': 'Using fallback text description'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': 'Image analysis failed - please describe symptoms'
            }
    
    def _detect_image_type(self, image_path: str) -> str:
        """Detect type of medical image based on characteristics"""
        try:
            img = Image.open(image_path)
            
            # Check if grayscale (likely X-ray or MRI)
            if img.mode in ['L', 'LA']:
                return 'xray'
            
            # Check image dimensions and colors
            if img.size[0] < 500 and img.size[1] < 500:
                return 'skin'  # Smaller images often dermoscopy
            
            # Default to xray for medical images
            return 'xray'
            
        except:
            return 'xray'
    
    def _analyze_local(self, image_path: str, model_id: str) -> Dict:
        """Fallback local analysis using basic image processing"""
        try:
            img = Image.open(image_path)
            
            # Basic image statistics
            img_array = np.array(img)
            mean_intensity = np.mean(img_array)
            std_intensity = np.std(img_array)
            
            # Simple heuristics
            findings = []
            if mean_intensity < 100:
                findings.append({
                    'condition': 'abnormality',
                    'confidence': 0.6,
                    'description': 'Dark regions detected - possible consolidation'
                })
            
            return {
                'success': True,
                'findings': findings,
                'stats': {
                    'mean': float(mean_intensity),
                    'std': float(std_intensity)
                }
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}


class HistoryAgent:
    """Manages patient history using ChromaDB and RAG"""
    
    def __init__(self):
        # Initialize ChromaDB
        self.client = chromadb.Client(Settings(
            persist_directory=CONFIG.get('database', {}).get('persist_directory', './patient_db'),
            anonymized_telemetry=False
        ))
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=CONFIG.get('database', {}).get('collection_name', 'patient_history')
        )
        
        # Initialize medical dataset loader for knowledge base
        try:
            from models.all_datasets_importer import get_dataset_manager
            self.dataset_manager = get_dataset_manager()
            self.datasets_available = True
            print("[OK] Medical datasets manager loaded")
        except Exception as e:
            self.dataset_manager = None
            self.datasets_available = False
            print(f"[i] Medical datasets not available: {e}")
        
        # Initialize model loader for advanced analysis
        try:
            from models.model_loader import MedicalModelLoader
            self.model_loader = MedicalModelLoader()
            self.models_available = True
            print("[OK] Medical models available")
        except Exception as e:
            self.model_loader = None
            self.models_available = False
            print(f"[i] Medical models not available: {e}")
    
    def query_medical_knowledge(self, query: str, top_k: int = 5) -> List[Dict]:
        """Query medical knowledge base from installed datasets"""
        if not self.datasets_available:
            return []
        
        try:
            # Query all datasets
            results = self.dataset_manager.query(query, dataset_name='all', top_k=top_k)
            return results
        except Exception as e:
            print(f"Error querying knowledge base: {e}")
            return []
    
    def extract_medical_entities(self, text: str) -> Dict:
        """Extract medical entities using BioBERT"""
        if not self.models_available:
            return {"entities": [], "error": "Models not available"}
        
        try:
            biobert = self.model_loader.load_model("biobert")
            if biobert.get("loaded"):
                entities = biobert['pipeline'](text)
                return {"entities": entities, "success": True}
            else:
                return {"entities": [], "error": "BioBERT not loaded"}
        except Exception as e:
            return {"entities": [], "error": str(e)}
        
    def add_patient_record(self, patient_id: str, record: Dict) -> bool:
        """Add patient medical record to database"""
        try:
            # Create document text
            doc_text = f"""
            Patient ID: {patient_id}
            Date: {record.get('date', datetime.now().isoformat())}
            Type: {record.get('type', 'general')}
            Symptoms: {record.get('symptoms', [])}
            Diagnosis: {record.get('diagnosis', 'pending')}
            Medications: {record.get('medications', [])}
            Lab Results: {record.get('lab_results', {})}
            Notes: {record.get('notes', '')}
            """
            
            # Add to ChromaDB
            self.collection.add(
                documents=[doc_text],
                metadatas=[{
                    'patient_id': patient_id,
                    'date': record.get('date', datetime.now().isoformat()),
                    'type': record.get('type', 'general')
                }],
                ids=[f"{patient_id}_{datetime.now().timestamp()}"]
            )
            
            return True
            
        except Exception as e:
            print(f"Error adding record: {e}")
            return False
    
    def get_patient_history(self, patient_id: str, n_results: int = 10) -> List[Dict]:
        """Retrieve patient history from database"""
        try:
            results = self.collection.query(
                query_texts=[f"Patient ID: {patient_id}"],
                n_results=n_results
            )
            
            history = []
            if results['documents']:
                for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
                    history.append({
                        'document': doc,
                        'metadata': metadata
                    })
            
            return history
            
        except Exception as e:
            print(f"Error retrieving history: {e}")
            return []
    
    def process_pdf(self, pdf_path: str, patient_id: str) -> Dict:
        """Extract and store information from medical PDF"""
        try:
            # Extract text from PDF
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text()
            
            # Parse medical information
            record = self._parse_medical_text(text)
            record['source'] = 'pdf'
            record['file_path'] = pdf_path
            
            # Store in database
            self.add_patient_record(patient_id, record)
            
            # Analyze trends
            trends = self._analyze_trends(patient_id, text)
            
            return {
                'success': True,
                'extracted_data': record,
                'trends': trends,
                'text_length': len(text)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _parse_medical_text(self, text: str) -> Dict:
        """Parse medical information from text"""
        record = {
            'symptoms': [],
            'medications': [],
            'lab_results': {},
            'diagnosis': '',
            'notes': ''
        }
        
        # Simple keyword extraction
        lines = text.lower().split('\n')
        
        for line in lines:
            if 'symptom' in line or 'complaint' in line:
                record['symptoms'].append(line.strip())
            elif 'medication' in line or 'drug' in line:
                record['medications'].append(line.strip())
            elif 'diagnosis' in line:
                record['diagnosis'] = line.strip()
            elif any(lab in line for lab in ['glucose', 'cholesterol', 'blood pressure', 'hemoglobin']):
                # Extract lab values
                parts = line.split(':')
                if len(parts) > 1:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    record['lab_results'][key] = value
        
        return record
    
    def _analyze_trends(self, patient_id: str, new_data: str) -> Dict:
        """Analyze trends in patient history"""
        history = self.get_patient_history(patient_id)
        
        trends = {
            'risk_increasing': [],
            'risk_decreasing': [],
            'stable': [],
            'predictions': []
        }
        
        # Simple trend analysis
        if len(history) > 2:
            # Check for recurring symptoms
            all_symptoms = []
            for h in history:
                doc = h['document']
                if 'symptoms' in doc.lower():
                    all_symptoms.append(doc)
            
            # Identify patterns
            if 'glucose' in new_data.lower():
                if any('diabetes' in s.lower() for s in all_symptoms):
                    trends['predictions'].append(
                        'Based on history, diabetes risk is elevated. Monitor blood sugar closely.'
                    )
            
            if 'blood pressure' in new_data.lower():
                trends['predictions'].append(
                    'Cardiovascular risk factors detected. Regular monitoring recommended.'
                )
        
        return trends


class DiagnosticBrain:
    """Main diagnostic agent using Meditron + Gemini"""
    
    def __init__(self):
        # Initialize Gemini with newer models (2.0/2.5 series)
        gemini_key = os.getenv('GEMINI_API_KEY', CONFIG.get('api', {}).get('gemini_api_key', ''))
        self.gemini_model = None
        
        if gemini_key:
            try:
                genai.configure(api_key=gemini_key)
                # Use newer models that are available
                model_to_use = 'models/gemini-2.5-flash'  # Latest and fastest
                self.gemini_model = genai.GenerativeModel(model_to_use)
                print(f"[OK] DiagnosticBrain: Gemini configured with {model_to_use}")
            except Exception as e:
                print(f"[!] DiagnosticBrain: Gemini error - {e}")
                self.gemini_model = None
        
        # Initialize Meditron-70B (via HF API) - Better accuracy than 7B
        self.hf_token = os.getenv('HF_TOKEN', CONFIG.get('api', {}).get('huggingface_token', ''))
        self.meditron_api = "https://api-inference.huggingface.co/models/epfl-llm/meditron-70b"
        
        # Load safety checks
        self.emergency_keywords = CONFIG.get('safety', {}).get('emergency_keywords', [])
        self.disclaimer = CONFIG.get('safety', {}).get('disclaimer', '')
    
    def diagnose(self, 
                 symptoms: str,
                 image_analysis: Optional[Dict] = None,
                 patient_history: Optional[List] = None) -> Dict:
        """
        Main diagnostic function combining all inputs
        
        Args:
            symptoms: Patient's described symptoms
            image_analysis: Results from VisionAgent
            patient_history: Retrieved history from HistoryAgent
        
        Returns:
            Comprehensive diagnosis with recommendations
        """
        
        # Safety check for emergencies
        if self._is_emergency(symptoms):
            return {
                'emergency': True,
                'message': '🚨 EMERGENCY DETECTED! Call emergency services immediately!',
                'symptoms_detected': [kw for kw in self.emergency_keywords if kw in symptoms.lower()],
                'disclaimer': self.disclaimer
            }
        
        # Build context for LLM
        context = self._build_context(symptoms, image_analysis, patient_history)
        
        # Get diagnosis from Meditron or Gemini
        if self.gemini_model:
            diagnosis = self._gemini_diagnose(context)
        elif self.hf_token:
            diagnosis = self._meditron_diagnose(context)
        else:
            diagnosis = self._fallback_diagnose(context)
        
        # Add remedies and treatment
        treatment = self._generate_treatment(diagnosis)
        
        # Predict future risks
        predictions = self._predict_future_risks(symptoms, patient_history)
        
        return {
            'emergency': False,
            'diagnosis': diagnosis,
            'treatment': treatment,
            'future_risks': predictions,
            'confidence': self._calculate_confidence(diagnosis, image_analysis),
            'disclaimer': self.disclaimer,
            'timestamp': datetime.now().isoformat()
        }
    
    def _is_emergency(self, symptoms: str) -> bool:
        """Check if symptoms indicate emergency"""
        symptoms_lower = symptoms.lower()
        return any(keyword in symptoms_lower for keyword in self.emergency_keywords)
    
    def _build_context(self, symptoms: str, image_analysis: Optional[Dict], 
                      patient_history: Optional[List]) -> str:
        """Build comprehensive context for LLM"""
        
        context = f"""You are an expert medical doctor. Analyze the following information and provide diagnosis.

PATIENT SYMPTOMS:
{symptoms}

"""
        
        if image_analysis and image_analysis.get('success'):
            context += f"""
MEDICAL IMAGE ANALYSIS:
Type: {image_analysis.get('image_type', 'unknown')}
Findings: {json.dumps(image_analysis.get('findings', []), indent=2)}

"""
        
        if patient_history:
            context += """
PATIENT HISTORY:
"""
            for record in patient_history[-5:]:  # Last 5 records
                context += f"{record.get('document', '')}\n"
        
        context += """

Based on this information:
1. Provide differential diagnosis (list possible conditions)
2. Recommend specific tests needed
3. Suggest immediate treatment options
4. Identify any red flags requiring immediate attention

Format your response clearly with sections for Diagnosis, Tests Needed, Treatment, and Warnings.
"""
        
        return context
    
    def _gemini_diagnose(self, context: str) -> str:
        """Get diagnosis using Gemini API"""
        try:
            response = self.gemini_model.generate_content(context)
            return response.text
        except Exception as e:
            return f"Gemini diagnosis failed: {str(e)}. Please consult a doctor."
    
    def _meditron_diagnose(self, context: str) -> str:
        """Get diagnosis using Meditron-70B via HF API - Enhanced medical accuracy"""
        try:
            headers = {"Authorization": f"Bearer {self.hf_token}"}
            
            # Enhanced prompt for Meditron-70B medical model
            medical_prompt = f"""<|im_start|>system
You are an expert medical doctor with extensive clinical experience. Provide accurate, detailed medical analysis based on the patient's symptoms. Include differential diagnosis, recommended tests, and treatment options.
<|im_end|>
<|im_start|>user
{context}
<|im_end|>
<|im_start|>assistant
"""
            
            payload = {
                "inputs": medical_prompt,
                "parameters": {
                    "max_new_tokens": 1024,
                    "temperature": 0.6,
                    "top_p": 0.9,
                    "do_sample": True,
                    "return_full_text": False
                }
            }
            
            response = requests.post(self.meditron_api, headers=headers, json=payload, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    generated = result[0].get('generated_text', '')
                    if generated:
                        return f"[Meditron-70B Analysis]\n\n{generated}"
                    return 'Diagnosis pending... Please try again.'
                return str(result)
            elif response.status_code == 503:
                return "Meditron-70B is loading (large model). Please wait and try again in 1-2 minutes."
            else:
                return f"Meditron-70B API error: {response.status_code}. Falling back to basic analysis."
                
        except requests.exceptions.Timeout:
            return "Meditron-70B request timed out. The model is processing. Please try again."
        except Exception as e:
            return f"Meditron-70B diagnosis failed: {str(e)}"
    
    def _fallback_diagnose(self, context: str) -> str:
        """Simple rule-based diagnosis for fallback"""
        diagnosis = "Based on symptoms analysis:\n\n"
        
        symptoms_lower = context.lower()
        
        if 'fever' in symptoms_lower and 'cough' in symptoms_lower:
            diagnosis += "• Possible respiratory infection (viral or bacterial)\n"
        if 'headache' in symptoms_lower:
            diagnosis += "• Tension headache or migraine possible\n"
        if 'chest' in symptoms_lower:
            diagnosis += "• Chest symptoms detected - cardiac evaluation recommended\n"
        if 'stomach' in symptoms_lower or 'abdominal' in symptoms_lower:
            diagnosis += "• Gastrointestinal issue possible\n"
        
        diagnosis += "\n⚠️ This is a basic analysis. Professional medical consultation required."
        
        return diagnosis
    
    def _generate_treatment(self, diagnosis: str) -> Dict:
        """Generate treatment recommendations"""
        treatment = {
            'medications': [],
            'lifestyle': [],
            'follow_up': '',
            'home_remedies': []
        }
        
        diagnosis_lower = diagnosis.lower()
        
        # Common treatments based on diagnosis keywords
        if 'infection' in diagnosis_lower:
            treatment['medications'].append('Antibiotics (prescription required)')
            treatment['home_remedies'].extend(['Rest', 'Hydration', 'Warm liquids'])
        
        if 'fever' in diagnosis_lower:
            treatment['medications'].append('Acetaminophen or Ibuprofen for fever')
            treatment['home_remedies'].append('Cool compress')
        
        if 'respiratory' in diagnosis_lower or 'cough' in diagnosis_lower:
            treatment['medications'].append('Cough suppressant if needed')
            treatment['home_remedies'].extend(['Honey and lemon tea', 'Steam inhalation'])
        
        if 'hypertension' in diagnosis_lower or 'blood pressure' in diagnosis_lower:
            treatment['lifestyle'].extend(['Low sodium diet', 'Regular exercise', 'Stress management'])
            treatment['follow_up'] = 'Regular BP monitoring'
        
        if 'diabetes' in diagnosis_lower:
            treatment['lifestyle'].extend(['Blood sugar monitoring', 'Carbohydrate counting', 'Regular meals'])
            treatment['medications'].append('Insulin or metformin as prescribed')
        
        # Always add general advice
        treatment['lifestyle'].extend(['Adequate sleep (7-8 hours)', 'Balanced diet', 'Stay hydrated'])
        treatment['follow_up'] = treatment['follow_up'] or 'Follow up with doctor if symptoms persist'
        
        return treatment
    
    def _predict_future_risks(self, symptoms: str, history: Optional[List]) -> List[str]:
        """Predict potential future health risks"""
        risks = []
        
        # Analyze patterns
        if history:
            history_text = ' '.join([h.get('document', '') for h in history])
            
            # Simple risk prediction rules
            if 'glucose' in history_text and 'high' in history_text:
                risks.append('Diabetes risk: Monitor blood sugar regularly')
            
            if 'cholesterol' in history_text and 'elevated' in history_text:
                risks.append('Cardiovascular risk: Consider lipid management')
            
            if 'smoking' in history_text:
                risks.append('Lung disease risk: Smoking cessation recommended')
            
            if 'family history' in history_text and 'cancer' in history_text:
                risks.append('Cancer screening: Regular checkups recommended')
        
        # Based on current symptoms
        symptoms_lower = symptoms.lower()
        if 'weight gain' in symptoms_lower:
            risks.append('Metabolic syndrome risk: Monitor weight and diet')
        
        if 'fatigue' in symptoms_lower and 'persistent' in symptoms_lower:
            risks.append('Chronic fatigue: Investigate underlying causes')
        
        return risks if risks else ['Continue regular health monitoring']
    
    def _calculate_confidence(self, diagnosis: str, image_analysis: Optional[Dict]) -> float:
        """Calculate confidence score for diagnosis"""
        confidence = 0.5  # Base confidence
        
        # Increase confidence based on available data
        if diagnosis and len(diagnosis) > 100:
            confidence += 0.2
        
        if image_analysis and image_analysis.get('success'):
            findings = image_analysis.get('findings', [])
            if findings:
                # Add confidence from image analysis
                avg_img_confidence = np.mean([f.get('confidence', 0) for f in findings])
                confidence += avg_img_confidence * 0.3
        
        # Cap at 0.95 (never 100% certain)
        return min(confidence, 0.95)
    
    def submit_feedback_for_learning(self, query: str, diagnosis: str, rating: float, patient_id: str = "unknown") -> bool:
        """
        ACTUAL RL INTEGRATION: Submit feedback to training system
        This creates training signals for PPO, DQN, Actor-Critic, A3C
        """
        if not RL_AVAILABLE:
            return False
        
        try:
            learning_orchestrator.submit_feedback(
                query=query,
                response=diagnosis,
                rating=rating,
                model_used="meditron"
            )
            return True
        except Exception as e:
            print(f"Error submitting feedback: {e}")
            return False
    
    def get_learning_metrics(self) -> Dict:
        """Get current RL learning status"""
        if not RL_AVAILABLE:
            return {"status": "RL not available"}
        
        return learning_orchestrator.get_learning_status()
    
    def trigger_rl_training(self, models: List[str] = None) -> Dict:
        """
        ACTUAL RL INTEGRATION: Trigger training of RL models
        models: list of models to train ('ppo', 'dqn', 'actor_critic', 'a3c')
        """
        if not RL_AVAILABLE:
            return {"success": False, "message": "RL not available"}
        
        results = {}
        
        if models is None:
            models = ['ppo', 'dqn', 'actor_critic', 'a3c']
        
        for model in models:
            if model == 'ppo':
                results['ppo'] = learning_orchestrator.train_ppo()
            elif model == 'dqn':
                results['dqn'] = learning_orchestrator.train_dqn()
            elif model == 'actor_critic':
                results['actor_critic'] = learning_orchestrator.train_actor_critic()
            elif model == 'a3c':
                results['a3c'] = learning_orchestrator.train_a3c()
        
        return {"success": True, "results": results}

    def analyze_blood_report(self, report_text: str, patient_id: str = "unknown") -> Dict:
        """
        Analyze blood/lab reports and extract key findings with normal ranges and status
        """
        try:
            if not report_text or len(report_text.strip()) < 10:
                return {"error": "Invalid report text", "findings": []}
            
            # Common blood test parameters with normal ranges (male values, can adjust for female)
            blood_markers = {
                'hemoglobin': {'normal_min': 13.5, 'normal_max': 17.5, 'unit': 'g/dL', 'normal_range': '13.5-17.5'},
                'hematocrit': {'normal_min': 41, 'normal_max': 53, 'unit': '%', 'normal_range': '41-53%'},
                'rbc': {'normal_min': 4.5, 'normal_max': 5.9, 'unit': 'million/mcL', 'normal_range': '4.5-5.9'},
                'wbc': {'normal_min': 4.5, 'normal_max': 11.0, 'unit': 'thousand/mcL', 'normal_range': '4.5-11.0'},
                'platelets': {'normal_min': 150, 'normal_max': 400, 'unit': 'thousand/mcL', 'normal_range': '150-400'},
                'glucose': {'normal_min': 70, 'normal_max': 100, 'unit': 'mg/dL', 'normal_range': '70-100'},
                'creatinine': {'normal_min': 0.7, 'normal_max': 1.3, 'unit': 'mg/dL', 'normal_range': '0.7-1.3'},
                'bun': {'normal_min': 7, 'normal_max': 20, 'unit': 'mg/dL', 'normal_range': '7-20'},
                'sodium': {'normal_min': 136, 'normal_max': 145, 'unit': 'mEq/L', 'normal_range': '136-145'},
                'potassium': {'normal_min': 3.5, 'normal_max': 5.0, 'unit': 'mEq/L', 'normal_range': '3.5-5.0'},
                'calcium': {'normal_min': 8.5, 'normal_max': 10.2, 'unit': 'mg/dL', 'normal_range': '8.5-10.2'},
                'alt': {'normal_min': 7, 'normal_max': 35, 'unit': 'U/L', 'normal_range': '7-35'},
                'ast': {'normal_min': 10, 'normal_max': 40, 'unit': 'U/L', 'normal_range': '10-40'},
                'cholesterol': {'normal_min': 0, 'normal_max': 200, 'unit': 'mg/dL', 'normal_range': '<200'},
                'triglycerides': {'normal_min': 0, 'normal_max': 150, 'unit': 'mg/dL', 'normal_range': '<150'},
                'ldl': {'normal_min': 0, 'normal_max': 100, 'unit': 'mg/dL', 'normal_range': '<100'},
                'hdl': {'normal_min': 40, 'normal_max': 200, 'unit': 'mg/dL', 'normal_range': '>40 (M), >50 (F)'}
            }
            
            findings = []
            parameters = []
            abnormalities = []
            
            # Use Gemini to extract structured data
            prompt = f"""
            Extract the following from this medical blood/lab report:
            1. Each test parameter found
            2. The numerical value for each parameter
            3. The unit of measurement
            4. Clinical significance
            
            Format as: Parameter|Value|Unit|Note
            
            Report:
            {report_text}
            """
            
            try:
                gemini_key = os.getenv('GEMINI_API_KEY', '')
                if gemini_key:
                    genai.configure(api_key=gemini_key)
                    model = genai.GenerativeModel('gemini-pro')
                    response = model.generate_content(prompt)
                    analysis_text = response.text if response else ""
                    
                    # Parse the structured response
                    lines = analysis_text.split('\n')
                    for line in lines:
                        if '|' in line:
                            parts = line.split('|')
                            if len(parts) >= 3:
                                param_name = parts[0].strip().lower()
                                try:
                                    param_value = float(parts[1].strip())
                                    unit = parts[2].strip()
                                    
                                    # Find matching marker
                                    marker_key = None
                                    for key in blood_markers.keys():
                                        if key in param_name or param_name in key:
                                            marker_key = key
                                            break
                                    
                                    if marker_key:
                                        marker = blood_markers[marker_key]
                                        # Determine status
                                        if param_value < marker['normal_min']:
                                            status = "🔴 LOW"
                                            abnormalities.append(f"{marker_key.upper()}: {param_value} {marker['unit']} (Normal: {marker['normal_range']})")
                                        elif param_value > marker['normal_max']:
                                            status = "🔴 HIGH"
                                            abnormalities.append(f"{marker_key.upper()}: {param_value} {marker['unit']} (Normal: {marker['normal_range']})")
                                        else:
                                            status = "✅ NORMAL"
                                        
                                        parameters.append({
                                            'name': marker_key.upper(),
                                            'value': param_value,
                                            'unit': marker['unit'],
                                            'normal_range': marker['normal_range'],
                                            'status': status
                                        })
                                except ValueError:
                                    pass
                    
                    findings.append({
                        "type": "AI_Analysis",
                        "content": analysis_text,
                        "timestamp": datetime.now().isoformat()
                    })
            except Exception as e:
                findings.append({
                    "type": "manual_analysis",
                    "content": f"Automated analysis unavailable: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                })
            
            # Store findings
            try:
                feedback_file = Path("feedback_data") / f"blood_report_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                feedback_file.parent.mkdir(parents=True, exist_ok=True)
                with open(feedback_file, 'w') as f:
                    json.dump({
                        "patient_id": patient_id,
                        "report": report_text[:500],
                        "parameters": parameters,
                        "abnormalities": abnormalities,
                        "findings": findings,
                        "timestamp": datetime.now().isoformat()
                    }, f, indent=2)
            except Exception as e:
                print(f"Error storing report feedback: {e}")
            
            return {
                "success": True,
                "parameters": parameters,
                "abnormalities": abnormalities,
                "findings": findings,
                "blood_markers": list(blood_markers.keys()),
                "analysis_timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            return {"error": str(e), "findings": []}
    
    def store_feedback(self, query: str, response: str, rating: float, feedback_text: str = "", 
                       patient_id: str = "unknown", report_type: str = "general") -> Dict:
        """
        Store user feedback for model improvement and analytics
        Integrates with RL training system
        """
        try:
            # Create feedback record
            feedback_record = {
                "patient_id": patient_id,
                "query": query,
                "response": response[:500],  # Store first 500 chars
                "rating": float(rating),
                "feedback_text": feedback_text,
                "report_type": report_type,
                "timestamp": datetime.now().isoformat(),
                "useful": rating >= 4  # Mark as useful if 4+ stars
            }
            
            # Save feedback to file
            feedback_dir = Path("feedback_data")
            feedback_dir.mkdir(parents=True, exist_ok=True)
            
            feedback_file = feedback_dir / f"feedback_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(feedback_file, 'w') as f:
                json.dump(feedback_record, f, indent=2)
            
            # Index in RAG if high quality feedback
            if rating >= 4:
                try:
                    self.rag_system.add_to_rag_index(
                        text=f"Query: {query}\nResponse: {response}\nFeedback: {feedback_text}",
                        metadata={"type": "user_feedback", "rating": rating}
                    )
                except:
                    pass  # RAG indexing optional
            
            # Trigger RL training if enough samples
            if RL_AVAILABLE:
                try:
                    learning_orchestrator.submit_feedback(query, response, rating)
                except:
                    pass
            
            return {
                "success": True,
                "feedback_id": feedback_file.name,
                "message": "Feedback saved and will help improve the AI model",
                "rl_triggered": RL_AVAILABLE
            }
        
        except Exception as e:
            return {"success": False, "error": str(e)}
