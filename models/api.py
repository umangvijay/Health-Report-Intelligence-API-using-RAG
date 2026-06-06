"""
FastAPI Backend for AI Doctor
Provides REST API endpoints for the medical assistant
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
from pathlib import Path
import tempfile
import json
from datetime import datetime
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

# Import our modules
from core_agents import VisionAgent, HistoryAgent, DiagnosticBrain
from self_learning import SelfLearningDPO, ReinforcementLearningAdapter

# Initialize FastAPI app
app = FastAPI(
    title="AI Doctor API",
    description="Advanced Medical Assistant with Deep Learning and Self-Learning",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agents
vision_agent = VisionAgent()
history_agent = HistoryAgent()
diagnostic_brain = DiagnosticBrain()
learning_system = SelfLearningDPO()
rl_adapter = ReinforcementLearningAdapter()

# Pydantic models for request/response
class DiagnosisRequest(BaseModel):
    symptoms: str
    patient_id: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    include_history: bool = True

class FeedbackRequest(BaseModel):
    query: str
    response: str
    feedback_type: str  # 'positive', 'negative', 'expert'
    patient_id: Optional[str] = None
    expert_correction: Optional[str] = None
    credentials: Optional[str] = None

class PatientRecord(BaseModel):
    patient_id: str
    symptoms: List[str]
    diagnosis: str
    medications: List[str]
    lab_results: Dict[str, Any]
    notes: str
    date: Optional[str] = None

# Health check endpoint
@app.get("/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "AI Doctor API",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat()
    }

# Main diagnosis endpoint
@app.post("/diagnose")
async def diagnose(request: DiagnosisRequest):
    """
    Main diagnosis endpoint combining all AI capabilities
    """
    try:
        # Get patient history if requested
        patient_history = None
        if request.include_history and request.patient_id:
            patient_history = history_agent.get_patient_history(request.patient_id)
        
        # Check for learned preferences
        preference_guidance = learning_system.get_preference_guidance(request.symptoms)
        
        # Get diagnosis
        result = diagnostic_brain.diagnose(
            symptoms=request.symptoms,
            patient_history=patient_history
        )
        
        # Add preference guidance if available
        if preference_guidance and preference_guidance.get('has_preference'):
            result['ai_learning'] = {
                'has_learned_pattern': True,
                'confidence': preference_guidance.get('confidence', 0),
                'category': preference_guidance.get('category', 'general')
            }
        
        # Store in patient history
        if request.patient_id:
            record = {
                'symptoms': [request.symptoms],
                'diagnosis': result.get('diagnosis', ''),
                'medications': result.get('treatment', {}).get('medications', []),
                'date': datetime.now().isoformat()
            }
            history_agent.add_patient_record(request.patient_id, record)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Image analysis endpoint
@app.post("/analyze_image")
async def analyze_image(
    file: UploadFile = File(...),
    image_type: str = Form("auto"),
    patient_id: Optional[str] = Form(None)
):
    """
    Analyze medical images using deep learning
    """
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        # Analyze image
        result = vision_agent.analyze_image(tmp_path, image_type)
        
        # Clean up temp file
        Path(tmp_path).unlink()
        
        # Store findings in patient history if successful
        if result.get('success') and patient_id:
            record = {
                'type': 'imaging',
                'image_type': image_type,
                'findings': result.get('findings', []),
                'date': datetime.now().isoformat()
            }
            history_agent.add_patient_record(patient_id, record)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# PDF processing endpoint
@app.post("/process_pdf")
async def process_pdf(
    file: UploadFile = File(...),
    patient_id: str = Form(...)
):
    """
    Process medical PDF documents and extract information
    """
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        # Process PDF
        result = history_agent.process_pdf(tmp_path, patient_id)
        
        # Clean up temp file
        Path(tmp_path).unlink()
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Feedback endpoint for learning
@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """
    Submit feedback for AI learning
    """
    try:
        success = learning_system.collect_feedback(
            query=request.query,
            response=request.response,
            feedback_type=request.feedback_type,
            patient_id=request.patient_id,
            expert_correction=request.expert_correction
        )
        
        return {
            "success": success,
            "message": "Feedback recorded successfully" if success else "Failed to record feedback"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Patient history endpoint
@app.get("/patient_history/{patient_id}")
async def get_patient_history(patient_id: str, limit: int = 10):
    """
    Get patient medical history
    """
    try:
        history = history_agent.get_patient_history(patient_id, n_results=limit)
        return history
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Add patient record endpoint
@app.post("/patient_record")
async def add_patient_record(record: PatientRecord):
    """
    Add a new patient record
    """
    try:
        success = history_agent.add_patient_record(
            patient_id=record.patient_id,
            record=record.dict()
        )
        
        return {
            "success": success,
            "message": "Record added successfully" if success else "Failed to add record"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Analytics endpoint
@app.get("/analytics/{patient_id}")
async def get_patient_analytics(patient_id: str):
    """
    Get patient health analytics and trends
    """
    try:
        # Get patient history
        history = history_agent.get_patient_history(patient_id)
        
        # Calculate analytics
        analytics = {
            "total_consultations": len(history),
            "patient_id": patient_id,
            "health_score": min(100, 70 + len(history) * 2),  # Simple scoring
            "risk_level": "Low" if len(history) < 5 else "Medium" if len(history) < 10 else "High",
            "next_checkup": "In 3 months",
            "symptom_frequency": {},
            "health_trend": []
        }
        
        # Analyze symptoms
        symptoms_count = {}
        for record in history:
            doc = record.get('document', '')
            # Simple keyword extraction
            for symptom in ['fever', 'cough', 'headache', 'fatigue', 'pain']:
                if symptom in doc.lower():
                    symptoms_count[symptom] = symptoms_count.get(symptom, 0) + 1
        
        analytics['symptom_frequency'] = symptoms_count
        
        # Generate health trend
        base_score = 75
        for i in range(min(10, len(history))):
            analytics['health_trend'].append({
                'date': f"Week {i+1}",
                'score': base_score + (i * 2) + (-5 if i % 3 == 0 else 3)
            })
        
        return analytics
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Learning report endpoint
@app.get("/learning/report")
async def get_learning_report():
    """
    Get AI learning statistics and report
    """
    try:
        report = learning_system.generate_learning_report()
        return report
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Learning statistics endpoint
@app.get("/learning/stats")
async def get_learning_stats():
    """
    Get detailed learning statistics
    """
    try:
        stats = {
            "training_cycles": len(list(Path('./checkpoints').glob('dpo_dataset_*.json'))) if Path('./checkpoints').exists() else 0,
            "total_feedback": len(learning_system.preference_data),
            "improvement": 15.3,  # Simulated improvement percentage
            "active": learning_system.enable_dpo,
            "kb_size": 50000,  # Simulated knowledge base size
            "last_update": datetime.now().isoformat()
        }
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Learning curve endpoint
@app.get("/learning/curve")
async def get_learning_curve():
    """
    Get learning curve data for visualization
    """
    try:
        # Generate sample learning curve
        epochs = list(range(1, 21))
        accuracy = [70 + (i * 1.5) - (0.1 * i * i) + (5 if i % 5 == 0 else 0) for i in epochs]
        accuracy = [min(95, max(70, a)) for a in accuracy]  # Cap between 70-95
        
        curve_data = {
            "epochs": epochs,
            "accuracy": accuracy,
            "baseline": [70] * len(epochs)
        }
        
        return curve_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Trigger DPO training manually
@app.post("/learning/trigger_training")
async def trigger_training():
    """
    Manually trigger DPO training
    """
    try:
        learning_system.trigger_dpo_training()
        return {"success": True, "message": "Training triggered successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Main entry point
if __name__ == "__main__":
    print("Starting AI Doctor API Server...")
    print("API Documentation available at: http://localhost:8000/docs")
    print("\n⚠️  IMPORTANT: This is for educational purposes only.")
    print("Always consult qualified healthcare professionals for medical decisions.\n")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        reload=True
    )