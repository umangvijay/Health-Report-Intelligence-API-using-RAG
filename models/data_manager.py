"""
Data Storage Manager for AI Doctor
Handles all data storage and retrieval operations + RAG + embeddings
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import chromadb
from chromadb.config import Settings
import pandas as pd
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except:
    EMBEDDINGS_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except:
    FAISS_AVAILABLE = False

class DataStorageManager:
    def __init__(self):
        # Create data directories
        self.data_root = Path("./ai_doctor_data")
        self.users_dir = self.data_root / "users"
        self.consultations_dir = self.data_root / "consultations"
        self.images_dir = self.data_root / "medical_images"
        self.reports_dir = self.data_root / "medical_reports"
        self.feedback_dir = self.data_root / "feedback"
        self.analytics_dir = self.data_root / "analytics"
        self.embeddings_dir = self.data_root / "embeddings"
        
        # Create all directories
        for dir_path in [self.users_dir, self.consultations_dir, self.images_dir, 
                         self.reports_dir, self.feedback_dir, self.analytics_dir, self.embeddings_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB for vector storage
        self.chroma_client = chromadb.Client(Settings(
            persist_directory=str(self.data_root / "chromadb"),
            anonymized_telemetry=False
        ))
        
        # Create collections
        self.medical_records = self.chroma_client.get_or_create_collection(
            name="medical_records"
        )
        
        self.symptoms_db = self.chroma_client.get_or_create_collection(
            name="symptoms_database"
        )
        
        # Initialize embeddings
        self.embedding_model = None
        self.faiss_index = None
        self._init_embeddings()
        
        # Data structure info
        self.data_structure = {
            "users": "JSON files - User profiles and authentication",
            "consultations": "JSON files - All consultation history",
            "medical_images": "Binary files - Uploaded medical images",
            "medical_reports": "PDF/Text files - Medical reports",
            "feedback": "JSON files - User feedback for AI learning",
            "analytics": "CSV/JSON files - Health analytics data",
            "chromadb": "Vector database - Semantic search for medical records",
            "embeddings": "FAISS indices + embeddings for RAG"
        }
    
    def _init_embeddings(self):
        """Initialize sentence-transformers embeddings"""
        if EMBEDDINGS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            except:
                print("⚠️ Embedding model failed to load - RAG disabled")
                self.embedding_model = None
    
    def embed_text(self, text: str) -> Optional[np.ndarray]:
        """Embed text to vector"""
        if self.embedding_model is None:
            return None
        try:
            return self.embedding_model.encode(text, convert_to_numpy=True)
        except:
            return None
    
    def add_to_rag_index(self, text: str, metadata: Dict) -> bool:
        """Add document to RAG index"""
        if self.embedding_model is None:
            return False
        try:
            embedding = self.embed_text(text)
            if embedding is None:
                return False
            
            doc_id = f"rag_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
            
            # Store in ChromaDB with embeddings
            self.medical_records.add(
                documents=[text],
                embeddings=[embedding.tolist()],
                metadatas=[metadata],
                ids=[doc_id]
            )
            return True
        except Exception as e:
            print(f"RAG indexing error: {e}")
            return False
    
    def retrieve_similar_docs(self, query: str, n_results: int = 5) -> List[Dict]:
        """Retrieve similar documents using RAG"""
        try:
            results = self.medical_records.query(
                query_texts=[query],
                n_results=n_results
            )
            
            docs = []
            for i, doc in enumerate(results.get('documents', [[]])[0]):
                docs.append({
                    "content": doc,
                    "metadata": results['metadatas'][0][i] if results.get('metadatas') else {},
                    "distance": results['distances'][0][i] if results.get('distances') else 0
                })
            return docs
        except Exception as e:
            print(f"RAG retrieval error: {e}")
            return []
    
    def save_consultation(self, patient_id: str, consultation_data: Dict) -> str:
        """Save consultation data"""
        consultation_id = f"consult_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save to JSON file
        file_path = self.consultations_dir / f"{consultation_id}.json"
        consultation_data['id'] = consultation_id
        consultation_data['timestamp'] = datetime.now().isoformat()
        
        with open(file_path, 'w') as f:
            json.dump(consultation_data, f, indent=2)
        
        # Save to ChromaDB for semantic search
        self.medical_records.add(
            documents=[json.dumps(consultation_data)],
            metadatas=[{
                "patient_id": patient_id,
                "type": "consultation",
                "date": consultation_data['timestamp']
            }],
            ids=[consultation_id]
        )
        
        return consultation_id
    
    def get_patient_consultations(self, patient_id: str, limit: int = 10) -> List[Dict]:
        """Get patient's consultation history"""
        consultations = []
        
        # Search in ChromaDB
        results = self.medical_records.query(
            query_texts=[f"patient_id: {patient_id}"],
            n_results=limit,
            where={"patient_id": patient_id}
        )
        
        if results['documents']:
            for doc in results['documents'][0]:
                try:
                    consultations.append(json.loads(doc))
                except:
                    pass
        
        return consultations
    
    def save_medical_image(self, patient_id: str, image_data: bytes, image_type: str) -> str:
        """Save medical image"""
        image_id = f"img_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        file_path = self.images_dir / f"{image_id}.png"
        
        with open(file_path, 'wb') as f:
            f.write(image_data)
        
        # Save metadata
        metadata = {
            "image_id": image_id,
            "patient_id": patient_id,
            "type": image_type,
            "path": str(file_path),
            "timestamp": datetime.now().isoformat()
        }
        
        metadata_path = self.images_dir / f"{image_id}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return image_id
    
    def save_feedback(self, patient_id: str, feedback_data: Dict) -> bool:
        """Save user feedback for AI learning"""
        feedback_id = f"feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        feedback_data['id'] = feedback_id
        feedback_data['patient_id'] = patient_id
        feedback_data['timestamp'] = datetime.now().isoformat()
        
        file_path = self.feedback_dir / f"{feedback_id}.json"
        with open(file_path, 'w') as f:
            json.dump(feedback_data, f, indent=2)
        
        return True
    
    def get_analytics_data(self, patient_id: str) -> Dict:
        """Get patient analytics data"""
        # Get all consultations
        consultations = self.get_patient_consultations(patient_id, limit=100)
        
        # Calculate analytics
        analytics = {
            "total_consultations": len(consultations),
            "common_symptoms": {},
            "health_score_trend": [],
            "consultation_dates": []
        }
        
        # Process consultations
        for consultation in consultations:
            # Extract symptoms
            if 'symptoms' in consultation:
                symptoms = consultation['symptoms'].lower().split()
                for symptom in ['fever', 'headache', 'cough', 'fatigue', 'pain']:
                    if symptom in symptoms:
                        analytics['common_symptoms'][symptom] = analytics['common_symptoms'].get(symptom, 0) + 1
            
            # Add dates
            if 'timestamp' in consultation:
                analytics['consultation_dates'].append(consultation['timestamp'])
        
        # Generate health score trend (simulated)
        base_score = 75
        for i in range(min(10, len(consultations))):
            score = base_score + (i * 2) + (5 if i % 2 == 0 else -2)
            analytics['health_score_trend'].append({
                'index': i,
                'score': min(100, max(0, score))
            })
        
        return analytics
    
    def export_patient_data(self, patient_id: str, format: str = 'json') -> str:
        """Export all patient data"""
        export_data = {
            "patient_id": patient_id,
            "export_date": datetime.now().isoformat(),
            "consultations": self.get_patient_consultations(patient_id, limit=1000),
            "analytics": self.get_analytics_data(patient_id)
        }
        
        export_file = self.data_root / f"export_{patient_id}_{datetime.now().strftime('%Y%m%d')}.{format}"
        
        if format == 'json':
            with open(export_file, 'w') as f:
                json.dump(export_data, f, indent=2)
        elif format == 'csv':
            df = pd.DataFrame(export_data['consultations'])
            df.to_csv(export_file, index=False)
        
        return str(export_file)
    
    def get_storage_info(self) -> Dict:
        """Get information about data storage"""
        info = {
            "storage_location": str(self.data_root.absolute()),
            "structure": self.data_structure,
            "statistics": {}
        }
        
        # Count files in each directory
        for dir_name, dir_path in [
            ("users", self.users_dir),
            ("consultations", self.consultations_dir),
            ("images", self.images_dir),
            ("reports", self.reports_dir),
            ("feedback", self.feedback_dir)
        ]:
            file_count = len(list(dir_path.glob("*.json"))) + len(list(dir_path.glob("*.png")))
            info["statistics"][dir_name] = file_count
        
        # ChromaDB statistics
        try:
            info["statistics"]["vector_records"] = len(self.medical_records.get()['ids'])
        except:
            info["statistics"]["vector_records"] = 0
        
        return info

# Singleton instance
data_manager = DataStorageManager()