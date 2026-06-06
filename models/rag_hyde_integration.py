"""
RAG (Retrieval-Augmented Generation) + HYDE (Hypothetical Document Embeddings)
Integration Module for AI Doctor

This module provides:
1. Partial RAG System - Manual context retrieval from medical datasets
2. HYDE Ready - Prepared for vector database enhancement
3. Context Management - Dynamically augments prompts with medical knowledge
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import json
from pathlib import Path
from datetime import datetime
import hashlib

class RAGSystem:
    """
    Retrieval-Augmented Generation System
    - Manual context retrieval from medical datasets
    - Searchable knowledge base
    - Integration with LLM prompts
    """
    
    def __init__(self):
        """Initialize RAG system with medical datasets"""
        self.medical_datasets = {}
        self.context_cache = {}
        self.load_medical_datasets()
        
    def load_medical_datasets(self):
        """Load medical datasets for retrieval"""
        datasets_path = Path("D:\\medical_datasets")
        
        # Load DrugBank for drug queries
        try:
            drug_path = datasets_path / "drug_bank" / "drugbank_clean.csv"
            if drug_path.exists():
                self.medical_datasets['drugs'] = pd.read_csv(drug_path)
                print(f"[OK] Loaded {len(self.medical_datasets['drugs'])} drugs from DrugBank")
        except Exception as e:
            print(f"[!] Drug dataset error: {str(e)[:50]}")
        
        # Load ECG dataset for cardiac queries
        try:
            ecg_path = datasets_path / "ecg_dataset" / "ecg.csv"
            if ecg_path.exists():
                self.medical_datasets['ecg'] = pd.read_csv(ecg_path)
                print(f"[OK] Loaded ECG dataset with {len(self.medical_datasets['ecg'])} samples")
        except Exception as e:
            print(f"[!] ECG dataset error: {str(e)[:50]}")
        
        # Load Heartbeat for arrhythmia queries
        try:
            hb_path = datasets_path / "heartbeat" / "mitbih_train.csv"
            if hb_path.exists():
                self.medical_datasets['heartbeat'] = pd.read_csv(hb_path, nrows=5000)
                print(f"[OK] Loaded {len(self.medical_datasets['heartbeat'])} heartbeat samples")
        except Exception as e:
            print(f"[!] Heartbeat dataset error: {str(e)[:50]}")
    
    def retrieve_drug_info(self, drug_name: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve drug information from DrugBank
        
        Args:
            drug_name: Drug name to search
            top_k: Number of results to return
            
        Returns:
            List of drug information dicts
        """
        if 'drugs' not in self.medical_datasets:
            return []
        
        drugs_df = self.medical_datasets['drugs']
        
        # Search by drug name (case-insensitive)
        matches = drugs_df[drugs_df['drug_name'].str.lower().str.contains(
            drug_name.lower(), na=False
        )].head(top_k)
        
        results = []
        for _, row in matches.iterrows():
            results.append({
                'name': row.get('drug_name', ''),
                'indication': row.get('indication', ''),
                'mechanism': row.get('mechanism_of_action', ''),
                'side_effects': row.get('side_effects', ''),
                'interactions': row.get('drug_interactions', '')
            })
        
        return results
    
    def retrieve_symptom_context(self, symptom: str, top_k: int = 5) -> Dict:
        """
        Retrieve context for symptoms from medical knowledge base
        
        Args:
            symptom: Symptom description
            top_k: Number of related cases to retrieve
            
        Returns:
            Context dictionary with related information
        """
        context = {
            'symptom': symptom,
            'related_conditions': [],
            'recommended_tests': [],
            'severity_indicators': []
        }
        
        # Define symptom-to-condition mappings (knowledge base)
        symptom_knowledge = {
            'chest pain': {
                'conditions': ['Heart Attack', 'Angina', 'Pulmonary Embolism', 'GERD', 'Panic Disorder'],
                'tests': ['ECG', 'Troponin Test', 'X-ray', 'CT Angiography'],
                'severity': ['Sudden onset', 'Radiating to arm', 'Shortness of breath']
            },
            'shortness of breath': {
                'conditions': ['Asthma', 'COPD', 'Pneumonia', 'Heart Failure', 'Anaphylaxis'],
                'tests': ['Spirometry', 'Chest X-ray', 'D-dimer Test', 'ECG'],
                'severity': ['At rest', 'During minimal activity', 'With chest pain']
            },
            'palpitations': {
                'conditions': ['Atrial Fibrillation', 'Tachycardia', 'Anxiety', 'Hyperthyroidism'],
                'tests': ['ECG', 'Holter Monitor', 'Echocardiogram', 'TSH Test'],
                'severity': ['Persistent', 'With dizziness', 'With syncope']
            },
            'arrhythmia': {
                'conditions': ['Atrial Fibrillation', 'Premature Beats', 'Heart Block', 'SVT'],
                'tests': ['12-lead ECG', '24-48hr Holter', 'Event Monitor', 'Electrophysiology Study'],
                'severity': ['Symptomatic', 'Hemodynamically unstable', 'Ventricular']
            }
        }
        
        # Find matching symptom (case-insensitive substring)
        symptom_lower = symptom.lower()
        for key, knowledge in symptom_knowledge.items():
            if key in symptom_lower or symptom_lower in key:
                context['related_conditions'] = knowledge['conditions']
                context['recommended_tests'] = knowledge['tests']
                context['severity_indicators'] = knowledge['severity']
                break
        
        return context
    
    def create_augmented_prompt(self, 
                               base_prompt: str,
                               patient_data: Dict,
                               retrieved_context: List[str]) -> str:
        """
        Create augmented prompt with retrieved context
        
        Args:
            base_prompt: Original user prompt
            patient_data: Patient information
            retrieved_context: Retrieved relevant information
            
        Returns:
            Augmented prompt with medical context
        """
        augmented = f"""You are an advanced medical AI assistant with access to medical knowledge base.

PATIENT CONTEXT:
- Age: {patient_data.get('age', 'Unknown')}
- Medical History: {patient_data.get('history', 'None provided')}
- Current Medications: {patient_data.get('medications', 'None')}
- Allergies: {patient_data.get('allergies', 'None')}

RETRIEVED MEDICAL KNOWLEDGE:
"""
        for i, context in enumerate(retrieved_context, 1):
            augmented += f"{i}. {context}\n"
        
        augmented += f"""
USER QUERY:
{base_prompt}

INSTRUCTIONS:
1. Use the retrieved medical knowledge to inform your response
2. Provide evidence-based recommendations
3. Suggest relevant tests or specialist consultations when appropriate
4. Always emphasize consulting with healthcare professionals
5. Disclaimer: This is AI assistance, not a substitute for professional medical advice
"""
        return augmented
    
    def format_for_llm(self, drug_info: List[Dict], symptom_context: Dict) -> str:
        """Format retrieved information for LLM consumption"""
        formatted = ""
        
        if drug_info:
            formatted += "DRUG INFORMATION:\n"
            for drug in drug_info:
                formatted += f"- Drug: {drug.get('name', 'Unknown')}\n"
                formatted += f"  Indication: {drug.get('indication', 'N/A')}\n"
                formatted += f"  Mechanism: {drug.get('mechanism', 'N/A')}\n"
                formatted += f"  Side Effects: {drug.get('side_effects', 'N/A')}\n\n"
        
        if symptom_context and symptom_context.get('related_conditions'):
            formatted += "SYMPTOM ANALYSIS:\n"
            formatted += f"- Symptom: {symptom_context.get('symptom', '')}\n"
            formatted += f"- Related Conditions: {', '.join(symptom_context.get('related_conditions', []))}\n"
            formatted += f"- Recommended Tests: {', '.join(symptom_context.get('recommended_tests', []))}\n"
            formatted += f"- Severity Indicators: {', '.join(symptom_context.get('severity_indicators', []))}\n"
        
        return formatted


class HYDESystem:
    """
    HYDE (Hypothetical Document Embeddings) Ready System
    Prepared for vector database enhancement
    
    Features:
    - Generates hypothetical medical documents
    - Ready for embedding generation
    - Compatible with Pinecone/ChromaDB/Weaviate
    """
    
    def __init__(self):
        """Initialize HYDE system"""
        self.hypothetical_docs = []
        self.vector_db_config = {
            'backend': None,
            'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
            'dimension': 384,
            'metric': 'cosine'
        }
    
    def generate_hypothetical_documents(self, query: str) -> List[str]:
        """
        Generate hypothetical medical documents for a query
        Improves retrieval by creating diverse document examples
        
        Args:
            query: Medical query or symptom description
            
        Returns:
            List of hypothetical document texts
        """
        hypothetical_templates = {
            'chest pain': [
                "Patient presents with acute substernal chest pain radiating to left arm with diaphoresis",
                "ECG shows ST elevation in anterior leads consistent with acute MI",
                "Troponin levels elevated indicating myocardial necrosis",
                "Chest pain characterized as crushing pressure worsened with exertion",
                "Differential diagnosis includes ACS, pulmonary embolism, and aortic dissection"
            ],
            'arrhythmia': [
                "Continuous ECG monitoring reveals irregular rhythm at 120 bpm",
                "Atrial fibrillation with rapid ventricular response detected",
                "Patient reports palpitations and lightheadedness lasting 2 hours",
                "Holter monitor shows 150+ episodes of premature beats daily",
                "Rate control achieved with beta-blocker therapy"
            ],
            'breathlessness': [
                "Shortness of breath at rest and with minimal exertion",
                "Spirometry shows FEV1/FVC ratio < 70% indicating airflow obstruction",
                "Chest X-ray reveals pulmonary edema with bilateral infiltrates",
                "DLCO reduced suggesting possible interstitial lung disease",
                "Oxygen saturation drops to 88% with minimal activity"
            ],
            'fever': [
                "Patient presents with high fever (39.5°C) and chills",
                "Blood cultures positive for gram-positive cocci",
                "WBC count elevated at 15,000/μL with left shift",
                "Lactate elevated suggesting sepsis",
                "Broad-spectrum antibiotics initiated"
            ]
        }
        
        # Find matching category
        query_lower = query.lower()
        docs = []
        
        for key, templates in hypothetical_templates.items():
            if key in query_lower or query_lower in key:
                docs = templates
                break
        
        # Default hypothetical docs if no match
        if not docs:
            docs = [
                f"Medical case involving {query}",
                f"Clinical presentation of {query} in patient",
                f"Diagnostic workup for {query}",
                f"Treatment plan for {query}",
                f"Follow-up management of {query}"
            ]
        
        return docs
    
    def get_vector_db_setup_code(self) -> str:
        """Get code for setting up vector database"""
        setup_code = """
# Vector DB Setup for HYDE + RAG

# Option 1: Pinecone
import pinecone
pinecone.init(api_key='YOUR_API_KEY', environment='us-west1-gcp')
index = pinecone.Index('medical-rag')

# Option 2: ChromaDB (Local)
import chromadb
client = chromadb.Client()
collection = client.create_collection(name='medical-knowledge')

# Option 3: Weaviate (Docker)
import weaviate
client = weaviate.Client('http://localhost:8080')

# Embedding Model
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings and store
embeddings = model.encode(medical_documents)
# Store in vector DB with metadata
"""
        return setup_code
    
    def get_retrieval_function(self) -> str:
        """Get function template for HYDE-enhanced retrieval"""
        retrieval_code = """
def hyde_enhanced_retrieval(query, k=5):
    # Generate hypothetical documents for query
    hyde_docs = generate_hypothetical_documents(query)
    
    # Embed hypothetical documents
    hyde_embeddings = embed_documents(hyde_docs)
    
    # Average embeddings for better query representation
    query_embedding = np.mean(hyde_embeddings, axis=0)
    
    # Retrieve similar documents from vector DB
    results = vector_db.similarity_search(
        query_embedding,
        k=k,
        filters={'domain': 'medical'}
    )
    
    return results
"""
        return retrieval_code


class RAGHYDEIntegration:
    """Complete RAG + HYDE Integration"""
    
    def __init__(self):
        self.rag_system = RAGSystem()
        self.hyde_system = HYDESystem()
        self.integration_log = []
    
    def process_medical_query(self, 
                             query: str,
                             patient_data: Dict = None,
                             use_hyde: bool = False) -> Dict:
        """
        Process medical query with RAG + optional HYDE
        
        Args:
            query: Medical query/symptom
            patient_data: Patient information
            use_hyde: Whether to use HYDE enhancement
            
        Returns:
            Dictionary with retrieved context and LLM prompt
        """
        result = {
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'rag_retrieved': [],
            'hyde_enhanced': False,
            'augmented_prompt': '',
            'metadata': {}
        }
        
        # Standard RAG retrieval
        drug_queries = self._extract_drugs_from_query(query)
        symptom_context = self.rag_system.retrieve_symptom_context(query)
        
        retrieved_info = []
        for drug in drug_queries:
            drug_info = self.rag_system.retrieve_drug_info(drug)
            retrieved_info.extend(self.rag_system.format_for_llm(drug_info, {}))
        
        retrieved_info.append(
            self.rag_system.format_for_llm([], symptom_context)
        )
        
        result['rag_retrieved'] = retrieved_info
        
        # Optional HYDE enhancement
        if use_hyde:
            hyde_docs = self.hyde_system.generate_hypothetical_documents(query)
            result['hyde_enhanced'] = True
            result['metadata']['hypothetical_docs'] = hyde_docs
        
        # Create augmented prompt
        if not patient_data:
            patient_data = {'age': 'Unknown', 'history': 'Not provided'}
        
        result['augmented_prompt'] = self.rag_system.create_augmented_prompt(
            query,
            patient_data,
            retrieved_info
        )
        
        self.integration_log.append(result)
        return result
    
    def _extract_drugs_from_query(self, query: str) -> List[str]:
        """Extract drug names from query"""
        # Simple extraction - can be enhanced with NER
        drugs = []
        common_drugs = ['aspirin', 'metformin', 'lisinopril', 'atorvastatin', 
                       'metoprolol', 'amoxicillin', 'ibuprofen']
        
        query_lower = query.lower()
        for drug in common_drugs:
            if drug in query_lower:
                drugs.append(drug)
        
        return drugs
    
    def get_status_report(self) -> Dict:
        """Get RAG + HYDE system status"""
        return {
            'system': 'RAG + HYDE Integration',
            'rag_status': 'ACTIVE - Partial implementation',
            'hyde_status': 'READY - Awaiting vector DB',
            'medical_datasets_loaded': len(self.rag_system.medical_datasets),
            'queries_processed': len(self.integration_log),
            'vector_db_compatible': ['Pinecone', 'ChromaDB', 'Weaviate'],
            'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
            'next_upgrade': 'Integrate vector database (ChromaDB recommended)'
        }


# Initialization
rag_hyde_system = RAGHYDEIntegration()

if __name__ == "__main__":
    # Test the system
    print("RAG + HYDE System Initialized")
    print(rag_hyde_system.get_status_report())
