"""
DATABASE MODULE - PostgreSQL + SQLite Support
=============================================
Stores structured medical data:
- Patient records
- Consultation history
- Drug interactions
- Lab results
- User feedback (for RLHF)

Supports:
- PostgreSQL (production)
- SQLite (development/fallback)
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import database libraries
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    logger.warning("psycopg2 not installed. PostgreSQL unavailable.")

try:
    import sqlite3
    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False

# Database configuration
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///./ai_doctor.db')
POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'localhost')
POSTGRES_PORT = os.getenv('POSTGRES_PORT', '5432')
POSTGRES_DB = os.getenv('POSTGRES_DB', 'ai_doctor')
POSTGRES_USER = os.getenv('POSTGRES_USER', 'postgres')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', '')


class MedicalDatabase:
    """
    Database for storing medical AI data
    Supports PostgreSQL and SQLite
    """
    
    def __init__(self, use_postgres: bool = False, db_path: str = None):
        """
        Initialize database connection
        
        Args:
            use_postgres: Use PostgreSQL instead of SQLite
            db_path: Path to SQLite database file
        """
        self.use_postgres = use_postgres and POSTGRES_AVAILABLE
        self.db_path = db_path or "ai_doctor_data/ai_doctor.db"
        self.conn = None
        
        self._connect()
        self._create_tables()
    
    def _connect(self):
        """Establish database connection"""
        if self.use_postgres:
            try:
                self.conn = psycopg2.connect(
                    host=POSTGRES_HOST,
                    port=POSTGRES_PORT,
                    database=POSTGRES_DB,
                    user=POSTGRES_USER,
                    password=POSTGRES_PASSWORD
                )
                logger.info("✅ Connected to PostgreSQL")
            except Exception as e:
                logger.error(f"PostgreSQL connection failed: {e}")
                logger.info("Falling back to SQLite...")
                self.use_postgres = False
        
        if not self.use_postgres:
            # Use SQLite
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            logger.info(f"✅ Connected to SQLite: {self.db_path}")
    
    def _create_tables(self):
        """Create database tables"""
        cursor = self.conn.cursor()
        
        # Consultations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS consultations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                query TEXT NOT NULL,
                response TEXT NOT NULL,
                models_used TEXT,
                confidence REAL,
                entities TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Blood reports table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS blood_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                report_data TEXT NOT NULL,
                analysis TEXT,
                abnormal_values TEXT,
                critical_values TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Drug lookups table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS drug_lookups (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                drug_name TEXT NOT NULL,
                drug_info TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # User feedback table (for RLHF)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                consultation_id INTEGER,
                rating INTEGER,
                feedback_text TEXT,
                chosen_response TEXT,
                rejected_response TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (consultation_id) REFERENCES consultations(id)
            )
        """)
        
        # Model performance tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                query_type TEXT,
                response_time_ms INTEGER,
                user_rating INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Patients table (optional, for EHR)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT UNIQUE,
                name TEXT,
                age INTEGER,
                gender TEXT,
                medical_history TEXT,
                allergies TEXT,
                current_medications TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Symptoms history
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS symptoms_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                symptoms TEXT NOT NULL,
                diagnosis TEXT,
                severity TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.conn.commit()
        logger.info("✅ Database tables created/verified")
    
    # ============ CONSULTATION METHODS ============
    
    def save_consultation(self, user_id: str, query: str, response: str,
                         models_used: List[str] = None, confidence: float = None,
                         entities: Dict = None) -> int:
        """Save a medical consultation"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO consultations (user_id, query, response, models_used, confidence, entities)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            query,
            response,
            json.dumps(models_used) if models_used else None,
            confidence,
            json.dumps(entities) if entities else None
        ))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def get_consultation_history(self, user_id: str, limit: int = 50) -> List[Dict]:
        """Get consultation history for a user"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT * FROM consultations
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (user_id, limit))
        
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    # ============ BLOOD REPORT METHODS ============
    
    def save_blood_report(self, user_id: str, report_data: Dict,
                         analysis: str, abnormal: List, critical: List) -> int:
        """Save blood report analysis"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO blood_reports (user_id, report_data, analysis, abnormal_values, critical_values)
            VALUES (?, ?, ?, ?, ?)
        """, (
            user_id,
            json.dumps(report_data),
            analysis,
            json.dumps(abnormal),
            json.dumps(critical)
        ))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def get_blood_reports(self, user_id: str, limit: int = 20) -> List[Dict]:
        """Get blood report history"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT * FROM blood_reports
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (user_id, limit))
        
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    # ============ FEEDBACK METHODS (for RLHF) ============
    
    def save_feedback(self, consultation_id: int, rating: int,
                     feedback_text: str = None,
                     chosen: str = None, rejected: str = None) -> int:
        """Save user feedback for RLHF training"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO user_feedback 
            (consultation_id, rating, feedback_text, chosen_response, rejected_response)
            VALUES (?, ?, ?, ?, ?)
        """, (consultation_id, rating, feedback_text, chosen, rejected))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def get_feedback_for_rlhf(self, min_samples: int = 100) -> List[Dict]:
        """Get feedback data for RLHF training"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT f.*, c.query, c.response
            FROM user_feedback f
            JOIN consultations c ON f.consultation_id = c.id
            WHERE f.chosen_response IS NOT NULL 
              AND f.rejected_response IS NOT NULL
            ORDER BY f.created_at DESC
            LIMIT ?
        """, (min_samples,))
        
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    # ============ PATIENT METHODS ============
    
    def save_patient(self, user_id: str, name: str = None, age: int = None,
                    gender: str = None, medical_history: List[str] = None,
                    allergies: List[str] = None, medications: List[str] = None) -> int:
        """Save or update patient information"""
        cursor = self.conn.cursor()
        
        # Check if patient exists
        cursor.execute("SELECT id FROM patients WHERE user_id = ?", (user_id,))
        existing = cursor.fetchone()
        
        if existing:
            # Update
            cursor.execute("""
                UPDATE patients SET
                    name = COALESCE(?, name),
                    age = COALESCE(?, age),
                    gender = COALESCE(?, gender),
                    medical_history = COALESCE(?, medical_history),
                    allergies = COALESCE(?, allergies),
                    current_medications = COALESCE(?, current_medications),
                    updated_at = CURRENT_TIMESTAMP
                WHERE user_id = ?
            """, (
                name, age, gender,
                json.dumps(medical_history) if medical_history else None,
                json.dumps(allergies) if allergies else None,
                json.dumps(medications) if medications else None,
                user_id
            ))
        else:
            # Insert
            cursor.execute("""
                INSERT INTO patients 
                (user_id, name, age, gender, medical_history, allergies, current_medications)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id, name, age, gender,
                json.dumps(medical_history) if medical_history else None,
                json.dumps(allergies) if allergies else None,
                json.dumps(medications) if medications else None
            ))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def get_patient(self, user_id: str) -> Optional[Dict]:
        """Get patient information"""
        cursor = self.conn.cursor()
        
        cursor.execute("SELECT * FROM patients WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()
        
        if row:
            patient = dict(row)
            # Parse JSON fields
            for field in ['medical_history', 'allergies', 'current_medications']:
                if patient.get(field):
                    try:
                        patient[field] = json.loads(patient[field])
                    except:
                        pass
            return patient
        return None
    
    # ============ MODEL PERFORMANCE ============
    
    def log_model_performance(self, model_name: str, query_type: str,
                             response_time_ms: int, user_rating: int = None):
        """Log model performance for analytics"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO model_performance (model_name, query_type, response_time_ms, user_rating)
            VALUES (?, ?, ?, ?)
        """, (model_name, query_type, response_time_ms, user_rating))
        
        self.conn.commit()
    
    def get_model_stats(self, model_name: str = None) -> Dict:
        """Get model performance statistics"""
        cursor = self.conn.cursor()
        
        if model_name:
            cursor.execute("""
                SELECT 
                    model_name,
                    COUNT(*) as total_queries,
                    AVG(response_time_ms) as avg_response_time,
                    AVG(user_rating) as avg_rating
                FROM model_performance
                WHERE model_name = ?
                GROUP BY model_name
            """, (model_name,))
        else:
            cursor.execute("""
                SELECT 
                    model_name,
                    COUNT(*) as total_queries,
                    AVG(response_time_ms) as avg_response_time,
                    AVG(user_rating) as avg_rating
                FROM model_performance
                GROUP BY model_name
            """)
        
        rows = cursor.fetchall()
        return {row['model_name']: dict(row) for row in rows}
    
    # ============ UTILITY ============
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Singleton instance
_db_instance = None

def get_database(use_postgres: bool = False) -> MedicalDatabase:
    """Get or create database instance"""
    global _db_instance
    if _db_instance is None:
        _db_instance = MedicalDatabase(use_postgres=use_postgres)
    return _db_instance


# PostgreSQL setup instructions
POSTGRES_SETUP = """
================================================================================
POSTGRESQL SETUP INSTRUCTIONS
================================================================================

1. INSTALL POSTGRESQL:
   - Windows: Download from https://www.postgresql.org/download/windows/
   - Or use: choco install postgresql

2. INSTALL PYTHON DRIVER:
   pip install psycopg2-binary

3. CREATE DATABASE:
   psql -U postgres
   CREATE DATABASE ai_doctor;
   \\q

4. SET ENVIRONMENT VARIABLES in .env:
   POSTGRES_HOST=localhost
   POSTGRES_PORT=5432
   POSTGRES_DB=ai_doctor
   POSTGRES_USER=postgres
   POSTGRES_PASSWORD=your_password

5. USE IN CODE:
   from models.database import get_database
   db = get_database(use_postgres=True)
   
   # Save consultation
   db.save_consultation("user123", "What is diabetes?", "Diabetes is...", 
                        models_used=["BioGPT", "ClinicalBERT"])
   
   # Get history
   history = db.get_consultation_history("user123")

================================================================================
"""

if __name__ == "__main__":
    print(POSTGRES_SETUP)
    
    # Test SQLite
    print("\nTesting SQLite database...")
    db = MedicalDatabase(use_postgres=False)
    
    # Test insert
    consultation_id = db.save_consultation(
        user_id="test_user",
        query="What are symptoms of flu?",
        response="Common flu symptoms include...",
        models_used=["BioGPT", "ClinicalBERT"],
        confidence=0.85
    )
    print(f"✅ Saved consultation: {consultation_id}")
    
    # Test retrieve
    history = db.get_consultation_history("test_user")
    print(f"✅ Retrieved {len(history)} consultations")
    
    db.close()
    print("\n✅ Database test complete!")
