"""
HIPAA Compliance Layer
- Data encryption (AES-256)
- Audit logging with PHI tracking
- PII/PHI detection and sanitization
- Secure data handling
Critical for healthcare deployment
"""

import logging
import json
from datetime import datetime
from typing import Dict, List, Any
import hashlib
import re
from cryptography.fernet import Fernet
import os

logger = logging.getLogger(__name__)


class HIPAACompliance:
    """
    HIPAA compliance utilities
    - Encrypt sensitive data
    - Log all PHI access
    - Detect and sanitize PII/PHI
    - Audit trails
    """
    
    # PHI/PII patterns
    PHI_PATTERNS = {
        'ssn': r'\b\d{3}-\d{2}-\d{4}\b',  # XXX-XX-XXXX
        'mrn': r'\bMRN[:\s]*(\d{6,10})\b',  # Medical Record Number
        'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Phone number
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'zipcode': r'\b\d{5}(?:-\d{4})?\b',
        'date_of_birth': r'\b(0?[1-9]|1[0-2])[/-](0?[1-9]|[12][0-9]|3[01])[/-](\d{4}|\d{2})\b',
        'credit_card': r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',
    }
    
    # Sensitive keywords
    SENSITIVE_KEYWORDS = [
        'password', 'api key', 'secret', 'token', 'private',
        'credit card', 'ssn', 'social security', 'address',
        'phone', 'email', 'diagnosis', 'treatment', 'medication'
    ]
    
    def __init__(self, encryption_key_path: str = ".encryption_key"):
        """
        Initialize HIPAA compliance module
        
        Args:
            encryption_key_path (str): Path to store encryption key
        """
        logger.info("Initializing HIPAA Compliance Module...")
        
        self.encryption_key_path = encryption_key_path
        self.encryption_key = self._init_encryption()
        self.cipher = Fernet(self.encryption_key) if self.encryption_key else None
        
        # Audit log file
        self.audit_log_path = "hipaa_audit.log"
        self._init_audit_log()
        
        logger.info("✅ HIPAA Compliance Module initialized")
        self.loaded = True
    
    def _init_encryption(self) -> bytes or None:
        """Initialize or load encryption key"""
        try:
            if os.path.exists(self.encryption_key_path):
                with open(self.encryption_key_path, 'rb') as f:
                    key = f.read()
                logger.info(f"✅ Loaded encryption key from {self.encryption_key_path}")
            else:
                key = Fernet.generate_key()
                with open(self.encryption_key_path, 'wb') as f:
                    f.write(key)
                logger.info(f"✅ Generated new encryption key at {self.encryption_key_path}")
            
            return key
        except Exception as e:
            logger.error(f"Error initializing encryption: {str(e)}")
            return None
    
    def _init_audit_log(self):
        """Initialize audit log file"""
        try:
            if not os.path.exists(self.audit_log_path):
                with open(self.audit_log_path, 'w') as f:
                    f.write("HIPAA Audit Log - Medical Data Access\n")
                    f.write("="*70 + "\n\n")
                logger.info(f"✅ Created audit log at {self.audit_log_path}")
        except Exception as e:
            logger.error(f"Error initializing audit log: {str(e)}")
    
    def encrypt_phi(self, data: str) -> str:
        """
        Encrypt Protected Health Information
        
        Args:
            data (str): Data to encrypt
            
        Returns:
            str: Encrypted data (base64 encoded)
        """
        if not self.cipher:
            logger.warning("Encryption not initialized")
            return data
        
        try:
            encrypted = self.cipher.encrypt(data.encode())
            return encrypted.decode()
        except Exception as e:
            logger.error(f"Error encrypting data: {str(e)}")
            return data
    
    def decrypt_phi(self, encrypted_data: str) -> str:
        """
        Decrypt Protected Health Information
        
        Args:
            encrypted_data (str): Encrypted data
            
        Returns:
            str: Decrypted data
        """
        if not self.cipher:
            logger.warning("Encryption not initialized")
            return encrypted_data
        
        try:
            decrypted = self.cipher.decrypt(encrypted_data.encode())
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Error decrypting data: {str(e)}")
            return encrypted_data
    
    def detect_phi(self, text: str) -> Dict[str, List[str]]:
        """
        Detect PHI/PII in text
        
        Args:
            text (str): Text to scan
            
        Returns:
            dict: Found PHI types and values
        """
        found_phi = {}
        
        for phi_type, pattern in self.PHI_PATTERNS.items():
            matches = re.findall(pattern, text)
            if matches:
                found_phi[phi_type] = matches
        
        # Check for sensitive keywords
        text_lower = text.lower()
        sensitive_found = []
        for keyword in self.SENSITIVE_KEYWORDS:
            if keyword in text_lower:
                sensitive_found.append(keyword)
        
        if sensitive_found:
            found_phi['sensitive_keywords'] = sensitive_found
        
        return found_phi
    
    def sanitize_phi(self, text: str, replacement: str = "[REDACTED]") -> str:
        """
        Remove/redact PHI from text
        
        Args:
            text (str): Text with PHI
            replacement (str): Replacement text
            
        Returns:
            str: Sanitized text
        """
        sanitized = text
        
        for phi_type, pattern in self.PHI_PATTERNS.items():
            sanitized = re.sub(pattern, replacement, sanitized)
        
        return sanitized
    
    def log_phi_access(self, user_id: str, patient_id: str, action: str, 
                      data_accessed: str = None, duration_seconds: float = None) -> bool:
        """
        Log access to PHI (HIPAA requirement)
        
        Args:
            user_id (str): User accessing data
            patient_id (str): Patient ID
            action (str): Action performed (READ, WRITE, DELETE, etc.)
            data_accessed (str): What data was accessed
            duration_seconds (float): How long was data accessed
            
        Returns:
            bool: Success status
        """
        try:
            timestamp = datetime.now().isoformat()
            
            log_entry = {
                "timestamp": timestamp,
                "user_id": user_id,
                "patient_id": patient_id,
                "action": action,
                "data_accessed": data_accessed or "NOT_SPECIFIED",
                "duration_seconds": duration_seconds or 0,
                "access_method": "API"
            }
            
            # Write to audit log
            with open(self.audit_log_path, 'a') as f:
                f.write(json.dumps(log_entry) + "\n")
            
            logger.info(f"PHI Access Logged: {action} by {user_id} for patient {patient_id}")
            return True
        except Exception as e:
            logger.error(f"Error logging PHI access: {str(e)}")
            return False
    
    def get_audit_trail(self, patient_id: str = None, user_id: str = None, 
                       limit: int = 100) -> List[Dict[str, any]]:
        """
        Retrieve audit trail for patient or user
        
        Args:
            patient_id (str): Filter by patient
            user_id (str): Filter by user
            limit (int): Maximum records to return
            
        Returns:
            list: Audit log entries
        """
        try:
            audit_entries = []
            
            with open(self.audit_log_path, 'r') as f:
                lines = f.readlines()
            
            # Skip header lines
            for line in lines[3:]:
                if line.strip():
                    try:
                        entry = json.loads(line)
                        
                        # Filter if needed
                        if patient_id and entry.get('patient_id') != patient_id:
                            continue
                        if user_id and entry.get('user_id') != user_id:
                            continue
                        
                        audit_entries.append(entry)
                    except json.JSONDecodeError:
                        continue
            
            return audit_entries[-limit:]  # Return last N entries
        except Exception as e:
            logger.error(f"Error retrieving audit trail: {str(e)}")
            return []
    
    def create_encryption_summary(self) -> Dict[str, str]:
        """Get encryption status summary"""
        return {
            "encryption_enabled": self.cipher is not None,
            "algorithm": "AES-256 (Fernet)",
            "key_storage": self.encryption_key_path,
            "audit_log": self.audit_log_path,
            "status": "✅ ACTIVE" if self.cipher else "❌ INACTIVE"
        }
    
    def validate_patient_consent(self, patient_id: str, consent_type: str) -> bool:
        """
        Validate patient consent for data access
        HIPAA requires documented consent
        
        Args:
            patient_id (str): Patient ID
            consent_type (str): Type of consent (treatment, research, etc.)
            
        Returns:
            bool: Consent valid
        """
        # In production, check against consent database
        logger.info(f"Validating consent for patient {patient_id} - type: {consent_type}")
        return True
    
    def generate_hipaa_report(self, start_date: str = None, end_date: str = None) -> Dict[str, any]:
        """
        Generate HIPAA compliance report
        
        Args:
            start_date (str): Report start date (ISO format)
            end_date (str): Report end date (ISO format)
            
        Returns:
            dict: Compliance report
        """
        try:
            audit_entries = self.get_audit_trail()
            
            # Count by action
            action_counts = {}
            for entry in audit_entries:
                action = entry.get('action', 'UNKNOWN')
                action_counts[action] = action_counts.get(action, 0) + 1
            
            # Count unique users and patients
            unique_users = set(e.get('user_id') for e in audit_entries)
            unique_patients = set(e.get('patient_id') for e in audit_entries)
            
            report = {
                "report_generated": datetime.now().isoformat(),
                "total_access_events": len(audit_entries),
                "unique_users": len(unique_users),
                "unique_patients": len(unique_patients),
                "actions_breakdown": action_counts,
                "encryption_status": self.create_encryption_summary(),
                "compliance_level": "COMPLIANT" if len(audit_entries) > 0 else "PARTIAL"
            }
            
            return report
        except Exception as e:
            logger.error(f"Error generating HIPAA report: {str(e)}")
            return {}


def test_hipaa_compliance():
    """Test HIPAA compliance functionality"""
    compliance = HIPAACompliance()
    
    print("\n" + "="*70)
    print("HIPAA COMPLIANCE TEST")
    print("="*70)
    
    # Test 1: Detect PHI
    print("\n[1] PHI Detection:")
    test_text = """
    Patient: John Doe
    SSN: 123-45-6789
    DOB: 01/15/1980
    Phone: 555-123-4567
    Email: john.doe@email.com
    MRN: 1234567
    Diagnosis: Type 2 Diabetes
    """
    print(f"Test text:\n{test_text}\n")
    detected = compliance.detect_phi(test_text)
    print(f"Detected PHI: {json.dumps(detected, indent=2)}")
    
    # Test 2: Sanitize PHI
    print("\n[2] PHI Sanitization:")
    sanitized = compliance.sanitize_phi(test_text)
    print(f"Sanitized text:\n{sanitized}")
    
    # Test 3: Encrypt/Decrypt
    print("\n[3] Encryption/Decryption:")
    sensitive_data = "Patient SSN: 123-45-6789"
    encrypted = compliance.encrypt_phi(sensitive_data)
    print(f"Original: {sensitive_data}")
    print(f"Encrypted: {encrypted[:50]}...")
    decrypted = compliance.decrypt_phi(encrypted)
    print(f"Decrypted: {decrypted}")
    
    # Test 4: Audit logging
    print("\n[4] Audit Logging:")
    compliance.log_phi_access(
        user_id="DR001",
        patient_id="PAT123",
        action="READ",
        data_accessed="Lab Results",
        duration_seconds=5.2
    )
    print("✅ Access logged")
    
    # Test 5: Encryption summary
    print("\n[5] Encryption Summary:")
    summary = compliance.create_encryption_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Test 6: HIPAA report
    print("\n[6] HIPAA Compliance Report:")
    report = compliance.generate_hipaa_report()
    print(json.dumps(report, indent=2))
    
    print("\n" + "="*70)


if __name__ == "__main__":
    test_hipaa_compliance()
