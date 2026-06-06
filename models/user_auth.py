"""
User Authentication System for AI Doctor
Handles login, signup, and session management
"""

import hashlib
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict
import secrets

class UserAuthSystem:
    def __init__(self):
        self.users_dir = Path("./users_db")
        self.users_dir.mkdir(exist_ok=True)
        self.users_file = self.users_dir / "users.json"
        self.sessions_file = self.users_dir / "sessions.json"
        self.load_users()
        self.load_sessions()
    
    def load_users(self):
        """Load existing users from file"""
        if self.users_file.exists():
            with open(self.users_file, 'r') as f:
                self.users = json.load(f)
        else:
            self.users = {}
            self.save_users()
    
    def load_sessions(self):
        """Load active sessions"""
        if self.sessions_file.exists():
            with open(self.sessions_file, 'r') as f:
                self.sessions = json.load(f)
        else:
            self.sessions = {}
            self.save_sessions()
    
    def save_users(self):
        """Save users to file"""
        with open(self.users_file, 'w') as f:
            json.dump(self.users, f, indent=2)
    
    def save_sessions(self):
        """Save sessions to file"""
        with open(self.sessions_file, 'w') as f:
            json.dump(self.sessions, f, indent=2)
    
    def hash_password(self, password: str) -> str:
        """Hash password with SHA256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def create_user(self, username: str, password: str, email: str, full_name: str = "") -> Dict:
        """Create new user account"""
        if username in self.users:
            return {"success": False, "message": "Username already exists"}
        
        if len(password) < 6:
            return {"success": False, "message": "Password must be at least 6 characters"}
        
        # Create user
        self.users[username] = {
            "password_hash": self.hash_password(password),
            "email": email,
            "full_name": full_name,
            "created_at": datetime.now().isoformat(),
            "patient_id": f"patient_{username}_{datetime.now().strftime('%Y%m%d')}",
            "medical_history": [],
            "preferences": {}
        }
        
        self.save_users()
        return {"success": True, "message": "Account created successfully", "patient_id": self.users[username]["patient_id"]}
    
    def login(self, username: str, password: str) -> Dict:
        """Authenticate user"""
        if username not in self.users:
            return {"success": False, "message": "Invalid username or password"}
        
        if self.users[username]["password_hash"] != self.hash_password(password):
            return {"success": False, "message": "Invalid username or password"}
        
        # Create session token
        session_token = secrets.token_hex(32)
        self.sessions[session_token] = {
            "username": username,
            "login_time": datetime.now().isoformat(),
            "expires": (datetime.now() + timedelta(hours=24)).isoformat()
        }
        
        self.save_sessions()
        
        return {
            "success": True,
            "message": "Login successful",
            "session_token": session_token,
            "user_data": {
                "username": username,
                "email": self.users[username]["email"],
                "full_name": self.users[username]["full_name"],
                "patient_id": self.users[username]["patient_id"]
            }
        }
    
    def validate_session(self, session_token: str) -> Optional[Dict]:
        """Validate session token and return user data"""
        if session_token not in self.sessions:
            return None
        
        session = self.sessions[session_token]
        if datetime.fromisoformat(session["expires"]) < datetime.now():
            # Session expired
            del self.sessions[session_token]
            self.save_sessions()
            return None
        
        username = session["username"]
        return {
            "username": username,
            "email": self.users[username]["email"],
            "full_name": self.users[username]["full_name"],
            "patient_id": self.users[username]["patient_id"]
        }
    
    def logout(self, session_token: str):
        """Logout user by removing session"""
        if session_token in self.sessions:
            del self.sessions[session_token]
            self.save_sessions()
            return True
        return False
    
    def get_user_history(self, username: str) -> list:
        """Get user's medical history"""
        if username in self.users:
            return self.users[username].get("medical_history", [])
        return []
    
    def add_to_history(self, username: str, record: Dict):
        """Add record to user's medical history"""
        if username in self.users:
            if "medical_history" not in self.users[username]:
                self.users[username]["medical_history"] = []
            
            record["timestamp"] = datetime.now().isoformat()
            self.users[username]["medical_history"].append(record)
            self.save_users()
            return True
        return False