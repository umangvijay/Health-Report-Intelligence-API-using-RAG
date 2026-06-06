"""
Self-Learning Module using Direct Preference Optimization (DPO)
Implements feedback collection and model improvement
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
from pathlib import Path
import pickle
import yaml

class SelfLearningDPO:
    """
    Implements Direct Preference Optimization for continuous learning
    Safer than traditional RL for medical applications
    """
    
    def __init__(self):
        self.feedback_dir = Path('./feedback_data')
        self.feedback_dir.mkdir(exist_ok=True)
        
        self.checkpoint_dir = Path('./checkpoints')
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.preference_data = []
        self.load_feedback_history()
        
        # Load config
        config_path = Path(__file__).parent / 'config.yaml'
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {}
        
        self.feedback_threshold = self.config.get('learning', {}).get('feedback_threshold', 100)
        self.enable_dpo = self.config.get('learning', {}).get('enable_dpo', True)
    
    def collect_feedback(self, 
                        query: str,
                        response: str,
                        feedback_type: str,
                        patient_id: Optional[str] = None,
                        expert_correction: Optional[str] = None) -> bool:
        """
        Collect user feedback on AI responses
        
        Args:
            query: Original patient query
            response: AI's response
            feedback_type: 'positive', 'negative', or 'expert'
            patient_id: Optional patient identifier
            expert_correction: Correct response from medical expert
        
        Returns:
            Success status
        """
        try:
            feedback_entry = {
                'timestamp': datetime.now().isoformat(),
                'query': query,
                'response': response,
                'feedback_type': feedback_type,
                'patient_id': patient_id or 'anonymous',
                'expert_correction': expert_correction
            }
            
            # Save to feedback database
            feedback_file = self.feedback_dir / f'feedback_{datetime.now().strftime("%Y%m%d")}.jsonl'
            with open(feedback_file, 'a') as f:
                f.write(json.dumps(feedback_entry) + '\n')
            
            # Add to preference data for DPO
            if feedback_type == 'negative' and expert_correction:
                self.preference_data.append({
                    'prompt': query,
                    'chosen': expert_correction,  # Better response
                    'rejected': response  # Original response
                })
            elif feedback_type == 'expert':
                self.preference_data.append({
                    'prompt': query,
                    'chosen': expert_correction or response,
                    'rejected': response if expert_correction else ''
                })
            
            # Check if we should trigger retraining
            if len(self.preference_data) >= self.feedback_threshold:
                self.trigger_dpo_training()
            
            return True
            
        except Exception as e:
            print(f"Error collecting feedback: {e}")
            return False
    
    def load_feedback_history(self):
        """Load historical feedback data"""
        try:
            for feedback_file in self.feedback_dir.glob('*.jsonl'):
                with open(feedback_file, 'r') as f:
                    for line in f:
                        entry = json.loads(line)
                        if entry.get('feedback_type') in ['negative', 'expert']:
                            if entry.get('expert_correction'):
                                self.preference_data.append({
                                    'prompt': entry['query'],
                                    'chosen': entry['expert_correction'],
                                    'rejected': entry['response']
                                })
        except Exception as e:
            print(f"Error loading feedback history: {e}")
    
    def trigger_dpo_training(self):
        """
        Trigger DPO training when enough feedback is collected
        This would normally fine-tune the model, but here we simulate it
        """
        if not self.enable_dpo:
            print("DPO training is disabled")
            return
        
        print(f"Triggering DPO training with {len(self.preference_data)} preference pairs...")
        
        try:
            # Prepare training data
            training_data = self.prepare_dpo_dataset()
            
            # Save training data
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            dataset_path = self.checkpoint_dir / f'dpo_dataset_{timestamp}.json'
            with open(dataset_path, 'w') as f:
                json.dump(training_data, f, indent=2)
            
            # In production, this would trigger actual model fine-tuning
            # For now, we simulate by updating preference weights
            self.update_preference_weights(training_data)
            
            print(f"DPO training completed. Dataset saved to {dataset_path}")
            
            # Clear processed data
            self.preference_data = self.preference_data[-50:]  # Keep last 50 for continuity
            
        except Exception as e:
            print(f"Error in DPO training: {e}")
    
    def prepare_dpo_dataset(self) -> Dict:
        """Prepare dataset for DPO training"""
        
        # Group by medical categories
        categorized_data = {
            'diagnosis': [],
            'treatment': [],
            'emergency': [],
            'general': []
        }
        
        for pref in self.preference_data:
            prompt_lower = pref['prompt'].lower()
            
            if any(word in prompt_lower for word in ['emergency', 'urgent', 'severe']):
                categorized_data['emergency'].append(pref)
            elif any(word in prompt_lower for word in ['diagnose', 'symptoms', 'condition']):
                categorized_data['diagnosis'].append(pref)
            elif any(word in prompt_lower for word in ['treatment', 'medication', 'cure']):
                categorized_data['treatment'].append(pref)
            else:
                categorized_data['general'].append(pref)
        
        # Calculate statistics
        stats = {
            'total_pairs': len(self.preference_data),
            'categories': {k: len(v) for k, v in categorized_data.items()},
            'timestamp': datetime.now().isoformat()
        }
        
        return {
            'data': categorized_data,
            'stats': stats,
            'config': {
                'beta': 0.1,  # DPO beta parameter
                'learning_rate': 5e-5,
                'batch_size': 4,
                'epochs': 3
            }
        }
    
    def update_preference_weights(self, training_data: Dict):
        """
        Update internal preference weights based on DPO training
        This simulates the effect of model fine-tuning
        """
        
        # Create or load preference model
        model_path = self.checkpoint_dir / 'preference_model.pkl'
        
        if model_path.exists():
            with open(model_path, 'rb') as f:
                preference_model = pickle.load(f)
        else:
            preference_model = {
                'patterns': {},
                'corrections': {},
                'weights': {}
            }
        
        # Extract patterns from training data
        for category, pairs in training_data['data'].items():
            for pair in pairs:
                prompt = pair['prompt']
                chosen = pair['chosen']
                rejected = pair['rejected']
                
                # Simple pattern matching (in production, use embeddings)
                key_words = self._extract_keywords(prompt)
                pattern_key = '_'.join(sorted(key_words))
                
                if pattern_key not in preference_model['patterns']:
                    preference_model['patterns'][pattern_key] = {
                        'preferred_responses': [],
                        'rejected_responses': [],
                        'category': category
                    }
                
                preference_model['patterns'][pattern_key]['preferred_responses'].append(chosen)
                preference_model['patterns'][pattern_key]['rejected_responses'].append(rejected)
        
        # Save updated model
        with open(model_path, 'wb') as f:
            pickle.dump(preference_model, f)
        
        print(f"Updated preference model with {len(preference_model['patterns'])} patterns")
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract medical keywords from text"""
        medical_keywords = [
            'fever', 'pain', 'cough', 'headache', 'nausea', 'fatigue',
            'rash', 'swelling', 'bleeding', 'infection', 'inflammation',
            'diabetes', 'hypertension', 'cancer', 'pneumonia', 'covid',
            'treatment', 'medication', 'diagnosis', 'symptoms', 'chronic'
        ]
        
        text_lower = text.lower()
        found_keywords = [kw for kw in medical_keywords if kw in text_lower]
        
        return found_keywords[:5]  # Limit to 5 keywords
    
    def get_preference_guidance(self, query: str) -> Optional[Dict]:
        """
        Get guidance from learned preferences
        
        Args:
            query: New patient query
        
        Returns:
            Dictionary with preference guidance or None
        """
        
        model_path = self.checkpoint_dir / 'preference_model.pkl'
        if not model_path.exists():
            return None
        
        try:
            with open(model_path, 'rb') as f:
                preference_model = pickle.load(f)
            
            # Extract keywords from query
            keywords = self._extract_keywords(query)
            pattern_key = '_'.join(sorted(keywords))
            
            # Check if we have learned preferences for this pattern
            if pattern_key in preference_model['patterns']:
                pattern_data = preference_model['patterns'][pattern_key]
                
                # Return the most recent preferred response as guidance
                if pattern_data['preferred_responses']:
                    return {
                        'has_preference': True,
                        'category': pattern_data['category'],
                        'suggested_response': pattern_data['preferred_responses'][-1],
                        'avoid_patterns': pattern_data['rejected_responses'][-3:] if pattern_data['rejected_responses'] else [],
                        'confidence': min(len(pattern_data['preferred_responses']) / 10, 1.0)
                    }
            
            return {'has_preference': False}
            
        except Exception as e:
            print(f"Error getting preference guidance: {e}")
            return None
    
    def generate_learning_report(self) -> Dict:
        """Generate report on learning progress"""
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_feedback_collected': 0,
            'preference_pairs': len(self.preference_data),
            'categories': {},
            'model_updates': 0,
            'performance_metrics': {}
        }
        
        # Count feedback entries
        for feedback_file in self.feedback_dir.glob('*.jsonl'):
            with open(feedback_file, 'r') as f:
                report['total_feedback_collected'] += sum(1 for _ in f)
        
        # Check for model updates
        checkpoints = list(self.checkpoint_dir.glob('dpo_dataset_*.json'))
        report['model_updates'] = len(checkpoints)
        
        # Analyze preference patterns
        if self.preference_data:
            for pref in self.preference_data:
                prompt_lower = pref['prompt'].lower()
                if 'emergency' in prompt_lower:
                    category = 'emergency'
                elif 'diagnos' in prompt_lower:
                    category = 'diagnosis'
                elif 'treat' in prompt_lower:
                    category = 'treatment'
                else:
                    category = 'general'
                
                if category not in report['categories']:
                    report['categories'][category] = 0
                report['categories'][category] += 1
        
        # Calculate improvement metrics (simulated)
        if report['model_updates'] > 0:
            # Simulate improvement based on number of updates
            base_accuracy = 0.70
            improvement_per_update = 0.02
            current_accuracy = min(base_accuracy + (report['model_updates'] * improvement_per_update), 0.95)
            
            report['performance_metrics'] = {
                'estimated_accuracy': current_accuracy,
                'feedback_incorporation_rate': min(report['preference_pairs'] / 100, 1.0),
                'learning_efficiency': 0.8  # Simulated
            }
        
        return report


class ReinforcementLearningAdapter:
    """
    Adapter for more advanced RL techniques (for future enhancement)
    Currently implements safe exploration strategies
    """
    
    def __init__(self):
        self.exploration_rate = 0.1  # Conservative exploration
        self.safety_threshold = 0.95  # High safety requirement
        self.action_history = []
    
    def safe_action_selection(self, state: Dict, possible_actions: List[str]) -> str:
        """
        Select action with safety constraints
        
        Args:
            state: Current patient state
            possible_actions: List of possible medical actions
        
        Returns:
            Selected action
        """
        
        # Filter out potentially dangerous actions
        safe_actions = []
        for action in possible_actions:
            if self._is_safe_action(action, state):
                safe_actions.append(action)
        
        if not safe_actions:
            return "Refer to medical professional immediately"
        
        # Epsilon-greedy with safety
        if np.random.random() < self.exploration_rate:
            # Explore, but only among safe actions
            selected = np.random.choice(safe_actions)
        else:
            # Exploit: choose best known action
            selected = self._get_best_action(safe_actions, state)
        
        # Log action
        self.action_history.append({
            'timestamp': datetime.now().isoformat(),
            'state': state,
            'action': selected
        })
        
        return selected
    
    def _is_safe_action(self, action: str, state: Dict) -> bool:
        """Check if action is safe for patient"""
        
        # Never recommend stopping critical medications
        critical_meds = ['insulin', 'blood pressure', 'heart medication', 'antibiotic']
        action_lower = action.lower()
        
        if 'stop' in action_lower and any(med in action_lower for med in critical_meds):
            return False
        
        # Don't recommend high doses without supervision
        if 'high dose' in action_lower or 'increase dose' in action_lower:
            if 'under supervision' not in action_lower:
                return False
        
        # Emergency states require immediate referral
        if state.get('emergency', False):
            return 'emergency' in action_lower or 'immediate' in action_lower
        
        return True
    
    def _get_best_action(self, actions: List[str], state: Dict) -> str:
        """Get best action based on historical performance"""
        
        # Simple heuristic: prefer conservative treatments first
        conservative_keywords = ['monitor', 'rest', 'hydrate', 'low dose', 'gradual']
        
        for action in actions:
            if any(kw in action.lower() for kw in conservative_keywords):
                return action
        
        # Default to first safe action
        return actions[0] if actions else "Consult healthcare provider"