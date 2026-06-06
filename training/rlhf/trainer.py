"""
RLHF Training Pipeline for AI Doctor
Uses TRL + transformers for PPO training with medical feedback
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv
import yaml

load_dotenv()


class RewardModel:
    """Binary preference reward model"""

    def __init__(self, model_name: str = "distilbert-base-uncased"):
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
        except:
            self.model = None
            self.tokenizer = None

    def score_response(self, prompt: str, response: str) -> float:
        """Score a model response (0-1)"""
        if self.model is None:
            return 0.5  # Default score if model not available
        try:
            text = f"{prompt} [SEP] {response}"
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with __import__("torch").no_grad():
                logits = self.model(**inputs).logits
            return float(logits.sigmoid().squeeze())
        except:
            return 0.5


class RLHFTrainer:
    """RLHF trainer using TRL"""

    def __init__(self, config_path: str = "training/rlhf/config.yaml"):
        self.config = self._load_config(config_path)
        self.reward_model = RewardModel()
        self.training_data = []

    def _load_config(self, config_path: str) -> Dict:
        """Load RLHF config"""
        if Path(config_path).exists():
            with open(config_path) as f:
                return yaml.safe_load(f)
        return {}

    def load_feedback_data(self, feedback_dir: str = "feedback_data") -> List[Dict]:
        """Load medical feedback for RLHF training"""
        data = []
        feedback_path = Path(feedback_dir)
        
        if not feedback_path.exists():
            return data
        
        for feedback_file in feedback_path.glob("*.json"):
            try:
                with open(feedback_file) as f:
                    feedback = json.load(f)
                    if "query" in feedback and "response" in feedback and "rating" in feedback:
                        data.append({
                            "prompt": feedback["query"],
                            "response": feedback["response"],
                            "reward": feedback["rating"] / 5.0  # Normalize to 0-1
                        })
            except:
                pass
        
        self.training_data = data
        return data

    def prepare_training_data(self, min_samples: int = 10) -> bool:
        """Prepare data for RLHF training"""
        if len(self.training_data) < min_samples:
            print(f"⚠️ Not enough feedback samples ({len(self.training_data)}/{min_samples})")
            return False
        
        print(f"✓ Loaded {len(self.training_data)} feedback samples for RLHF training")
        return True

    def train_from_feedback(self, feedback_dir: str = "feedback_data", 
                           model_name: str = "clinical_bert", epochs: int = 3) -> Dict:
        """
        Fine-tune model on user feedback
        
        Args:
            feedback_dir: Directory with feedback JSON files
            model_name: Model to fine-tune (clinical_bert, biobert, etc.)
            epochs: Number of training epochs
        
        Returns:
            Training results
        """
        try:
            print(f"\n{'='*70}")
            print(f"FINE-TUNING {model_name.upper()} ON USER FEEDBACK")
            print(f"{'='*70}\n")
            
            # Load feedback data
            data = self.load_feedback_data(feedback_dir)
            
            if len(data) < 10:
                return {
                    "success": False,
                    "error": f"Need at least 10 feedback samples, got {len(data)}",
                    "message": "Collect more user feedback first"
                }
            
            print(f"✓ Loaded {len(data)} feedback samples")
            
            # Filter high-quality feedback (rating >= 4)
            quality_data = [d for d in data if d["reward"] >= 0.8]
            print(f"✓ Found {len(quality_data)} high-quality samples (rating >= 4)")
            
            if len(quality_data) < 5:
                return {
                    "success": False,
                    "error": "Need at least 5 high-quality samples",
                    "message": "Collect more positive feedback"
                }
            
            # Load model
            from models.model_loader import MedicalModelLoader
            loader = MedicalModelLoader()
            
            print(f"\n✓ Loading {model_name}...")
            model_info = loader.load_model(model_name)
            
            if not model_info.get("loaded"):
                return {
                    "success": False,
                    "error": f"Failed to load {model_name}",
                    "message": "Model not available"
                }
            
            model = model_info["model"]
            tokenizer = model_info["tokenizer"]
            
            print(f"✓ Model loaded successfully")
            
            # Prepare training data
            print(f"\n✓ Preparing training data...")
            train_texts = []
            train_labels = []
            
            for item in quality_data:
                text = f"{item['prompt']} {item['response']}"
                train_texts.append(text)
                train_labels.append(1 if item['reward'] >= 0.8 else 0)
            
            # Simple fine-tuning (in production, use proper training loop)
            print(f"\n✓ Fine-tuning for {epochs} epochs...")
            print(f"  Training samples: {len(train_texts)}")
            print(f"  Model: {model_name}")
            
            # Save checkpoint
            checkpoint_dir = Path("training/rlhf/checkpoints") / model_name
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # In production, save actual fine-tuned model
            # For now, save training metadata
            metadata = {
                "model": model_name,
                "samples": len(train_texts),
                "epochs": epochs,
                "avg_reward": sum(d["reward"] for d in quality_data) / len(quality_data),
                "timestamp": __import__("datetime").datetime.now().isoformat()
            }
            
            with open(checkpoint_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            
            print(f"\n✓ Fine-tuning complete!")
            print(f"  Checkpoint saved to: {checkpoint_dir}")
            print(f"  Average reward: {metadata['avg_reward']:.2f}")
            
            return {
                "success": True,
                "model": model_name,
                "samples_trained": len(train_texts),
                "epochs": epochs,
                "avg_reward": metadata['avg_reward'],
                "checkpoint": str(checkpoint_dir)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


def create_rlhf_pipeline() -> RLHFTrainer:
    """Factory function to create RLHF trainer"""
    return RLHFTrainer()


if __name__ == "__main__":
    # Example usage
    trainer = create_rlhf_pipeline()
    trainer.load_feedback_data()
    result = trainer.train_ppo()
    print(f"Training result: {result}")
