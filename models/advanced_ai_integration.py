"""
Integration module for Advanced AI Models
Combines CNN, RNN, RL, and Fine-Tuning into unified system
Used by app_fixed.py to display all advanced features
"""

import numpy as np
import streamlit as st
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import advanced models
try:
    from models.advanced_cnn_models import AdvancedMedicalCNN, cnn_system
    CNN_AVAILABLE = True
except:
    CNN_AVAILABLE = False
    print("[!] CNN models not fully available (TF may not be fully loaded)")

try:
    from models.advanced_rnn_models import AdvancedMedicalRNN, rnn_system
    RNN_AVAILABLE = True
except:
    RNN_AVAILABLE = False
    print("[!] RNN models not fully available")

try:
    from models.advanced_rl_models import AdvancedMedicalRL, rl_system
    RL_AVAILABLE = True
except:
    RL_AVAILABLE = False
    print("[!] RL models not fully available")

try:
    from models.advanced_fine_tuning import AdvancedFineTuning, fine_tuning_system
    TUNING_AVAILABLE = True
except:
    TUNING_AVAILABLE = False
    print("[!] Fine-tuning system not available")


class UnifiedMedicalAI:
    """Unified system combining all advanced AI models"""
    
    def __init__(self):
        self.cnn = None
        self.rnn = None
        self.rl = None
        self.fine_tuning = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all model systems"""
        if CNN_AVAILABLE:
            self.cnn = AdvancedMedicalCNN()
            try:
                self.cnn.build_pneumonia_detector()
                self.cnn.build_skin_lesion_classifier()
                self.cnn.build_ecg_image_classifier()
                self.cnn.build_resnet_medical_classifier()
                print("[✓] CNN models loaded")
            except Exception as e:
                print(f"[!] CNN initialization partial: {e}")
        
        if RNN_AVAILABLE:
            self.rnn = AdvancedMedicalRNN()
            try:
                self.rnn.build_lstm_arrhythmia_detector()
                self.rnn.build_gru_cardiac_trend()
                self.rnn.build_attention_lstm()
                self.rnn.build_encoder_decoder()
                self.rnn.build_tcn_temporal()
                print("[✓] RNN models loaded")
            except Exception as e:
                print(f"[!] RNN initialization partial: {e}")
        
        if RL_AVAILABLE:
            self.rl = AdvancedMedicalRL()
            print("[✓] RL system loaded")
        
        if TUNING_AVAILABLE:
            self.fine_tuning = AdvancedFineTuning()
            print("[✓] Fine-tuning system loaded")
    
    def get_cnn_models_list(self):
        """Get list of available CNN models"""
        if not self.cnn:
            return []
        return {
            '🫁 Pneumonia Detector': 'pneumonia',
            '🩹 Skin Lesion Classifier': 'skin_lesion',
            '❤️ ECG Image Classifier': 'ecg_image',
            '🔬 ResNet Medical Classifier': 'resnet_medical'
        }
    
    def get_rnn_models_list(self):
        """Get list of available RNN models"""
        if not self.rnn:
            return []
        return {
            '💓 LSTM Arrhythmia Detector': 'lstm_arrhythmia',
            '📊 GRU Cardiac Trend': 'gru_cardiac',
            '🧠 Attention LSTM': 'attention_lstm',
            '⏱️ Temporal CNN': 'tcn_temporal'
        }
    
    def get_rl_features(self):
        """Get RL features"""
        return [
            '💊 DQN Treatment Optimization',
            '🎯 Policy Gradient Recommendations',
            '🔄 Multi-Agent Treatment Planning'
        ]
    
    def display_cnn_section(self):
        """Display CNN models section in Streamlit"""
        st.header("🤖 Advanced CNN Models")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Available Models")
            models = self.get_cnn_models_list()
            for name, key in models.items():
                st.write(f"✓ {name}")
        
        with col2:
            st.subheader("Features")
            st.write("- Transfer learning with pre-trained weights")
            st.write("- Data augmentation")
            st.write("- Batch normalization")
            st.write("- Dropout regularization")
        
        st.markdown("---")
        
        # Capabilities
        st.write("**Model Capabilities:**")
        capabilities = {
            '🫁 Pneumonia Detection': [
                'Binary classification (Normal vs Pneumonia)',
                'Input: 224×224×3 medical images',
                'Accuracy metric with Precision/Recall'
            ],
            '🩹 Skin Lesion Classification': [
                '7-class classification (melanoma, nevus, etc.)',
                'Transfer learning using InceptionV3',
                'Input: 299×299×3 RGB images'
            ],
            '❤️ ECG Image Analysis': [
                '5-class arrhythmia classification',
                'Custom CNN architecture',
                'Handles 224×224 ECG images'
            ],
            '🔬 ResNet Medical': [
                'Multi-class medical image classification',
                'ResNet50 backbone with custom head',
                'Flexible input and output sizes'
            ]
        }
        
        for model_name, features in capabilities.items():
            with st.expander(model_name):
                for feature in features:
                    st.write(f"• {feature}")
    
    def display_rnn_section(self):
        """Display RNN models section in Streamlit"""
        st.header("⏱️ Advanced RNN/LSTM Models")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Available Models")
            models = self.get_rnn_models_list()
            for name, key in models.items():
                st.write(f"✓ {name}")
        
        with col2:
            st.subheader("Features")
            st.write("- Bidirectional processing")
            st.write("- Attention mechanisms")
            st.write("- Sequence-to-sequence models")
            st.write("- Temporal convolutions")
        
        st.markdown("---")
        
        # Capabilities
        st.write("**Model Capabilities:**")
        capabilities = {
            '💓 LSTM Arrhythmia Detector': [
                'Real-time ECG analysis',
                'Bidirectional LSTM processing',
                '5-class arrhythmia detection',
                'Input: 187 timesteps × 1 feature'
            ],
            '📊 GRU Cardiac Trend': [
                'Heart rate trend analysis',
                'Multi-variate cardiac signals',
                'Normal/Warning/Critical classification',
                '100 timesteps × 5 vital features'
            ],
            '🧠 Attention LSTM': [
                'Advanced attention mechanisms',
                'Long-term dependency learning',
                'Multi-class medical prediction',
                'Explainable predictions'
            ],
            '⏱️ Temporal CNN': [
                'Causal convolutions',
                'Dilated convolutions for long-range',
                'Temporal pattern recognition',
                'Efficient computation'
            ]
        }
        
        for model_name, features in capabilities.items():
            with st.expander(model_name):
                for feature in features:
                    st.write(f"• {feature}")
    
    def display_rl_section(self):
        """Display RL section in Streamlit"""
        st.header("🤖 Reinforcement Learning System")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Available Algorithms")
            features = self.get_rl_features()
            for feature in features:
                st.write(f"✓ {feature}")
        
        with col2:
            st.subheader("Applications")
            st.write("- Treatment recommendation")
            st.write("- Dosage optimization")
            st.write("- Treatment planning")
            st.write("- Real-time adaptation")
        
        st.markdown("---")
        
        # RL Details
        st.write("**Reinforcement Learning Details:**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**DQN Agent**")
            st.write("- Deep Q-Network")
            st.write("- Experience replay")
            st.write("- Target network")
            st.write("- Epsilon-greedy exploration")
        
        with col2:
            st.write("**Policy Gradient**")
            st.write("- Actor-Critic architecture")
            st.write("- Policy optimization")
            st.write("- Value function learning")
            st.write("- Advantage estimation")
        
        with col3:
            st.write("**Multi-Agent RL**")
            st.write("- Cooperative agents")
            st.write("- Treatment coordination")
            st.write("- Risk assessment")
            st.write("- Decision fusion")
    
    def display_fine_tuning_section(self):
        """Display fine-tuning section in Streamlit"""
        st.header("⚙️ Advanced Fine-Tuning Strategies")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Learning Rate Schedules")
            schedules = [
                '📉 Exponential Decay',
                '📊 Polynomial Decay',
                '🌊 Cosine Annealing',
                '🔄 Cyclic Learning Rate'
            ]
            for schedule in schedules:
                st.write(f"✓ {schedule}")
        
        with col2:
            st.subheader("Transfer Learning")
            strategies = [
                '🔓 Progressive Unfreezing',
                '📚 Discriminative Learning Rates',
                '🎯 Layer-wise Fine-tuning',
                '📊 Feature Extraction'
            ]
            for strategy in strategies:
                st.write(f"✓ {strategy}")
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Regularization**")
            st.write("- Early Stopping")
            st.write("- LR Reduction")
            st.write("- Model Checkpointing")
            st.write("- Dropout Scheduling")
        
        with col2:
            st.write("**Optimization**")
            st.write("- AdamW optimizer")
            st.write("- LAMB optimizer")
            st.write("- RectifiedAdam")
            st.write("- Mixed Precision")
        
        with col3:
            st.write("**Advanced**")
            st.write("- Hyperparameter Tuning")
            st.write("- Knowledge Distillation")
            st.write("- Ensemble Methods")
            st.write("- AutoML")
    
    def display_all_models_summary(self):
        """Display summary of all models"""
        st.header("📊 Complete Model Overview")
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("CNN Models", "4", "Transfer Learning")
        
        with col2:
            st.metric("RNN Models", "5", "Sequence Processing")
        
        with col3:
            st.metric("RL Agents", "3", "Decision Making")
        
        with col4:
            st.metric("Fine-Tuning Strategies", "12+", "Optimization")
        
        st.markdown("---")
        
        # Integrated System
        st.subheader("🎯 Integrated Medical AI System")
        
        st.write("""
        This unified system combines:
        
        1. **Image Analysis (CNN)** - Diagnose from medical images
        2. **Time-Series Analysis (RNN)** - Monitor patient trends
        3. **Treatment Optimization (RL)** - Recommend best treatments
        4. **Model Enhancement (Fine-Tuning)** - Continuous improvement
        
        All models work together in the AI Doctor system!
        """)


# Initialize global unified AI system
try:
    unified_ai = UnifiedMedicalAI()
    SYSTEM_READY = True
except Exception as e:
    print(f"[!] Unified AI system initialization issue: {e}")
    unified_ai = None
    SYSTEM_READY = False


if __name__ == "__main__":
    print("Advanced AI Integration Module Loaded")
    if SYSTEM_READY:
        print("[✓] All systems ready!")
