"""
Advanced RNN/LSTM Models for Medical Time-Series Analysis
- Cardiac arrhythmia detection from ECG sequences
- Heart rate variability analysis
- Temporal disease progression prediction
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')


class AdvancedMedicalRNN:
    """Advanced RNN/LSTM for medical time-series"""
    
    def __init__(self):
        self.models = {}
        self.history = {}
        print("[RNN] Initializing Advanced Medical RNN/LSTM...")
    
    def build_lstm_arrhythmia_detector(self, input_shape=(187, 1)):
        """LSTM for ECG arrhythmia detection"""
        print("[RNN] Building LSTM Arrhythmia Detector...")
        
        model = models.Sequential([
            layers.Input(shape=input_shape),
            
            # Data augmentation for time series
            layers.GaussianNoise(0.01),
            
            # Block 1: LSTM with bidirectional
            layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.2)),
            layers.BatchNormalization(),
            
            # Block 2: LSTM
            layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.2)),
            layers.BatchNormalization(),
            
            # Block 3: LSTM
            layers.Bidirectional(layers.LSTM(32, return_sequences=False, dropout=0.2)),
            layers.BatchNormalization(),
            
            # Dense layers
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(5, activation='softmax')  # 5 arrhythmia classes
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        self.models['lstm_arrhythmia'] = model
        print(f"[✓] LSTM Arrhythmia detector built: {model.count_params():,} parameters")
        return model
    
    def build_gru_cardiac_trend(self, input_shape=(100, 5)):
        """GRU for cardiac trend analysis"""
        print("[RNN] Building GRU Cardiac Trend Analyzer...")
        
        model = models.Sequential([
            layers.Input(shape=input_shape),
            
            # Block 1: GRU with attention
            layers.GRU(128, return_sequences=True, dropout=0.2),
            layers.BatchNormalization(),
            
            # Block 2: GRU
            layers.GRU(64, return_sequences=True, dropout=0.2),
            layers.BatchNormalization(),
            
            # Block 3: GRU
            layers.GRU(32, return_sequences=False, dropout=0.2),
            layers.BatchNormalization(),
            
            # Dense
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(3, activation='softmax')  # Normal, Warning, Critical
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.models['gru_cardiac'] = model
        print(f"[✓] GRU Cardiac analyzer built: {model.count_params():,} parameters")
        return model
    
    def build_attention_lstm(self, input_shape=(100, 10)):
        """LSTM with attention mechanism for advanced analysis"""
        print("[RNN] Building Attention-based LSTM...")
        
        inputs = layers.Input(shape=input_shape)
        
        # LSTM layers
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.2))(inputs)
        x = layers.BatchNormalization()(x)
        
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.2))(x)
        x = layers.BatchNormalization()(x)
        
        # Attention layer
        attention = layers.AdditiveAttention()([x, x])
        x = layers.Add()([attention, x])
        
        # Final LSTM
        x = layers.LSTM(32, return_sequences=False, dropout=0.2)(x)
        x = layers.BatchNormalization()(x)
        
        # Dense layers
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(4, activation='softmax')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.models['attention_lstm'] = model
        print(f"[✓] Attention LSTM built: {model.count_params():,} parameters")
        return model
    
    def build_encoder_decoder(self, input_shape=(100, 5), output_shape=(50, 1)):
        """Encoder-Decoder for sequence-to-sequence predictions"""
        print("[RNN] Building Encoder-Decoder Model...")
        
        # Encoder
        encoder_inputs = layers.Input(shape=input_shape)
        encoder_lstm = layers.LSTM(128, return_state=True, dropout=0.2)
        encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
        encoder_states = [state_h, state_c]
        
        # Decoder
        decoder_inputs = layers.Input(shape=output_shape)
        decoder_lstm = layers.LSTM(128, return_sequences=True, return_state=True, dropout=0.2)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        
        # Dense layer
        decoder_dense = layers.Dense(1, activation='linear')
        decoder_outputs = decoder_dense(decoder_outputs)
        
        model = models.Model([encoder_inputs, decoder_inputs], decoder_outputs)
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse'
        )
        
        self.models['encoder_decoder'] = model
        print(f"[✓] Encoder-Decoder built: {model.count_params():,} parameters")
        return model
    
    def build_tcn_temporal(self, input_shape=(100, 5)):
        """Temporal Convolutional Network for time-series"""
        print("[RNN] Building TCN for Temporal Analysis...")
        
        model = models.Sequential([
            layers.Input(shape=input_shape),
            
            # Dilated convolutions
            layers.Conv1D(64, 3, padding='causal', activation='relu', dilation_rate=1),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Conv1D(64, 3, padding='causal', activation='relu', dilation_rate=2),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Conv1D(32, 3, padding='causal', activation='relu', dilation_rate=4),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            # Global pooling
            layers.GlobalAveragePooling1D(),
            
            # Dense
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(3, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.models['tcn_temporal'] = model
        print(f"[✓] TCN built: {model.count_params():,} parameters")
        return model
    
    def fine_tune(self, model_name, train_data, val_data, epochs=100):
        """Fine-tune RNN with advanced strategies"""
        print(f"[RNN] Fine-tuning {model_name}...")
        
        if model_name not in self.models:
            print(f"[!] Model {model_name} not found")
            return None
        
        model = self.models[model_name]
        
        # Learning rate schedule - exponential decay
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=50,
            decay_rate=0.96,
            staircase=True
        )
        
        model.optimizer.learning_rate = lr_schedule
        
        # Advanced callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                f'models/checkpoints/{model_name}_best.h5',
                monitor='val_accuracy',
                save_best_only=True
            )
        ]
        
        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        self.history[model_name] = history
        print(f"[✓] Fine-tuning complete for {model_name}")
        return history
    
    def predict_sequence(self, model_name, sequence_data):
        """Make predictions on sequence data"""
        if model_name not in self.models:
            return None
        
        model = self.models[model_name]
        prediction = model.predict(sequence_data, verbose=0)
        return prediction
    
    def get_model(self, name):
        """Get compiled model"""
        return self.models.get(name)


# Initialize global RNN instance
rnn_system = AdvancedMedicalRNN()

if __name__ == "__main__":
    print("Building all medical RNN models...")
    rnn = AdvancedMedicalRNN()
    rnn.build_lstm_arrhythmia_detector()
    rnn.build_gru_cardiac_trend()
    rnn.build_attention_lstm()
    rnn.build_encoder_decoder()
    rnn.build_tcn_temporal()
    print("\n[✓] All RNN models built successfully!")
