"""
Advanced CNN Models for Medical Image Analysis
- X-ray pneumonia detection
- ECG image classification
- Skin lesion analysis
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB4, ResNet50, InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')


class AdvancedMedicalCNN:
    """Advanced CNN for medical image analysis"""
    
    def __init__(self):
        self.models = {}
        self.history = {}
        print("[CNN] Initializing Advanced Medical CNN...")
    
    def build_pneumonia_detector(self, input_shape=(224, 224, 3)):
        """CNN for X-ray pneumonia detection"""
        print("[CNN] Building Pneumonia Detector...")
        
        model = models.Sequential([
            # Input layer
            layers.Input(shape=input_shape),
            
            # Data augmentation
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.2),
            
            # Block 1
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            layers.Dropout(0.25),
            
            # Block 2
            layers.Conv2D(128, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            layers.Dropout(0.25),
            
            # Block 3
            layers.Conv2D(256, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(256, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            layers.Dropout(0.25),
            
            # Block 4
            layers.Conv2D(512, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(512, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            layers.Dropout(0.25),
            
            # Global pooling and dense layers
            layers.GlobalAveragePooling2D(),
            layers.Dense(1024, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')  # Binary: normal vs pneumonia
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        self.models['pneumonia'] = model
        print(f"[✓] Pneumonia detector built: {model.count_params():,} parameters")
        return model
    
    def build_skin_lesion_classifier(self, input_shape=(299, 299, 3)):
        """Advanced CNN for skin lesion classification"""
        print("[CNN] Building Skin Lesion Classifier...")
        
        # Use transfer learning with InceptionV3
        base_model = InceptionV3(
            input_shape=input_shape,
            weights='imagenet',
            include_top=False
        )
        
        # Freeze base model
        base_model.trainable = False
        
        model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Rescaling(1./255),
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(7, activation='softmax')  # 7 types of skin lesions
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.models['skin_lesion'] = model
        print(f"[✓] Skin lesion classifier built: {model.count_params():,} parameters")
        return model
    
    def build_ecg_image_classifier(self, input_shape=(224, 224, 1)):
        """CNN for ECG image classification"""
        print("[CNN] Building ECG Image Classifier...")
        
        model = models.Sequential([
            layers.Input(shape=input_shape),
            
            # Preprocessing
            layers.Rescaling(1./255),
            
            # Block 1
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            layers.Dropout(0.2),
            
            # Block 2
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            layers.Dropout(0.2),
            
            # Block 3
            layers.Conv2D(128, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            layers.Dropout(0.2),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(5, activation='softmax')  # 5 arrhythmia classes
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.models['ecg_image'] = model
        print(f"[✓] ECG classifier built: {model.count_params():,} parameters")
        return model
    
    def build_resnet_medical_classifier(self, input_shape=(224, 224, 3), num_classes=10):
        """ResNet50-based medical image classifier"""
        print("[CNN] Building ResNet Medical Classifier...")
        
        base_model = ResNet50(
            input_shape=input_shape,
            weights='imagenet',
            include_top=False
        )
        
        # Freeze base
        base_model.trainable = False
        
        model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Rescaling(1./255),
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(1024, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.models['resnet_medical'] = model
        print(f"[✓] ResNet Medical classifier built: {model.count_params():,} parameters")
        return model
    
    def fine_tune(self, model_name, train_data, val_data, epochs=50):
        """Fine-tune model with best practices"""
        print(f"[CNN] Fine-tuning {model_name}...")
        
        if model_name not in self.models:
            print(f"[!] Model {model_name} not found")
            return None
        
        model = self.models[model_name]
        
        # Learning rate schedule
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=100,
            decay_rate=0.96,
            staircase=True
        )
        
        model.optimizer.learning_rate = lr_schedule
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
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
    
    def predict(self, model_name, image_data, threshold=0.5):
        """Make predictions"""
        if model_name not in self.models:
            return None
        
        model = self.models[model_name]
        prediction = model.predict(image_data, verbose=0)
        return prediction
    
    def get_model(self, name):
        """Get compiled model"""
        return self.models.get(name)


# Initialize global CNN instance
cnn_system = AdvancedMedicalCNN()

if __name__ == "__main__":
    # Example usage
    print("Building all medical CNN models...")
    cnn = AdvancedMedicalCNN()
    cnn.build_pneumonia_detector()
    cnn.build_skin_lesion_classifier()
    cnn.build_ecg_image_classifier()
    cnn.build_resnet_medical_classifier()
    print("\n[✓] All CNN models built successfully!")
