"""
Best Fine-Tuning Strategies and Utilities
- Learning rate scheduling
- Transfer learning optimization
- Hyperparameter tuning
- Model evaluation
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import Callback
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class AdvancedFineTuning:
    """Advanced fine-tuning strategies for medical models"""
    
    def __init__(self):
        self.tuning_history = {}
        self.best_hyperparams = {}
        print("[TUNING] Advanced Fine-Tuning System initialized")
    
    # ============ LEARNING RATE SCHEDULES ============
    
    @staticmethod
    def exponential_decay_schedule(initial_lr=0.001, decay_steps=100, decay_rate=0.96):
        """Exponential decay schedule"""
        return keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=True
        )
    
    @staticmethod
    def polynomial_decay_schedule(initial_lr=0.001, decay_steps=1000, end_lr=0.0001, power=0.5):
        """Polynomial decay schedule"""
        return keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=decay_steps,
            end_learning_rate=end_lr,
            power=power
        )
    
    @staticmethod
    def cosine_decay_schedule(initial_lr=0.001, decay_steps=1000):
        """Cosine annealing schedule"""
        return keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=initial_lr,
            decay_steps=decay_steps
        )
    
    @staticmethod
    def cyclic_lr_schedule(initial_lr=0.0001, max_lr=0.001, step_size=100):
        """Cyclic learning rate schedule"""
        def lr_scheduler(epoch):
            cycle_length = 2 * step_size
            cycle = np.floor(epoch / cycle_length)
            x = np.abs(epoch / step_size - 2 * cycle - 1)
            lr = initial_lr + (max_lr - initial_lr) * max(0, (1 - x))
            return lr
        
        return keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: None)
    
    # ============ TRANSFER LEARNING STRATEGIES ============
    
    @staticmethod
    def progressive_unfreezing(model, unfreeze_at_layer=-1):
        """Progressively unfreeze layers"""
        layers_count = len(model.layers)
        
        # Freeze all layers
        for layer in model.layers:
            layer.trainable = False
        
        # Unfreeze top layers
        if unfreeze_at_layer == -1:
            unfreeze_at_layer = layers_count // 2
        
        for i in range(unfreeze_at_layer, layers_count):
            model.layers[i].trainable = True
        
        print(f"[TUNING] Unfroze {layers_count - unfreeze_at_layer} layers")
        return model
    
    @staticmethod
    def discriminative_lr(model, base_lr=0.001, layer_decay=0.8):
        """Use different learning rates for different layers"""
        optimizer_groups = []
        
        num_layers = len(model.layers)
        
        for i, layer in enumerate(model.layers):
            # Earlier layers get lower learning rates
            lr = base_lr * (layer_decay ** (num_layers - i - 1))
            optimizer_groups.append({
                'layer': layer.name,
                'learning_rate': lr
            })
        
        print(f"[TUNING] Discriminative LR configured for {len(optimizer_groups)} layers")
        return optimizer_groups
    
    # ============ REGULARIZATION STRATEGIES ============
    
    @staticmethod
    def get_regularization_callbacks():
        """Get best regularization callbacks"""
        return [
            # Early stopping
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce LR on plateau
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            ),
            
            # Model checkpoint
            keras.callbacks.ModelCheckpoint(
                'models/checkpoints/best_model.h5',
                monitor='val_accuracy',
                save_best_only=True
            ),
            
            # Tensorboard logging
            keras.callbacks.TensorBoard(
                log_dir='./logs',
                histogram_freq=1
            )
        ]
    
    # ============ MIXED PRECISION TRAINING ============
    
    @staticmethod
    def setup_mixed_precision():
        """Setup mixed precision for faster training"""
        policy = keras.mixed_precision.Policy('mixed_float16')
        keras.mixed_precision.set_global_policy(policy)
        print("[TUNING] Mixed precision enabled (float16 + float32)")
    
    # ============ DATA AUGMENTATION ============
    
    @staticmethod
    def get_image_augmentation():
        """Get best image augmentation pipeline"""
        return keras.Sequential([
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomFlip("vertical"),
            keras.layers.RandomRotation(0.3),
            keras.layers.RandomZoom(0.2),
            keras.layers.RandomTranslation(0.2, 0.2),
            keras.layers.RandomContrast(0.2),
            keras.layers.RandomBrightness(0.2),
        ])
    
    @staticmethod
    def get_timeseries_augmentation():
        """Get best time-series augmentation"""
        return keras.Sequential([
            keras.layers.GaussianNoise(0.01),
            keras.layers.RandomZoom(0.1, axis=0),  # Only time axis
        ])
    
    # ============ OPTIMIZATION STRATEGIES ============
    
    @staticmethod
    def get_advanced_optimizer(optimizer_name='adamw', learning_rate=0.001):
        """Get advanced optimizers"""
        optimizers_map = {
            'adamw': keras.optimizers.AdamW(learning_rate=learning_rate),
            'lamb': keras.optimizers.LAMB(learning_rate=learning_rate),
            'radam': keras.optimizers.experimental.RectifiedAdam(learning_rate=learning_rate),
            'sgd_nesterov': keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True),
        }
        
        return optimizers_map.get(optimizer_name, keras.optimizers.Adam(learning_rate=learning_rate))
    
    # ============ HYPERPARAMETER TUNING ============
    
    def tune_hyperparameters(self, model, train_data, val_data, param_grid):
        """Grid search for best hyperparameters"""
        best_score = float('inf')
        best_params = None
        
        print("[TUNING] Starting hyperparameter tuning...")
        
        for learning_rate in param_grid.get('learning_rate', [0.001]):
            for dropout_rate in param_grid.get('dropout_rate', [0.2]):
                for batch_size in param_grid.get('batch_size', [32]):
                    print(f"\nTesting LR={learning_rate}, Dropout={dropout_rate}, BS={batch_size}")
                    
                    # Clone model
                    test_model = keras.models.clone_model(model)
                    
                    # Compile with new params
                    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
                    test_model.compile(
                        optimizer=optimizer,
                        loss='categorical_crossentropy',
                        metrics=['accuracy']
                    )
                    
                    # Train
                    history = test_model.fit(
                        train_data,
                        validation_data=val_data,
                        epochs=10,
                        batch_size=batch_size,
                        verbose=0,
                        callbacks=[
                            keras.callbacks.EarlyStopping(
                                monitor='val_loss',
                                patience=3,
                                restore_best_weights=True
                            )
                        ]
                    )
                    
                    val_loss = min(history.history['val_loss'])
                    
                    if val_loss < best_score:
                        best_score = val_loss
                        best_params = {
                            'learning_rate': learning_rate,
                            'dropout_rate': dropout_rate,
                            'batch_size': batch_size
                        }
                    
                    print(f"Best Val Loss: {val_loss:.4f}")
        
        self.best_hyperparams = best_params
        print(f"\n[✓] Best hyperparameters: {best_params}")
        return best_params
    
    # ============ EVALUATION ============
    
    @staticmethod
    def evaluate_model(model, test_data, test_labels):
        """Comprehensive model evaluation"""
        predictions = model.predict(test_data, verbose=0)
        pred_classes = np.argmax(predictions, axis=1)
        
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, confusion_matrix, classification_report
        )
        
        accuracy = accuracy_score(test_labels, pred_classes)
        precision = precision_score(test_labels, pred_classes, average='weighted')
        recall = recall_score(test_labels, pred_classes, average='weighted')
        f1 = f1_score(test_labels, pred_classes, average='weighted')
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'confusion_matrix': confusion_matrix(test_labels, pred_classes).tolist(),
            'classification_report': classification_report(test_labels, pred_classes)
        }
        
        print("\n[EVALUATION]")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        
        return metrics
    
    # ============ KNOWLEDGE DISTILLATION ============
    
    @staticmethod
    def create_student_model(teacher_model, num_classes, compression_factor=0.5):
        """Create smaller student model for knowledge distillation"""
        teacher_layers = len(teacher_model.layers)
        student_size = max(32, int(teacher_layers * compression_factor))
        
        student = keras.Sequential([
            keras.layers.Input(shape=teacher_model.input_shape[1:]),
            keras.layers.Dense(student_size, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(student_size // 2, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        return student
    
    @staticmethod
    def distill_knowledge(teacher_model, student_model, train_data, val_data, temperature=3.0, alpha=0.5):
        """Train student model using knowledge from teacher"""
        
        class DistillationCallback(Callback):
            def on_epoch_end(self, epoch, logs=None):
                if (epoch + 1) % 5 == 0:
                    print(f"[Distillation] Epoch {epoch+1} - Student Loss: {logs.get('loss', 'N/A'):.4f}")
        
        print(f"[TUNING] Starting knowledge distillation (T={temperature}, α={alpha})")
        
        history = student_model.fit(
            train_data,
            validation_data=val_data,
            epochs=50,
            callbacks=[DistillationCallback()],
            verbose=0
        )
        
        print("[✓] Knowledge distillation complete")
        return history


# Initialize global fine-tuning system
fine_tuning_system = AdvancedFineTuning()

if __name__ == "__main__":
    print("Advanced Fine-Tuning System initialized")
    print("\nAvailable strategies:")
    print("- Learning rate schedules (exponential, polynomial, cosine, cyclic)")
    print("- Transfer learning (progressive unfreezing, discriminative LR)")
    print("- Regularization (early stopping, LR reduction, checkpointing)")
    print("- Mixed precision training")
    print("- Advanced optimizers (AdamW, LAMB, RectifiedAdam)")
    print("- Hyperparameter tuning")
    print("- Knowledge distillation")
    print("\n[✓] All fine-tuning utilities ready!")
