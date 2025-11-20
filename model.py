"""
Deep learning model architecture for DeepFake detection.
Uses transfer learning with XceptionNet as the base model.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import Xception
from tensorflow.keras.optimizers import Adam


def build_model(input_shape=(224, 224, 3), learning_rate=0.0001, 
                fine_tune=False, fine_tune_from_layer=None) -> keras.Model:
    """
    Build a DeepFake detection model using transfer learning.
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        learning_rate: Learning rate for the optimizer
        fine_tune: Whether to fine-tune the base model or freeze it
        fine_tune_from_layer: Layer name from which to start fine-tuning
                             (if None and fine_tune=True, unfreezes all layers)
    
    Returns:
        Compiled Keras model
    """
    # Load pre-trained Xception model (without top classification layer)
    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze or unfreeze base model layers
    if fine_tune:
        if fine_tune_from_layer:
            # Unfreeze layers from fine_tune_from_layer onwards
            trainable = False
            for layer in base_model.layers:
                if layer.name == fine_tune_from_layer:
                    trainable = True
                layer.trainable = trainable
        else:
            # Unfreeze all layers
            base_model.trainable = True
    else:
        # Freeze all base model layers
        base_model.trainable = False
    
    # Build the complete model
    inputs = keras.Input(shape=input_shape)
    
    # Base model (feature extraction)
    x = base_model(inputs, training=False)
    
    # Global average pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    # Custom classification head
    x = layers.Dense(128, activation='relu', name='dense_1')(x)
    x = layers.Dropout(0.5, name='dropout_1')(x)
    
    # Output layer (binary classification: 0 = real, 1 = fake)
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    # Create model
    model = models.Model(inputs, outputs, name='deepfake_detector')
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC']
    )
    
    return model


def build_model_efficientnet(input_shape=(224, 224, 3), learning_rate=0.0001) -> keras.Model:
    """
    Alternative model using EfficientNetB0 (can be used instead of Xception).
    
    Args:
        input_shape: Shape of input images
        learning_rate: Learning rate for the optimizer
    
    Returns:
        Compiled Keras model
    """
    from tensorflow.keras.applications import EfficientNetB0
    
    # Load pre-trained EfficientNetB0
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze base model
    base_model.trainable = False
    
    # Build the complete model
    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs, outputs, name='deepfake_detector_efficientnet')
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC']
    )
    
    return model


if __name__ == "__main__":
    # Test model creation
    print("Building model...")
    model = build_model()
    print("Model built successfully!")
    print(f"Total parameters: {model.count_params():,}")
    print(f"Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
    model.summary()

