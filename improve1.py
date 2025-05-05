import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import EfficientNetB0

def build_improved_model(input_shape=(150, 150, 1), num_classes=4):
    """
    Builds an improved model with EfficientNetB0 backbone, data augmentation, 
    and optimized layers, adapted for input shape (150, 150, 1) and 4 output classes.

    Args:
        input_shape: Shape of the input images (height, width, channels).
        num_classes: Number of output classes (fixed to 4).

    Returns:
        A compiled TensorFlow Keras model.
    """

    # Data Augmentation block (more comprehensive)
    data_augmentation = tf.keras.Sequential([
        layers.Rescaling(1./255),
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomContrast(0.2)
    ], name="data_augmentation")

    # EfficientNetB0 base model
    base_model = EfficientNetB0(
        include_top=False,
        input_shape=(150, 150, 3),
        weights='imagenet',
        pooling='avg'
    )
    base_model.trainable = False

    # Input layer (grayscale -> RGB)
    inputs = layers.Input(shape=input_shape)
    x = layers.Concatenate()([inputs, inputs, inputs])
    x = data_augmentation(x)
    x = base_model(x)

    # Improved Custom Head
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(512, activation='swish', kernel_regularizer='l2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(256, activation='swish', kernel_regularizer='l2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(128, activation='swish', kernel_regularizer='l2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    # Final classification layer (4 classes)
    outputs = layers.Dense(4, activation='softmax')(x) # Hardcoded num_classes to 4

    # Build model
    model = models.Model(inputs, outputs)

    # Optimized compilation
    optimizer = optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# Create and summarize the model
model = build_improved_model(input_shape=(150, 150, 1), num_classes=4) # num_classes is now redundant.
model.summary()
