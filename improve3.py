import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import EfficientNetB0

# Enhanced Data Augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.25),
    layers.RandomZoom(0.25),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.2),
], name="data_augmentation")

# Input processing
def efficientnet_preprocess(x):
    return tf.keras.applications.efficientnet.preprocess_input(x)

# Base model configuration
TARGET_SIZE = 160
base_model = EfficientNetB0(
    include_top=False,
    input_shape=(TARGET_SIZE, TARGET_SIZE, 3),
    weights='imagenet',
    pooling='avg'
)
base_model.trainable = False

# Enhanced classification head with better dimension transition
def create_head(inputs):
    x = layers.BatchNormalization()(inputs)
    x = layers.Dropout(0.5)(x)
    
    # Gradual dimension reduction
    x = layers.Dense(1024, kernel_regularizer=regularizers.L2(1e-4))(x)
    x = layers.GroupNormalization(groups=32)(x)  # Alternative to BN
    x = layers.Activation('swish')(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(512, kernel_regularizer=regularizers.L2(1e-4))(x)
    x = layers.LayerNormalization()(x)
    x = layers.Activation('swish')(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.Dense(256, kernel_regularizer=regularizers.L2(1e-4))(x)
    x = layers.WeightNormalization()(x)  # Weight normalization
    x = layers.Activation('swish')(x)
    x = layers.Dropout(0.3)(x)
    
    return x

# Input pipeline with learned grayscale conversion
inputs = layers.Input(shape=(150, 150, 1))
x = layers.Conv2D(3, (3, 3), padding='same', activation='linear')(inputs)  # Learnable RGB mapping
x = layers.Resizing(TARGET_SIZE, TARGET_SIZE)(x)
x = data_augmentation(x)
x = layers.Lambda(efficientnet_preprocess)(x)
x = base_model(x)

# Create enhanced head
x = create_head(x)

# Final layers with label smoothing
outputs = layers.Dense(4, activation='softmax', 
                      kernel_regularizer=regularizers.L2(1e-4),
                      bias_regularizer=regularizers.L2(1e-4))(x)

# Optimizer configuration
optimizer = tf.optimizers.Adam(
    learning_rate=1e-3,
    weight_decay=1e-4,
    global_clipnorm=1.0
)

model = models.Model(inputs, outputs)
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy', 
            tf.keras.metrics.TopCategoricalAccuracy(k=3),
            tf.keras.metrics.AUC(name='auc')]
)

model.summary()
