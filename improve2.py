import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import EfficientNetB0

# Data Augmentation (now applied after resizing)
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.1),
], name="data_augmentation")

# EfficientNetB0 base model with proper preprocessing
def efficientnet_preprocess(x):
    return tf.keras.applications.efficientnet.preprocess_input(x)

# Input size needs to be multiple of 32 for EfficientNet - nearest to 150 is 160
TARGET_SIZE = 160  
base_model = EfficientNetB0(
    include_top=False,
    input_shape=(TARGET_SIZE, TARGET_SIZE, 3),
    weights='imagenet',
    pooling='avg'
)
base_model.trainable = False  # Freeze base model initially

# Input pipeline
inputs = layers.Input(shape=(150, 150, 1))
x = layers.Concatenate()([inputs, inputs, inputs])  # Convert to RGB (150,150,3)
x = layers.Resizing(TARGET_SIZE, TARGET_SIZE)(x)  # Critical resize for EfficientNet compatibility
x = data_augmentation(x)
x = layers.Lambda(efficientnet_preprocess)(x)  # Proper EfficientNet scaling
x = base_model(x)

# Enhanced classification head
x = layers.BatchNormalization()(x)
x = layers.Activation('swish')(x)
x = layers.Dropout(0.5)(x)

x = layers.Dense(512, kernel_regularizer=regularizers.l2(1e-4))(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('swish')(x)
x = layers.Dropout(0.5)(x)

x = layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4))(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('swish')(x)
x = layers.Dropout(0.5)(x)

x = layers.Dense(128, kernel_regularizer=regularizers.l2(1e-4))(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('swish')(x)
x = layers.Dropout(0.4)(x)

x = layers.Dense(64, kernel_regularizer=regularizers.l2(1e-4))(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('swish')(x)
x = layers.Dropout(0.4)(x)

x = layers.Dense(32, kernel_regularizer=regularizers.l2(1e-4))(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('swish')(x)
x = layers.Dropout(0.4)(x)

# Final classification layer
outputs = layers.Dense(4, activation='softmax')(x)

# Compile with custom optimizer
model = models.Model(inputs, outputs)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(
    optimizer=optimizer,
    # loss='sparse_categorical_crossentropy',
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3)]
)

model.summary()
