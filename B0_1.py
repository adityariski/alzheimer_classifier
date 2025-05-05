import tensorflow as tf
from tensorflow.keras.layers import (
    Input, RandomFlip, RandomRotation, RandomZoom, RandomContrast,
    Dropout, Flatten, Dense, Activation, BatchNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0

# 🔹 Step 1: Define the input and augmentation layers
inputs = Input(shape=(224, 224, 3))
x = RandomFlip("horizontal")(inputs)
x = RandomRotation(0.1)(x)
x = RandomZoom(0.1)(x)
x = RandomContrast(0.1)(x)

# 🔹 Step 2: Load base model (EfficientNetB0) with input_tensor=x
base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_tensor=x,
    pooling=None
)

# 🔹 Step 3: Freeze all but last 3 layers
for layer in base_model.layers[:-3]:
    layer.trainable = False

# 🔹 Step 4: Add custom top layers
x = base_model.output
x = Dropout(0.5)(x)
x = Flatten()(x)
x = BatchNormalization()(x)

# 3 Dense blocks
for _ in range(3):
    x = Dense(32, kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

# Final output layer (softmax for 4-class classification)
outputs = Dense(4, activation='softmax')(x)

# 🔹 Step 5: Define the model
model = Model(inputs=inputs, outputs=outputs)

# 🔹 Step 6: View model summary
model.summary()

