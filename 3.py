from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import (Input, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Dense, Dropout, RandomFlip, RandomRotation, RandomZoom)

def create_alzheimer_model(input_shape=(176, 176, 1), num_classes=4):
    """
    Creates a lightweight CNN model for Alzheimer's classification
    
    Parameters:
    input_shape (tuple): Input shape of images (height, width, channels)
    num_classes (int): Number of output classes
    
    Returns:
    model (keras.Model): Compiled Keras model
    """
    
    model = Sequential(name="Alzheimer_CNN")
    
    # Explicit input layer
    model.add(Input(shape=input_shape))
    
    # Data augmentation
    model.add(RandomFlip("horizontal_and_vertical"))
    model.add(RandomRotation(0.1))
    model.add(RandomZoom(0.1))
    
    # Feature extraction
    model.add(Conv2D(32, (3, 3), padding='same', activation='leaky_relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), padding='same', activation='leaky_relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Conv2D(64, (3, 3), padding='same', activation='leaky_relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding='same', activation='leaky_relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    
    model.add(GlobalAveragePooling2D())
    
    # Classification head
    model.add(Dense(64, activation='leaky_relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='leaky_relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    
    return model

model = create_alzheimer_model()
model.summary()

from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    './dataset/train',
    target_size=(176, 176),
    color_mode='grayscale',
    batch_size=16,
    class_mode='sparse')

val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = train_datagen.flow_from_directory(
    './dataset/test',
    target_size=(176, 176),
    color_mode='grayscale',
    batch_size=16,
    class_mode='sparse')

hist = model.fit(
    train_generator,
    epochs=30,
    steps_per_epoch=100,
    validation_data=val_generator,
    validation_steps=50,
    verbose=2
)

import matplotlib.pyplot as plt
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc = 'lower right')
plt.show()
