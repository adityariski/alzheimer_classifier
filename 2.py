import os
import cv2
import numpy as np
# import matplotlib.pyplot as plt
# import tensorflow as tf
import keras
from keras._tf_keras.keras import layers
# from keras._tf_keras.keras.models import Sequential
# from keras._tf_keras.keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout,Input

# Load dataset
train_dir = "./dataset/train"
test_dir = "./dataset/test"

categories = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]

# Define image size and batch size
IMG_SIZE = (128, 128)

# Data Preprocessing Function
def load_images_from_folder(folder):
    images = []
    labels = []
    for category in categories:
        path = os.path.join(folder, category)
        label = categories.index(category)
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, IMG_SIZE)
            images.append(image)
            labels.append(label)
    return np.array(images), np.array(labels)

# Load training and testing images
X_train, y_train = load_images_from_folder(train_dir)
X_test, y_test = load_images_from_folder(test_dir)

# Normalize data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape for CNN input
#X_train = X_train.reshape(5121, 128, 128, 1)
#X_test = X_test.reshape(1279, 128, 128, 1)
X_train = X_train.reshape(-1, 128, 128, 1)
X_test = X_test.reshape(-1, 128, 128, 1)

# Convert labels to categorical
y_train = keras.utils.to_categorical(y_train, num_classes=len(categories))
y_test = keras.utils.to_categorical(y_test, num_classes=len(categories))

#cnn = Sequential()
#cnn.add(Conv2D(64,(3,3), padding="same", activation='relu', input_shape=(128, 128, 1)))
#cnn.add(MaxPooling2D())
#cnn.add(Conv2D(64,(3,3), padding="same", activation='relu'))
#cnn.add(MaxPooling2D())
#cnn.add(Conv2D(32,(2,2), padding="same", activation='relu'))
#cnn.add(MaxPooling2D())
#cnn.add(Flatten())
#cnn.add(Dense(100,activation='relu'))
#cnn.add(Dense(4,activation='softmax'))
#cnn.summary()

def build_model():
    inputs = keras.Input(shape=(128, 128, 1))
    x = layers.Conv2D(64, (3, 3), padding="same", activation='relu')(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, (2, 2), padding="same", activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(100, activation='relu')(x)
    #x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(len(categories), activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_model_alt():
    inputs = keras.Input(shape=(128, 128, 1))  # Define input separately
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(len(categories), activation='softmax')(x)

    model = keras.Model(inputs, outputs)  # Create model with explicit inputs
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

cnn = build_model_alt()
cnn.summary()
