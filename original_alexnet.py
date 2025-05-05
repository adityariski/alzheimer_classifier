# Suppresses INFO and WARNING messages
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import keras
from keras._tf_keras.keras import layers

def alexnet(input_shape=(227, 227, 3), num_classes=1000):
    """
    TensorFlow version of paper `AlexNet <https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf>`
    """
    # The AlexNet paper mentions 224×224, but in practice, it used 227×227 because of the first convolutional layer’s 11×11 kernel with stride 4.
    # Using 224×224 would cause misalignment when downsampling.
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(96, (11, 11), strides=4, activation='relu'),
        layers.MaxPooling2D((3, 3), strides=2),

        layers.Conv2D(256, (5, 5), padding='same', activation='relu'),
        layers.MaxPooling2D((3, 3), strides=2),

        layers.Conv2D(384, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(384, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((3, 3), strides=2),

        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Example usage:
model = alexnet()
model.summary()
