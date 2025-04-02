from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten


# AlexNet model
class AlexNet(Sequential):
    """
    A tensorflow implementation of the paper:
    `AlexNet <https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf>`
    """
    def __init__(self, input_shape, num_classes):
        super().__init__()

        self.add(Conv2D(96, kernel_size=(11, 11), strides=4, padding='valid', activation='relu', input_shape=input_shape, kernel_initializer='he_normal'))
        self.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', data_format=None))

        self.add(Conv2D(256, kernel_size=(5, 5), strides=1, padding='same', activation='relu', kernel_initializer='he_normal'))
        self.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', data_format=None))

        self.add(Conv2D(384, kernel_size=(3, 3), strides=1, padding='same', activation='relu', kernel_initializer='he_normal'))

        self.add(Conv2D(384, kernel_size=(3, 3), strides=1, padding='same', activation='relu', kernel_initializer='he_normal'))

        self.add(Conv2D(256, kernel_size=(3, 3), strides=1, padding='same', activation='relu', kernel_initializer='he_normal'))

        self.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', data_format=None))

        self.add(Flatten())
        self.add(Dense(4096, activation='relu'))
        self.add(Dense(4096, activation='relu'))
        self.add(Dense(1000, activation='relu'))
        self.add(Dense(num_classes, activation='softmax'))

        self.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
