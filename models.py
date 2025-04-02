import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, GlobalAveragePooling2D, Input, BatchNormalization, Activation
from keras._tf_keras.keras import applications, Model
from keras._tf_keras.keras.regularizers import l2

class AlexNet(Sequential):
    """
    Scaled-down TensorFlow version of paper "AlexNet <https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf>"

    Key Differences from Original AlexNet:

    Reduced Filters/Units:
        Conv1: 48 filters (vs. 96 in original AlexNet).
        Conv2: 128 filters (vs. 256).
        Dense Layers: 2048 → 1024 units (vs. 4096 → 4096).

    Kernel Initialization:
        Uses he_normal (modern best practice), while the original AlexNet didn’t specify this explicitly.
    """
    def __init__(self, input_shape=(128, 128, 1), num_classes=4):
        super().__init__()

        # Input Layer
        self.add(Input(shape=input_shape))

        # Conv Block 1 (Enhanced with proper BN placement)
        self.add(Conv2D(48, (11, 11), strides=4, kernel_initializer="he_normal", use_bias=False))
        self.add(BatchNormalization())
        self.add(Activation("relu"))  # ReLU after BN
        self.add(MaxPooling2D((3, 3), strides=2))

        # Conv Block 2
        self.add(Conv2D(128, (5, 5), padding="same", kernel_initializer="he_normal", use_bias=False))
        self.add(BatchNormalization())
        self.add(Activation("relu"))
        self.add(MaxPooling2D((3, 3), strides=2))

        # Conv Blocks 3-5 (Added BN to all conv layers)
        self.add(Conv2D(192, (3, 3), padding="same", kernel_initializer="he_normal", use_bias=False))
        self.add(BatchNormalization())
        self.add(Activation("relu"))
        
        self.add(Conv2D(192, (3, 3), padding="same", kernel_initializer="he_normal", use_bias=False))
        self.add(BatchNormalization())
        self.add(Activation("relu"))
        
        self.add(Conv2D(128, (3, 3), padding="same", kernel_initializer="he_normal", use_bias=False))
        self.add(BatchNormalization())
        self.add(Activation("relu"))
        self.add(MaxPooling2D((3, 3), strides=2))

        # Classifier (With kernel regularization)
        self.add(Flatten())
        self.add(Dense(2048, activation="relu", kernel_initializer="he_normal", kernel_regularizer=l2(0.01)))
        self.add(Dropout(0.7))
        self.add(Dense(1024, activation="relu", kernel_initializer="he_normal", kernel_regularizer=l2(0.01)))
        self.add(Dropout(0.7))
        self.add(Dense(num_classes, activation="softmax"))

        self.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

class EfficientNetB0(Model):
    def __init__(self, input_shape=(128, 128, 3), num_classes=4):
        super().__init__()

        self.base_model = applications.EfficientNetB0(include_top=False, input_shape=input_shape, weights="imagenet")
        self.base_model.trainable = False  # Freeze the base model

        self.global_avg_pool = GlobalAveragePooling2D()
        self.dense = Dense(256, activation="relu")
        self.dropout = Dropout(0.5)
        self.output_layer = Dense(num_classes, activation="softmax")
        self.build((None,) + input_shape)
        self.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    def call(self, inputs):
        x = self.base_model(inputs, training=False)
        x = self.global_avg_pool(x)
        x = self.dense(x)
        x = self.dropout(x)
        return self.output_layer(x)

    def build_model(self):
        inputs = Input(shape=(128, 128, 3))
        return Model(inputs, self.call(inputs))

# b0 = EfficientNetB0()
# b0.summary()
# mod = AlexNet()
# mod.summary()
