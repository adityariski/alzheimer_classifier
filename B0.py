from keras._tf_keras.keras import Model, layers, applications

def build_efficientnet(input_shape=(150, 150, 3), num_classes=4):
    inputs = layers.Input(shape=input_shape)

    # Load EfficientNetB0 with modified input size
    base_model = applications.EfficientNetB0(include_top=False, input_shape=input_shape)
    
    x = base_model(inputs, training=True)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)
    return model

# Example usage:
model = build_efficientnet()
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()
