import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import keras
from keras._tf_keras.keras import layers

from sklearn.metrics import confusion_matrix, classification_report

train_dir = "./dataset/train"
test_dir = "./dataset/test"

categories = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]

IMG_SIZE = (128, 128)
BATCH_SIZE = 32

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

X_train, Y_train = load_images_from_folder(train_dir)
X_test, Y_test = load_images_from_folder(test_dir)

X_train = X_train / 255.0
X_test = X_test / 255.0

X_train = X_train.reshape(-1, 128, 128, 1)
X_test = X_test.reshape(-1, 128, 128, 1)

y_train = keras.utils.to_categorical(Y_train, num_classes=len(categories))
y_test = keras.utils.to_categorical(Y_test, num_classes=len(categories))

def build_classification_model():
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(len(categories), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

classification_model = build_classification_model()

history = classification_model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=BATCH_SIZE)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc = 'lower right')
plt.show()

# Build U-Net Segmentation Model
def unet_model(input_size=(128, 128, 1)):
    inputs = keras.Input(input_size)
    
    # Encoding Path
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    # Bottleneck
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    
    # Decoding Path
    u1 = layers.UpSampling2D((2, 2))(c3)
    u1 = layers.concatenate([u1, c2])
    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u1)
    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c4)
    
    u2 = layers.UpSampling2D((2, 2))(c4)
    u2 = layers.concatenate([u2, c1])
    c5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u2)
    c5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c5)
    
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c5)
    
    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

segmentation_model = unet_model()

segmentation_model.summary()

def visualize_segmentation(sample_images, model):
    fig, axes = plt.subplots(len(sample_images), 3, figsize=(10, len(sample_images) * 3))
    for i, img in enumerate(sample_images):
        img = img / 255.0  # Normalize
        img_input = img.reshape(1, 128, 128, 1)
        prediction = model.predict(img_input)[0].reshape(128, 128)
        
        axes[i, 0].imshow(img.squeeze(), cmap='gray')
        axes[i, 0].set_title("Original Image")
        axes[i, 1].imshow(prediction, cmap='jet')
        axes[i, 1].set_title("Segmentation Output")
        axes[i, 2].imshow(img.squeeze(), cmap='gray')
        axes[i, 2].imshow(prediction, cmap='jet', alpha=0.5)
        axes[i, 2].set_title("Overlay")
    plt.tight_layout()
    plt.show()

sample_test_images = X_test[:5]
visualize_segmentation(sample_test_images, segmentation_model)

### CUSTOM
segmentation_model.save("model", save_format="h5")

y_pred = segmentation_model.predict(X_test)
y_val=[]
for y in y_pred:
    y_val.append(np.argmax(y))
y_true=[]
for y in Y_test:
    y_true.append(np.argmax(y))
print(confusion_matrix(y_true,y_val))
print("Classification Report")
print(classification_report(y_true,y_val))
