import keras
from custom_tools import load_images, show_sample
from models import AlexNet
from sklearn.model_selection import train_test_split

epochs = 10
batch_size = 6
img_size = (128, 128)
model = AlexNet(input_shape=(128, 128, 1))
model.summary()

categories = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
#workdir = "./workdir/"
workdir = "./dataset/train/"
images, labels = load_images(workdir, categories, img_size)
images = images / 255.0
show_sample(images, labels, categories)
#images = images.reshape(-1,128,128,1)
images = images.reshape(5121,128,128,1)
labels = keras.utils.to_categorical(labels)
print(f"{images.shape}\n{labels.shape}")

# train_data, test_data, train_label, test_label = train_test_split(images, labels, test_size=0.25,random_state=42, stratify=labels)
train_data, test_data, train_label, test_label = train_test_split(images, labels, test_size=0.25, random_state=42)
print(train_data.shape, test_data.shape, train_label.shape, test_label.shape)

cb = keras.callbacks.ReduceLROnPlateau(monitor = "val_loss", factor=0.5, verbose = 1, patience = 3)
history=model.fit(train_data, train_label, epochs=epochs, validation_data=(test_data, test_label), batch_size=batch_size, callbacks=cb)
