import os
import cv2
import numpy as np
from keras._tf_keras.keras.preprocessing import image
from keras._tf_keras.keras.saving import load_model
from models import AlexNet
# from keras._tf_keras.keras.utils import plot_model

pretrained_cnn = load_model("./alexnet.keras", AlexNet)
categories = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]

# plot_model(pretrained_cnn, "model_plot.png", show_shapes=True, show_layer_names=True)

def predict_label(img_path):
    #i = Image.open(img_path).convert("L")
    i = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    i = cv2.resize(i, (128, 128))
    i = image.img_to_array(i) / 255.0
    i = i.reshape(-1,128,128,1)
    predict_x = pretrained_cnn.predict(i) # type: ignore
    classes_x = np.argmax(predict_x, axis=1)
    return categories[classes_x[0]]

c = 0
ok = 0
ko = 0

test_category = categories[3]
for file in os.listdir(f"dataset/test/{test_category}/"):
    c += 1
    if test_category == predict_label(f"dataset/test/{test_category}/" + file):
        ok += 1
    else:
        ko += 1

print(f"OK: {ok}/{c}\n")
print(f"KO: {ko}/{c}\n")
