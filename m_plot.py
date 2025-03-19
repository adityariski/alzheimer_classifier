import os
import cv2
import numpy as np
from keras._tf_keras.keras.preprocessing import image
from keras._tf_keras.keras.saving import load_model

pretrained_cnn = load_model("./alzheimer_model.keras")
categories = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]

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
for file in os.listdir("dataset/test/MildDemented/"):
    c += 1
    if "MildDemented" == predict_label("dataset/test/MildDemented/" + file):
        ok += 1
    else:
        ko += 1

print(f"OK: {ok}/{c}")
print(f"KO: {ko}/{c}")
# print(predict_label("dataset/test/NonDemented/26.jpg"))
# print(predict_label("dataset/test/VeryMildDemented/32 (10).jpg"))
