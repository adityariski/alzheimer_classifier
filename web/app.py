from keras._tf_keras.keras.saving import load_model
from keras._tf_keras.keras.preprocessing import image
from keras._tf_keras.keras.metrics import AUC
from flask import Flask, render_template, request, send_from_directory
import numpy as np
import cv2
import os

app = Flask(__name__)
dependencies = {"auc_roc": AUC}

categories = {
    0: "Demensia Ringan",
    1: "Demensia Sedang",
    2: "Tidak Demensia",
    3: "Demensia Sangat Ringan",
}

model = load_model("alzheimer_model.keras")
model.make_predict_function() # type: ignore

def predict_label(img_path):
    i = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    i = cv2.resize(i, (128, 128))
    i = image.img_to_array(i) / 255.0
    i = i.reshape(-1,128,128,1)
    predict_x = model.predict(i) # type: ignore
    classes_x = np.argmax(predict_x, axis=1)
    return categories[classes_x[0]]

@app.route("/", methods=["GET", "POST"])
def main(): return render_template("index.html")

@app.route("/submit", methods=["GET", "POST"])
def get_output():
    img_path = ""
    predict_result = ""
    if request.method == "POST":
        img = request.files["my_image"]
        img_path = f"static/upload/{img.filename}"
        img.save(img_path)
        predict_result = predict_label(img_path)
    return render_template("index.html", prediction=predict_result, img_path=img_path)

if __name__ == "__main__":
    app.run(port=1212, debug=False)
