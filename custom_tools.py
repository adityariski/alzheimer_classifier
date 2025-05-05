import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras._tf_keras.keras.preprocessing import image
from sklearn.metrics import confusion_matrix, classification_report

def load_images(folder, categories, img_size=(128, 128)):
    images = []
    labels = []

    for category in categories:
        path = os.path.join(folder, category)
        if not os.path.isdir(path):
            continue

        label = categories.index(category)
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

        for img_name in os.listdir(path):
            if not img_name.lower().endswith(valid_extensions):
                continue

            img_path = os.path.join(path, img_name)
            try:
                # Read image
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue

                h, w = image.shape
                target_size = img_size[0]  # 128

                # Calculate new dimensions preserving aspect ratio
                if w > h:  # Landscape orientation
                    new_h = target_size
                    new_w = int(w * (new_h / h))
                else:  # Portrait or square
                    new_w = target_size
                    new_h = int(h * (new_w / w))

                # Resize
                resized = cv2.resize(image, (new_w, new_h))

                # Calculate crop coordinates
                if new_w > target_size:
                    start_x = (new_w - target_size) // 2
                    cropped = resized[:, start_x:start_x + target_size]
                elif new_h > target_size:
                    start_y = (new_h - target_size) // 2
                    cropped = resized[start_y:start_y + target_size, :]
                else:
                    # Handle rare case where both dimensions are smaller
                    cropped = cv2.resize(resized, (target_size, target_size))

                # Final resize guarantee (if needed)
                if cropped.shape != (target_size, target_size):
                    cropped = cv2.resize(cropped, (target_size, target_size))

                # Add channel dimension and normalize
                cropped = np.expand_dims(cropped, axis=-1)  # (128, 128, 1)
                cropped = cropped.astype(np.float32) / 255.0

                images.append(cropped)
                labels.append(label)

            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                continue

    return np.array(images), np.array(labels)

def load_data(folder, categories, img_size):
    images = []
    labels = []
    for category in categories:
        path = os.path.join(folder, category)
        label = categories.index(category)
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            # image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.imread(img_path)
            image = cv2.resize(image, img_size)
            images.append(image)
            labels.append(label)
    return np.array(images), np.array(labels)

# def load_images(folder, categories, img_size):
#     images = []
#     labels = []
#     for category in categories:
#         path = os.path.join(folder, category)
#         label = categories.index(category)
#         for img in os.listdir(path):
#             img_path = os.path.join(path, img)
#             # image = cv2.imread(img_path)
#             image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#             image = cv2.resize(image, img_size)
#             images.append(image)
#             labels.append(label)
#     return np.array(images).reshape(-1,img_size[0],img_size[1],1)/255.0, np.array(labels)

def model_accuracy(model, train: list, test: list):
    # Only after full model development!
    test_loss, test_acc_1 = model.evaluate(train[0], train[1])
    del test_loss
    print(f"Test Accuracy: {test_acc_1*100:.2f}%\n")
    test_loss, test_acc_2 = model.evaluate(test[0], test[1])
    del test_loss
    print(f"Final Test Accuracy (from test data): {test_acc_2*100:.2f}%\n")

def predict_label(model, img_path, img_size, categories):
    #i = Image.open(img_path).convert("L")
    i = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    i = cv2.resize(i, img_size)
    i = image.img_to_array(i)
    i = i.reshape(-1,128,128,1) / 255.0

    predict_x = model.predict(i) 
    classes_x = np.argmax(predict_x, axis=1)

    #return show_image(img_path), Labels[classes_x[0]]
    return categories[classes_x[0]]

def report(model, categories, test_data, test_label):
    y_pred=model.predict(test_data)
    y_val=[]
    for y in y_pred:
        y_val.append(np.argmax(y))
    y_true=[]
    for y in test_label:
        y_true.append(np.argmax(y))

    print("Confusion Matrix")
    print(confusion_matrix(y_true, y_val), "\n")

    print("Classification Report")
    print(classification_report(y_true, y_val), "\n")

    cm = confusion_matrix(y_true, y_val)
    plt.figure(figsize=(8, 6), dpi=80, facecolor="w", edgecolor="k")
    ax = sns.heatmap(cm, cmap="Blues", annot=True, fmt="d", xticklabels=categories, yticklabels=categories)
    plt.title("Alzheimer\'s Disease Diagnosis")
    plt.xlabel("Prediction")
    plt.ylabel("Truth")
    plt.show(ax)

def show_sample(data, label, categories):
    k=0
    fig, ax = plt.subplots(1,4,figsize=(20,20))
    fig.text(s="", size=18, fontweight="bold", fontname="monospace", color="#000000", y=0.62, x=0.4, alpha=0.8)

    for i in categories:
        j=0
        while True :
            if  label[j] == categories.index(i):
                ax[k].imshow(data[j])
                ax[k].set_title(categories[label[j]])
                ax[k].axis("off")
                k+=1
                break
            j+=1

def cv2_show_sample(data, label, categories):
    k=0
    fig, ax = plt.subplots(1,4,figsize=(20,20))
    fig.text(s="", size=18, fontweight="bold", fontname="monospace", color="#000000", y=0.62, x=0.4, alpha=0.8)

    for i in categories:
        j=0
        while True :
            if  label[j] == categories.index(i):
                img = data[j].squeeze()
                if img.max() <= 1.0: img = (img * 255).astype(np.uint8)
                ax[k].imshow(img, cmap='gray', vmin=0, vmax=255)
                ax[k].set_title(categories[label[j]])
                ax[k].axis("off")
                k+=1
                break
            j+=1
