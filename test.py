import imutils
import cv2
import io
import matplotlib.pyplot as plt
import joblib
from skimage.feature import hog
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from pathlib import Path

from scipy.ndimage.morphology import binary_dilation
from skimage.io import imread_collection

'''from tensorflow.keras import models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical'''

current_directory = Path(__file__).resolve().parent

train_new_model = False
option = input("Quieres entrar un nuevo MODELO? (y/n): ")
if option in ["y", "Y"]:
    train_new_model = True

mnist = tf.keras.datasets.mnist

if train_new_model:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(
        256, activation='relu', input_shape=(28*28, )))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

    history = model.fit(x_train[:, :], y_train, epochs=6, batch_size=64)
    model.save('handwritten_digits.model')

else:
    model = tf.keras.models.load_model('handwritten_digits.model')


def digit_recognition(image, height, width):
    im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
    threshold, bwImage = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)
    ctrs, hier = cv2.findContours(
        bwImage.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]
    detected_images = []
    for rect in rects:
        leng = int(rect[3] * 1.6)
        pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
        pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
        roi = bwImage[pt1:pt1+leng, pt2:pt2+leng]
        roi = cv2.resize(roi, (height, width), interpolation=cv2.INTER_AREA)
        roi = cv2.dilate(roi, (3, 3))
        detected_images.append(roi)
    cv2.waitKey()
    predicted = []
    convertedImage = detected_images
    for i in range(len(convertedImage)):
        predict = model.predict(np.reshape(convertedImage[i], (1, 784)))
        predicted.append(predict.argmax())
    y = 0
    for rect in rects:
        cv2.rectangle(image, (rect[0], rect[1]), (rect[0] +
                      rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 4
        color = (255, 0, 0)
        thickness = 2
        cv2.putText(image, str(
            predicted[y]), (rect[0], rect[1]), font, fontScale, color, thickness, cv2.LINE_AA)
        y = y + 1
    return predicted


def convert(lst):
    s = [str(i) for i in lst]
    res = int("".join(s))
    return (res)


path = f"{current_directory}/content/"
err = []
img = []
valid_images = [".jpg", ".gif", ".png", ".jpeg"]
predicted_rst = []
actual_rst = []

for f in os.listdir(path):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    nm, e = f.split('.')
    image = cv2.imread(os.path.join(path, f))
    plt.imshow(image)
    plt.show()
    try:
        predicted_rst.append(digit_recognition(image, 28, 28))
        img.append(image)
        actual_rst.append(nm)
        plt.imshow(image)
        plt.show()
    except:
        err.append(nm)

correct_predictions = 0
for k in range(0, len(img)):
    if (int(actual_rst[k]) == convert(predicted_rst[k])):
        correct_predictions = correct_predictions + 1

accuracy = (correct_predictions/len(img))*100
# print("Accuracy of the model is: ", accuracy)
