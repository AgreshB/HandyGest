import cv2
import tensorflow as tf 

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import os

img = cv2.imread("capture.jpg")
#model = tf.keras.models.load_model("model/handrecognition_model.h5")
class_names = ["down", "palm", "l", "fist", "fist_moved", "thumb", "index", "ok", "palm_moved", "c"]

def predictImage(image):
    X = np.array(image,dtype="float16")
    X = X.reshape(1,120,320,1)
    prediction = model.predict(X)
    print(prediction)
    return np.argmax(prediction)

def plot_image(path):
    img = cv2.imread(path) # Reads the image into a numpy.array
    img_cvt = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Converts into the corret colorspace (RGB)
    print(img_cvt.shape) # Prints the shape of the image just to check
    plt.grid(False) # Without grid so we can see better
    plt.imshow(img_cvt) # Shows the image
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.title("Image " + path)

img = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
plot_image("capture.jpg")
print(img.shape)
'''
predictions = predictImage(img)
print(class_names[np.argmax(predictions)])
print(predictions)
'''