import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model
import tensorflow as tf

class ModelPredictor:
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def predict_image(self, img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        prediction = self.model.predict(img_array)
        predicted_class = "Stroke" if prediction[0][0] > 0.5 else "Normal"
        confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]

        return predicted_class, confidence, img_array

    def generate_gradcam(self, img_array, activation_layer):
        grad_model = Model(
            [self.model.inputs], [self.model.get_layer(activation_layer).output, self.model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, 0]

        output = conv_outputs[0]
        grads = tape.gradient(loss, conv_outputs)[0]

        gate_f = tf.cast(output > 0, 'float32')
        gate_r = tf.cast(grads > 0, 'float32')

        guided_grads = tf.cast(output > 0, 'float32') * tf.cast(grads > 0, 'float32') * grads

        weights = tf.reduce_mean(guided_grads, axis=(0, 1))

        cam = np.dot(output, weights)

        cam = cv2.resize(cam, (224, 224))
        cam = np.maximum(cam, 0)
        heatmap = (cam - cam.min()) / (cam.max() - cam.min())

        return heatmap, weights

    
 