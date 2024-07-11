import tensorflow as tf
import streamlit as st
from PIL import Image
import numpy as np
from PredictImage import ModelPredictor
import cv2
import io
import matplotlib.pyplot as plt
import tempfile
import os
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from keras import Sequential
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, Reshape, Dense, multiply, add, Permute, Lambda
from keras.layers import Input, Dense,Conv2D , MaxPooling2D, Flatten,BatchNormalization,Dropout
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report , confusion_matrix , accuracy_score
from sklearn.model_selection import train_test_split

class EvaluasiModel:
    def __init__(self, model_path, resize_shape):
        self.model_path = model_path
        self.model = self.load_model()
        self.resize_shape = resize_shape
    
    # Memuat model dari file .h5
    def load_model(self):
        model = tf.keras.models.load_model(self.model_path)
        return model
    

    # Melatih ulang model dengan augmentasi data yang disesuaikan
    def evaluate_model(self):
        # Mengambil jumlah data setiap kelas

        normal_path = "CT_scan_Brain\\Brain_Data_Organised\\Normal"

        stroke_path = "CT_scan_Brain\Brain_Data_Organised\Stroke"

        normal_folder = os.listdir(normal_path)
        stroke_folder = os.listdir(stroke_path)

        print("Images in Normal Data:",len(normal_folder))
        print("Images in Stroke Data:",len(stroke_folder))

        data = []
        for img_file in normal_folder:
            #Membuka gambar
            image = Image.open("CT_scan_Brain\\Brain_Data_Organised\\Normal\\" + img_file)
            #Ubah ukuran gambar
            image = image.resize(self.resize_shape)
            #Konversikan gambar ke RGB
            image = image.convert('RGB')
            #Mengonversi gambar ke array numpy
            image = np.array(image)
            #Menerapkan pemfilteran median untuk menghilangkan noise menggunakan OpenCV
            image = cv2.medianBlur(image, 5)  #Ukuran kernel 5
            # Menambahkan gambar yang telah direduksi ke dataset
            data.append(image)

        for img_file in stroke_folder:
            #Membuka gambar
            image = Image.open("CT_scan_Brain\Brain_Data_Organised\Stroke\\" + img_file)
            #Ubah ukuran gambar
            image = image.resize(self.resize_shape)
            #Konversikan gambar ke RGB
            image = image.convert('RGB')
            #Mengonversi gambar ke array numpy
            image = np.array(image)
            #Menerapkan pemfilteran median untuk menghilangkan noise menggunakan OpenCV
            image = cv2.medianBlur(image, 5)  #Ukuran kernel 5
            # Menambahkan gambar yang telah direduksi ke dataset
            data.append(image)

        #Mendapatkan jumlah label kelas
        normal_label = [0]*len(normal_folder)
        stroke_label = [1]*len(stroke_folder)
        Target_label = normal_label + stroke_label
        len(Target_label)

        #Mengkonversikan data gambar dan label target menjadi array

        x = np.array(data)
        y = np.array(Target_label)
        x[0]


        #Split data menjadi training dan testing (training 80% dan testing 20%)

        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,shuffle=True)

        # Memprediksi probabilitas prediksi untuk setiap kelas pada data uji menggunakan model
        y_pred_probs = self.model.predict(x_test)
        y_pred = [1 if prob >= 0.5 else 0 for prob in y_pred_probs]

        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred, target_names=["Normal", "Stroke"])


        TN, FP, FN, TP = cm.ravel()
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        sensitivitas = TP / (TP + FN)
        f1_score = 2 * (precision * recall) / (precision + recall)

        st.write("Confusion Matrix Values:")
        st.write(f"True Negative (TN): {TN}")
        st.write(f"False Positive (FP): {FP}")
        st.write(f"False Negative (FN): {FN}")
        st.write(f"True Positive (TP): {TP}")
        st.write(f"Precision: {precision:.2f}")
        st.write(f"Recall: {recall:.2f}")
        st.write(f"Sensitivitas: {sensitivitas:.2f}")
        st.write(f"Accuracy: {accuracy:.2f}")
        st.write(f"F1-score: {f1_score:.2f}")

        st.set_option('deprecation.showPyplotGlobalUse', False)

        st.subheader("Confusion Matrix")
        plt.figure(figsize=(8, 6))
        sns.set(font_scale=1.2)
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False, xticklabels=["Normal", "Stroke"], yticklabels=["Normal", "Stroke"])
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        st.pyplot()