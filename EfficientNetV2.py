import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalMaxPool2D

class EfficientNetV2Model:
    #Mengambil jumlah data setiap kelas

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
        image = image.resize((224,224))
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
        image = image.resize((224,224))
        #Konversikan gambar ke RGB
        image = image.convert('RGB')
        #Mengonversi gambar ke array numpy
        image = np.array(image)
        #Menerapkan pemfilteran median untuk menghilangkan noise menggunakan OpenCV
        image = cv2.medianBlur(image, 5)  #Ukuran kernel 5
        # Menambahkan gambar yang telah direduksi ke dataset
        data.append(image)

    print(len(data))
    print(data[0])
    print(data[0].shape)

    #Mendapatkan jumlah label kelas
    normal_label = [0]*len(normal_folder)
    stroke_label = [1]*len(stroke_folder)
    Target_label = normal_label + stroke_label
    len(Target_label)

    #Mengkonversikan data gambar dan label target menjadi array

    x = np.array(data)
    y = np.array(Target_label)
    x[0]

    print(y)

    #Split data menjadi training dan testing (training 80% dan testing 20%)

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,shuffle=True)
    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    # Inisialisasi ImageDataGenerator untuk augmentasi data pada data pelatihan
    train_datagen = ImageDataGenerator(
        rescale=1./255,  # Normalisasi nilai pixel (opsional)
        rotation_range=20,  # Rentang rotasi gambar
        width_shift_range=0.1,  # Pergeseran horizontal gambar
        height_shift_range=0.1,  # Pergeseran vertikal gambar
        shear_range=0.2,  # Shear mapping
        zoom_range=0.2,  # Zoom-in/Zoom-out gambar
        vertical_flip=True,
        horizontal_flip=True,  # Flipping gambar secara horizontal
        fill_mode='nearest'  # Metode pengisian piksel saat terjadi transformasi
    )

    # Normalisasi data testing (tidak perlu augmentasi data pada data testing)
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Menerapkan augmentasi pada data training
    augmented_train_data = []
    augmented_train_labels = []
    batch_size = 128

    for x_batch, y_batch in train_datagen.flow(x_train, y_train, batch_size=batch_size):
        augmented_train_data.append(x_batch)
        augmented_train_labels.append(y_batch)
        if len(augmented_train_data) >= len(x_train) / batch_size:
            break

    augmented_train_data = np.concatenate(augmented_train_data)
    augmented_train_labels = np.concatenate(augmented_train_labels)

    # Normalisasi data testing
    x_test_normalized = test_datagen.flow(x_test, batch_size=len(x_test), shuffle=False)[0]

    # Pastikan augmented_train_data, augmented_train_labels, dan x_test_normalized dapat digunakan
    print(augmented_train_data.shape)
    print(augmented_train_labels.shape)
    print(x_test_normalized.shape)

    #Merepresentasi visual sampel data
    class_labels = ["Normal","Stroke"]
    plt.figure(figsize=(16,24))
    for i in range(10):
        plt.subplot(6,4,i+1)
        plt.imshow(x_train[i])
        plt.title(f"Actual Label:{class_labels[y_train[i]]}")
        plt.axis("off")

    base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(


        weights = 'imagenet',

        include_top=False,

        input_shape=(224, 224, 3)
    )

    avg = keras.layers.GlobalAveragePooling2D()(base_model.output)

    output = keras.layers.Dense(1,activation='sigmoid')(avg)

    model = keras.Model(inputs = base_model.input,outputs = output)

    for layer in base_model.layers[:5]:
        layer.trainable = False

    #model.build()
    model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    history = model.fit(x_train,y_train, epochs=25, validation_data=(x_test,y_test))

    # Menyimpan model ke dalam file HDF5 (H5)
    model.save('modelefficientnetv2.h5')

    loss, acc = model.evaluate(x_test,y_test)
    print("Loss on Test Data:",loss)
    print("Accuracy on Test Data:",acc)

    loss, acc = model.evaluate(x_train,y_train)
    print("Loss on Train Data:",loss)
    print("Accuracy on Train Data:",acc)

    y_pred_test = model.predict(x_test)
    y_pred_test_label = [1 if i>=0.5 else 0 for i in y_pred_test]

    print("Actual Label:",y_test[:10])
    print("Predicted Label:",y_pred_test_label[:10])

    print("-----Metrics Evaluation On Test Data -----")
    print()
    print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred_test_label))
    print()
    print("Classification Report:\n",classification_report(y_test,y_pred_test_label))

    class_labels = ["Normal","Stroke"]
    plt.figure(figsize=(16,32))
    for i in range(30):
        plt.subplot(8,4,i+1)
        plt.imshow(x_test[i])
        plt.title(f"Actual:{class_labels[y_test[i]]}\nPredicted:{class_labels[y_pred_test_label[i]]}")
        plt.axis("off")
