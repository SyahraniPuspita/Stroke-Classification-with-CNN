import streamlit as st
from PIL import Image
import numpy as np
from PredictImage import ModelPredictor
from EvaluateModel import EvaluasiModel
from oznetcoba import OznetModel
from efficientnetv2coba import EfficientNetV2Model
from resnet50v2coba import ResNet50V2Model
import cv2
import io
import matplotlib.pyplot as plt
import tempfile
import os


class StreamlitApp:
    def __init__(self):
        self.model_predictor = None

    def run(self):

        st.write("""
        <style>
            .centered-title {
                text-align: center;
            }
        </style>
        """, unsafe_allow_html=True)

        # Menampilkan judul dengan kelas CSS yang ditentukan sebelumnya
        st.write('<h1 class="centered-title">Prediksi Penyakit Stroke dan Evaluasi Model</h1>', unsafe_allow_html=True)

        st.sidebar.image("logo-unsri.png", width=140)

        # Sidebar for navigation
        page = st.sidebar.radio("Fitur", ["Training Model","Prediksi Gambar", "Evaluasi Model"])

        if page == "Prediksi Gambar":
            self.image_prediction_page()
        elif page == "Evaluasi Model":
            self.model_evaluation_page()
        elif page == "Training Model":
            self.model_training_page()

        # Tambahkan footer menggunakan Markdown
        st.markdown(
            """
            <footer style="text-align:center; margin-top:20px;">
                Copyright &copy; 2024 Syahrani Puspita Wijaya Sriwijaya University. All rights reserved.
            </footer>
            """,
            unsafe_allow_html=True
        )

    def image_prediction_page(self):
        st.subheader("Prediksi dengan Convolutional Neral Network (CNN) dan Explainable Artificial Intelligence (XAI)")

        self.model_path = st.selectbox(
            'Pilih Model:',
            ('OzNet', 
            'ResNet50V2', 
            'EfficientNetV2'))

        st.write('Model terpilih:', self.model_path)

        if self.model_path == 'OzNet':
            self.model_predictor = ModelPredictor('D:\ProjectSkripsi\modeloznet.h5')
            uploaded_file = st.file_uploader("Silahkan Upload Gambar Anda",  accept_multiple_files=True, type=["jpg", "jpeg", "png"])

            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, caption="Gambar Terunggah", use_column_width=True)

                predicted_label, _, img_array = self.model_predictor.predict_image(uploaded_file)
                target_layer = 'conv2d_6'
                heatmap, _ = self.model_predictor.generate_gradcam(img_array, target_layer)

                img_array = np.array(image)
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                img = cv2.resize(img_array, (227, 227))
                heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
                heatmap = np.float32(heatmap) / 255
                superimposed_img = heatmap + np.float32(img / 255)
                superimposed_img = superimposed_img / np.max(superimposed_img)

                plt.imshow(superimposed_img)
                plt.axis('off')
                plt.colorbar(ticks=[0, 0.5, 1], orientation='vertical')
                plt.savefig('temp_heatmap.png')

                st.subheader("Hasil Prediksi:")
                st.write(f"Label Prediksi: {predicted_label}")

                img_with_colorbar = Image.open('temp_heatmap.png')
                img_with_colorbar.thumbnail((400, 400))
                st.image(img_with_colorbar, caption="Heatmap dengan GradCAM Weight", use_column_width=True)

        if self.model_path == 'ResNet50V2':
            self.model_predictor = ModelPredictor('D:\ProjectSkripsi\modelresnet50v2.h5')
            uploaded_files = st.file_uploader("Silahkan Upload Gambar Anda", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

            if uploaded_files:
                # Inisialisasi colorbar di luar loop
                fig, ax = plt.subplots()
                cbar = fig.colorbar(ax.imshow(np.zeros((0, 10))), ticks=[0, 0.5, 1], orientation='vertical')

                for uploaded_file in uploaded_files:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Gambar Terunggah", use_column_width=True)

                    predicted_label, _, img_array = self.model_predictor.predict_image(uploaded_file)
                    target_layer = 'post_relu'
                    heatmap, _ = self.model_predictor.generate_gradcam(img_array, target_layer)

                    img_array = np.array(image)
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    img = cv2.resize(img_array, (224, 224))
                    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
                    heatmap = np.float32(heatmap) / 255
                    superimposed_img = heatmap + np.float32(img / 255)
                    superimposed_img = superimposed_img / np.max(superimposed_img)

                    plt.imshow(superimposed_img)
                    plt.axis('off')
                    plt.savefig('temp_heatmap.png')

                    st.subheader("Hasil Prediksi:")
                    st.write(f"Label Prediksi: {predicted_label}")

                    img_with_colorbar = Image.open('temp_heatmap.png')
                    img_with_colorbar.thumbnail((400, 400))
                    st.image(img_with_colorbar, caption="Heatmap dengan GradCAM Weight", use_column_width=True)

                    os.remove(f'temp_heatmap.png')

        if self.model_path == 'EfficientNetV2':
            self.model_predictor = ModelPredictor('D:\ProjectSkripsi\modelefficientnetv2.h5')
            uploaded_file = st.file_uploader("Silahkan Upload Gambar Anda", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, caption="Gambar Terunggah", use_column_width=True)

                predicted_label, _, img_array = self.model_predictor.predict_image(uploaded_file)
                target_layer = 'top_activation'
                heatmap, _ = self.model_predictor.generate_gradcam(img_array, target_layer)

                img_array = np.array(image)
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                img = cv2.resize(img_array, (224, 224))
                heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
                heatmap = np.float32(heatmap) / 255
                superimposed_img = heatmap + np.float32(img / 255)
                superimposed_img = superimposed_img / np.max(superimposed_img)

                plt.imshow(superimposed_img)
                plt.axis('off')
                plt.colorbar(ticks=[0, 0.5, 1], orientation='vertical')
                plt.savefig('temp_heatmap.png')

                st.subheader("Hasil Prediksi:")
                st.write(f"Label Prediksi: {predicted_label}")

                img_with_colorbar = Image.open('temp_heatmap.png')
                img_with_colorbar.thumbnail((400, 400))
                st.image(img_with_colorbar, caption="Heatmap dengan GradCAM Weight", use_column_width=True)


    def model_evaluation_page(self):
        st.title("Evaluasi Model")

        model_option = st.selectbox(
                'Pilih Model:',
                ('ResNet50V2', 'OzNet', 'EfficientNetV2'))

        st.write('Model terpilih:', model_option)

        if model_option == 'ResNet50V2':
            model_path = "D:\ProjectSkripsi\modelresnet50v2.h5"
            resize_shape = (224, 224)
            evaluasi_model = EvaluasiModel(model_path, resize_shape)
            evaluasi_model.evaluate_model()
        elif model_option == 'OzNet':
            model_path = "D:\ProjectSkripsi\modeloznet.h5"
            resize_shape = (227, 227)
            evaluasi_model = EvaluasiModel(model_path, resize_shape)
            evaluasi_model.evaluate_model()
            
        elif model_option == 'EfficientNetV2':
            model_path = "D:\ProjectSkripsi\modelefficientnetv2.h5"
            resize_shape = (224, 224)
            evaluasi_model = EvaluasiModel(model_path, resize_shape)
            evaluasi_model.evaluate_model()

    def model_training_page(self):
        st.title("Training Model")
    
        model_option = st.selectbox("Pilih Model", ["OzNet", "ResNet50V2" , "EfficientNetV2"])

        # Mendefinisikan antarmuka pengguna untuk parameter augmentasi data
        augmentation_params = None
        with st.expander("Pengaturan Augmentasi Data"):
            rotation_range = st.slider("Rentang Rotasi", 0.0, 360.0, 30.0)
            width_shift_range = st.slider("Rentang Geser Lebar", 0.0, 1.0, 0.1)
            height_shift_range = st.slider("Rentang Geser Tinggi", 0.0, 1.0, 0.1)
            shear_range = st.slider("Rentang Shear", 0.0, 1.0, 0.1)
            zoom_range = st.slider("Rentang Zoom", 0.0, 1.0, 0.1)
            horizontal_flip = st.checkbox("Flip Horizontal", True)
            vertical_flip = st.checkbox("Flip Vertikal", False)

            # Tombol "Proses" untuk memulai training set
            if st.button("Latih Model"):
                augmentation_params = {
                    "rotation_range": rotation_range,
                    "width_shift_range": width_shift_range,
                    "height_shift_range": height_shift_range,
                    "shear_range": shear_range,
                    "zoom_range": zoom_range,
                    "horizontal_flip": horizontal_flip,
                    "vertical_flip": vertical_flip
                }
            
        # Jika parameter augmentasi sudah diatur, lakukan training model
        if augmentation_params:
            # Memanggil metode `retrain_model` dari kelas model yang dipilih
            if model_option == 'OzNet':
                OznetModel.train_model(augmentation_params)
            elif model_option == 'ResNet50V2':
                ResNet50V2Model.train_model(augmentation_params)
            elif model_option == 'EfficientNetV2':
                EfficientNetV2Model.train_model(augmentation_params)
            
            # Menampilkan pemberitahuan bahwa model telah tersimpan
            st.success("Model telah berhasil dilatih dan disimpan.")


  
def main():
    app = StreamlitApp()
    app.run()

if __name__ == "__main__":
    main()

# class StreamlitApp:
#     def __init__(self):
#         self.model_path = 'D:\ProjectSkripsi\modelresnet50v2.h5'
#         self.model_predictor = ModelPredictor(self.model_path)
#         self.image_selector = ImageSelector(self.model_path)

#     def run(self):
#         st.title("Predict with GradCAM")

#         uploaded_file = st.file_uploader("Silahkan Upload Gambar Anda", type=["jpg", "jpeg", "png"])

#         if uploaded_file:
#             # Display the uploaded image
#             image = Image.open(uploaded_file)
#             st.image(image, caption="Uploaded Image", use_column_width=True)

#             predicted_label, confidence_score, img_array = self.model_predictor.predict_image(uploaded_file)
#             target_layer = 'post_relu'
#             heatmap, weights = self.model_predictor.generate_gradcam(img_array, target_layer)

#             img = cv2.imread(uploaded_file)
#             img = cv2.resize(img, (224, 224))
#             heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
#             heatmap = np.float32(heatmap) / 255
#             superimposed_img = heatmap + np.float32(img / 255)
#             superimposed_img = superimposed_img / np.max(superimposed_img)

#             # Save the image with colorbar
#             plt.imshow(superimposed_img)
#             plt.axis('off')
#             plt.colorbar(ticks=[0, 0.5, 1], orientation='vertical')
#             plt.savefig('temp_heatmap.png')

#             # Convert the image to a numpy array
#             # img_array = np.array(image)
#             img_array = np.array(image.resize((224, 224)))


#             st.subheader("Hasil Prediksi:")
#             st.write(f"Predicted Label: {predicted_label}")

#             # Display the heatmap image with colorbar
#             img_with_colorbar = Image.open('temp_heatmap.png')
#             img_with_colorbar.thumbnail((400, 400))  # Adjust the image size
#             st.image(img_with_colorbar, caption="Heatmap with GradCAM Weight", use_column_width=True)

# def main():
#     app = StreamlitApp()
#     app.run()
    

# if __name__ == "__main__":
#     main()
