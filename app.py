import os
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

# Başlık ve açıklamalar
st.title("🔥 Yangın Algılama Uygulaması 🔥")
st.markdown("""
    Bu uygulama, **yangın** tespiti için çeşitli makine öğrenimi ve derin öğrenme modellerini kullanarak 
    fotoğraflar üzerinden analiz yapmaktadır. 
    📸👨‍🚒
""")

# Model ve Sonuç Dosyalarını Yükleme
def load_model_from_file(model_path):
    model = load_model(model_path)
    return model

def load_ml_model_from_file(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Yangın Algılama Modelinin Yüklenmesi
def load_fire_detection_model():
    st.write("🔄 Derin Öğrenme modelini yüklüyoruz...")
    model = load_model_from_file('fire.h5')
    st.success("✅ Derin öğrenme modeli başarıyla yüklendi!")
    return model

# SVM Modelinin Yüklenmesi
def load_svm_model():
    st.write("🔄 SVM modelini yüklüyoruz...")
    model = load_ml_model_from_file('SVM.pkl')
    st.success("✅ SVM modeli başarıyla yüklendi!")
    return model

# Model Seçimi
model_choice = st.selectbox(
    "🎯 Bir model seçin:",
    ["Derin Öğrenme Modeli (CNN)", "SVM Modeli", "Random Forest Modeli", "Logistic Regression Modeli"]
)

# Modeli Seçip Yükleme
if model_choice == "Derin Öğrenme Modeli (CNN)":
    model = load_fire_detection_model()

elif model_choice == "SVM Modeli":
    model = load_svm_model()

# Modeli Kullanarak Tahmin Yapma
def predict_image(model, img):
    img = img.resize((256, 256))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    if model_choice == "Derin Öğrenme Modeli (CNN)":
        prediction = model.predict(img_array)
        return "Yangın" if prediction[0][0] > 0.5 else "Yangın Değil"

    elif model_choice == "SVM Modeli":
        prediction = model.predict(img_array.flatten().reshape(1, -1))
        return "Yangın" if prediction == 1 else "Yangın Değil"

# Görsel Yükleme
uploaded_image = st.file_uploader("📤 Bir resim yükleyin (Yangın veya Yangın Değil)", type=["jpg", "png"])

if uploaded_image is not None:
    img = Image.open(uploaded_image)
    st.image(img, caption="Yüklenen Görsel", use_column_width=True)
    if st.button("💥 Tahmin Et"):
        prediction = predict_image(model, img)
        st.write(f"🔮 Tahmin: **{prediction}**")

# Model Sonuçları Gösterme
st.header("📊 Model Sonuçları")
if st.button("💻 Model Sonuçlarını Göster"):
    if model_choice == "SVM Modeli":
        with open('SVM_results.txt', 'r') as file:
            svm_results = file.read()
        st.text_area("SVM Sonuçları", svm_results, height=200)

    elif model_choice == "Random Forest Modeli":
        with open('Random_Forest_results.txt', 'r') as file:
            rf_results = file.read()
        st.text_area("Random Forest Sonuçları", rf_results, height=200)

    elif model_choice == "Logistic Regression Modeli":
        with open('Logistic_Regression_results.txt', 'r') as file:
            lr_results = file.read()
        st.text_area("Logistic Regression Sonuçları", lr_results, height=200)

# Veri Seti Keşfi
st.header("🔍 Veri Seti Keşfi")
st.markdown("""
    Eğitim, test ve doğrulama veri setleri hakkında daha fazla bilgi almak için aşağıdaki butonları kullanabilirsiniz.
""")

if st.button("📂 Eğitim Setini Göster"):
    # Eğitim setindeki örnek resimleri görselleştirme
    display_sample_images()

def display_sample_images():
    num_samples = 4
    for class_name in ['Fire', 'Non-Fire']:
        st.write(f"📸 **{class_name} Sınıfı**")
        filenames = os.listdir(os.path.join('fire/Train', class_name))
        
        for i in range(num_samples):
            img = Image.open(os.path.join('fire/Train', class_name, filenames[i]))
            st.image(img, caption=f"{class_name} - {i + 1}", use_column_width=True)