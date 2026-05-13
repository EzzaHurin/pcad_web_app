import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Load model dan preprocessor
@st.cache_resource
def load_assets():
    model = joblib.load('pcad_model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    return model, preprocessor

model, preprocessor = load_assets()

st.title("Sistem Prediksi Risiko PCAD")
st.write("Sila masukkan data pesakit untuk klasifikasi risiko (Low, Medium, High).")

# 2. Bina input form
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
    gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
    smoking = st.selectbox("Smoking Status", options=[0, 1], format_func=lambda x: "Smoker" if x == 1 else "Non-Smoker")
    crp = st.number_input("CRP Level", value=3.0)
    il6 = st.number_input("IL-6 Level", value=5.0)

with col2:
    vcam = st.number_input("VCAM-1 Level", value=600.0)
    glutathione = st.number_input("Glutathione", value=8.0)
    lipid = st.number_input("Lipid Profile", value=200.0)
    renal = st.number_input("Renal Profile", value=1.0)
    liver = st.number_input("Liver Profile", value=35.0)

# 3. Proses Prediksi
if st.button("Predict Risk Level"):
    # Susun data dalam DataFrame (SUSUNAN MESTI SAMA DENGAN MODEL LATIHAN)
    # Pastikan turutan kolum di sini sama seperti dalam X_train semasa anda melatih model
    data = pd.DataFrame([[age, bmi, gender, smoking, crp, il6, vcam, glutathione, lipid, renal, liver]], 
                        columns=['Age', 'BMI', 'Gender', 'Smoking_Status', 'CRP', 'IL_6', 'VCAM_1', 'Glutathione', 'Lipid_Profile', 'Renal_Profile', 'Liver_Profile'])
    
    # Transform guna preprocessor asal
    data_scaled = preprocessor.transform(data)
        
    # Buat prediksi
    prediction = model.predict(data_scaled)
        
    # Papar keputusan dengan warna
    risk = prediction[0]
    st.subheader("Hasil Analisis:")
    if risk == 'High' or risk == 2: # Bergantung kepada label model anda
        st.error(f"Tahap Risiko: HIGH")
    elif risk == 'Medium' or risk == 1:
        st.warning(f"Tahap Risiko: MEDIUM")
    else:
        st.success(f"Tahap Risiko: LOW")
            
   
