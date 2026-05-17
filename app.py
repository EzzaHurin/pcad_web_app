import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. CONFIG & ASSETS ---
st.set_page_config(page_title="PCAD Risk Assessment", layout="wide")

# Custom CSS for the Dashboard
def local_css():
    st.markdown("""
    <style>
    .patient-banner {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 20px;
    }
    .heartbeat-icon { font-size: 30px; color: #ff4b4b; }
    .banner-item label { font-size: 12px; color: #666; display: block; }
    .banner-item span { font-weight: bold; font-size: 18px; }
    
    .risk-card {
        background: white;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .donut-outer {
        margin: 20px auto;
        width: 150px;
        height: 150px;
        border-radius: 50%;
        background: conic-gradient(var(--risk-color) var(--risk-degree), #eee 0deg);
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .donut-inner {
        width: 120px;
        height: 120px;
        background: white;
        border-radius: 50%;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    .risk-score-text { font-size: 32px; font-weight: bold; }
    .risk-label-box { padding: 10px; border-radius: 8px; font-weight: bold; margin-top: 10px; }
    
    .bio-card {
        background: #f9f9f9;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #ddd;
        margin-bottom: 10px;
    }
    .bio-title { font-size: 14px; color: #555; font-weight: 600; }
    .bio-value { font-size: 20px; font-weight: bold; }
    .bio-desc { font-size: 12px; color: #777; }
    .status-high { color: #ff4b4b; font-weight: bold; }
    .status-normal { color: #28a745; font-weight: bold; }
    .disclaimer-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 5px;
        font-size: 13px;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    # Load your original files
    try:
        model = joblib.load('pcad_model.pkl')
        preprocessor = joblib.load('preprocessor.pkl')
        return model, preprocessor
    except:
        st.error("Model or Preprocessor files not found!")
        return None, None

model, preprocessor = load_assets()

# --- 2. SESSION STATE MANAGEMENT ---
if 'page' not in st.session_state:
    st.session_state.page = 'form'
if 'patient_data' not in st.session_state:
    st.session_state.patient_data = {}
if 'risk_result' not in st.session_state:
    st.session_state.risk_result = {}

def go_to_result(): st.session_state.page = 'result'
def go_to_form(): st.session_state.page = 'form'

local_css()

# --- 3. PAGE 1: INPUT FORM ---
if st.session_state.page == 'form':
    st.title("Sistem Prediksi Risiko PCAD")
    st.write("Sila masukkan data pesakit untuk klasifikasi risiko.")

    with st.form("prediction_form"):
        # Added Patient Name
        name = st.text_input("Patient Full Name", value="John Doe")
        
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=1, max_value=120, value=50)
            bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
            gender = st.selectbox("Gender", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
            smoking = st.selectbox("Smoking Status", options=[1, 0], format_func=lambda x: "Smoker" if x == 1 else "Non-Smoker")
            crp = st.number_input("CRP Level (mg/L)", value=3.0)
            il6 = st.number_input("IL-6 Level (pg/mL)", value=5.0)

        with col2:
            vcam = st.number_input("VCAM-1 Level (ng/mL)", value=600.0)
            glutathione = st.number_input("Glutathione (mmol/L)", value=8.0)
            lipid = st.number_input("Lipid Profile", value=200.0)
            renal = st.number_input("Renal Profile", value=1.0)
            liver = st.number_input("Liver Profile", value=35.0)

        submit = st.form_submit_button("Predict Risk Level", use_container_width=True)

    if submit:
        # Prepare Data for Prediction
        input_data = pd.DataFrame([[age, bmi, gender, smoking, crp, il6, vcam, glutathione, lipid, renal, liver]], 
                            columns=['Age', 'BMI','Gender', 'Smoking_Status', 'CRP', 'IL_6', 'VCAM_1', 'Glutathione', 'Lipid_Profile', 'Renal_Profile', 'Liver_Profile'])
        
        # Original Prediction Logic
        data_scaled = preprocessor.transform(input_data)
        prediction = model.predict(data_scaled)[0]
        
        # Map Prediction to Visuals
        # Adjust mapping logic based on your specific model labels (0,1,2 or Low, Medium, High)
        res_map = {
            'High':   {'label': 'HIGH RISK', 'color': '#ff4b4b', 'degree': '300deg', 'score': 85},
            2:        {'label': 'HIGH RISK', 'color': '#ff4b4b', 'degree': '300deg', 'score': 85},
            'Medium': {'label': 'MEDIUM RISK', 'color': '#ffa500', 'degree': '180deg', 'score': 50},
            1:        {'label': 'MEDIUM RISK', 'color': '#ffa500', 'degree': '180deg', 'score': 50},
            'Low':    {'label': 'LOW RISK', 'color': '#28a745', 'degree': '60deg', 'score': 15},
            0:        {'label': 'LOW RISK', 'color': '#28a745', 'degree': '60deg', 'score': 15}
        }
        
        # Store data for the Dashboard page
        st.session_state.patient_data = {
            'name': name, 'age': age, 'bmi': bmi, 
            'smoking': "Smoker" if smoking == 1 else "Non-Smoker",
            'crp': crp, 'il6': il6, 'vcam': vcam, 'glutathione': glutathione
        }
        st.session_state.risk_result = res_map.get(prediction, res_map['Low'])
        
        go_to_result()
        st.rerun()

# --- 4. PAGE 2: RESULT DASHBOARD ---
elif st.session_state.page == 'result':
    data = st.session_state.patient_data
    res = st.session_state.risk_result

    # Navigation Header
    col_nav1, col_nav2, col_nav3 = st.columns([1, 8, 1])
    with col_nav1:
        if st.button("← Back"):
            go_to_form()
            st.rerun()
    with col_nav2:
        st.markdown("<h3 style='text-align: center; margin:0;'>❤️ CAD Risk Assessment Results</h3>", unsafe_allow_html=True)
    with col_nav3:
        if st.button("Log Out"):
            st.session_state.page = 'form' # Reset
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Patient Banner
    st.markdown(f"""
        <div class="patient-banner">
            <div class="heartbeat-icon">∿</div>
            <div class="banner-item"><label>PATIENT NAME</label><span>{data['name']}</span></div>
            <div class="banner-item"><label>AGE</label><span>{data['age']} years</span></div>
            <div class="banner-item"><label>BMI</label><span>{data['bmi']} kg/m²</span></div>
            <div class="banner-item"><label>SMOKING</label><span>{data['smoking']}</span></div>
        </div>
    """, unsafe_allow_html=True)

    # 3. Main Dashboard Layout
    left_col, right_col = st.columns([1, 2])

    with left_col:
        st.markdown(f"""
            <div class="risk-card">
                <div style="font-weight:600; color:#555;">PREMATURE CAD RISK</div>
                <div class="donut-outer" style="--risk-color: {res['color']}; --risk-degree: {res['degree']};">
                    <div class="donut-inner">
                        <div class="risk-score-text" style="color: {res['color']}">{res['score']}</div>
                        <div style="color:#888; font-size:14px;">/ 100</div>
                    </div>
                </div>
                <div class="risk-label-box" style="color: {res['color']}; background: {res['color']}20;">
                    {res['label']}
                </div>
            </div>
        """, unsafe_allow_html=True)

    with right_col:
        st.markdown('<h6>🕒 BIOMARKER ANALYSIS</h6>', unsafe_allow_html=True)
        
        def get_bio_html(label, value, unit, threshold, is_reverse=False):
            is_bad = (value < threshold) if is_reverse else (value > threshold)
            status_text = "High" if is_bad else "Normal"
            status_class = "status-high" if is_bad else "status-normal"
            
            desc = "Levels are within range."
            if label == "CRP" and is_bad: desc = "Indicates systemic inflammation."
            elif label == "VCAM-1" and is_bad: desc = "Endothelial dysfunction detected."
            elif label == "Glutathione" and is_bad: desc = "Depleted antioxidants increase stress."
            elif not is_bad: desc = "Within healthy clinical range."
            
            return f"""
                <div class="bio-card">
                    <div class="bio-title">{label}</div>
                    <div class="bio-value">{value} <span style="font-size:12px; color:#888;">{unit}</span></div>
                    <div class="bio-desc">{desc}</div>
                    <div class="bio-status {status_class}">{status_text}</div>
                </div>
            """

        row1_1, row1_2 = st.columns(2)
        with row1_1: st.markdown(get_bio_html("CRP", data['crp'], "mg/L", 3.0), unsafe_allow_html=True)
        with row1_2: st.markdown(get_bio_html("IL-6", data['il6'], "pg/mL", 5.0), unsafe_allow_html=True)
        
        row2_1, row2_2 = st.columns(2)
        with row2_1: st.markdown(get_bio_html("VCAM-1", data['vcam'], "ng/mL", 500), unsafe_allow_html=True)
        with row2_2: st.markdown(get_bio_html("Glutathione", data['glutathione'], "mmol/L", 4.0, is_reverse=True), unsafe_allow_html=True)

    st.markdown("""
        <div class="disclaimer-box">
            <b>Medical Disclaimer:</b> This assessment is for informational purposes only. 
            It should not replace professional medical diagnosis. Please consult with a cardiologist.
        </div>
    """, unsafe_allow_html=True)
    
    if st.button("Update Patient Data", type="primary", use_container_width=True):
        go_to_form()
        st.rerun()
