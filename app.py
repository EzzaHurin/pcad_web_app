import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# --- 1. CONFIG & ASSETS ---
st.set_page_config(page_title="PCAD Risk Assessment", layout="wide")

# Custom CSS for the Dashboard
def local_css():
    st.markdown("""
    <style>
    /* ---- SIDEBAR STYLES ---- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f36 0%, #16213e 100%);
    }
    [data-testid="stSidebar"] * { color: #e0e6f0 !important; }

    .sidebar-user-card {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 8px;
        text-align: center;
    }
    .sidebar-avatar {
        width: 56px;
        height: 56px;
        border-radius: 50%;
        background: linear-gradient(135deg, #ff4b4b, #ff9a9e);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
        margin: 0 auto 10px auto;
        box-shadow: 0 4px 12px rgba(255,75,75,0.4);
    }
    .sidebar-username {
        font-weight: 700;
        font-size: 15px;
        color: #ffffff !important;
        margin: 0;
    }
    .sidebar-role {
        font-size: 11px;
        color: #9aafc7 !important;
        margin: 2px 0 0 0;
    }
    .sidebar-divider {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.1);
        margin: 12px 0;
    }
    .nav-label {
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1.5px;
        color: #6b82a6 !important;
        text-transform: uppercase;
        padding: 4px 0 8px 0;
    }

    /* ---- MAIN PAGE STYLES ---- */
    .patient-banner {
        background: linear-gradient(135deg, #f0f4ff 0%, #e8f0fe 100%);
        padding: 20px 24px;
        border-radius: 12px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 20px;
        border: 1px solid #d0dcf5;
    }
    .heartbeat-icon { font-size: 30px; color: #ff4b4b; }
    .banner-item label { font-size: 11px; color: #666; display: block; text-transform: uppercase; letter-spacing: 0.5px; }
    .banner-item span { font-weight: bold; font-size: 18px; color: #1a1f36; }

    .risk-card {
        background: white;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid #f0f0f0;
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
        border-radius: 8px;
        font-size: 13px;
        margin-top: 20px;
        border-left: 4px solid #ffc107;
    }

    /* ---- DASHBOARD / MAIN MENU STYLES ---- */
    .dash-metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        border: 1px solid #f0f0f0;
        transition: transform 0.2s;
    }
    .dash-metric-card:hover { transform: translateY(-2px); }
    .dash-metric-icon { font-size: 36px; margin-bottom: 8px; }
    .dash-metric-value { font-size: 28px; font-weight: 800; color: #1a1f36; }
    .dash-metric-label { font-size: 13px; color: #888; margin-top: 4px; }

    .info-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        border: 1px solid #f0f0f0;
        margin-bottom: 16px;
    }
    .info-card h4 { margin: 0 0 8px 0; color: #1a1f36; }
    .info-card p { margin: 0; font-size: 14px; color: #555; line-height: 1.6; }

    /* ---- PATIENT LIST STYLES ---- */
    .db-status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
    .db-connected { background: #d4edda; color: #155724; }
    .db-disconnected { background: #f8d7da; color: #721c24; }

    /* Hide default streamlit elements in sidebar nav */
    div[data-testid="stSidebarUserContent"] .stButton button {
        width: 100%;
        text-align: left;
        background: transparent;
        border: none;
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 14px;
        font-weight: 500;
        color: #c8d6ea !important;
        transition: background 0.2s;
        cursor: pointer;
    }
    div[data-testid="stSidebarUserContent"] .stButton button:hover {
        background: rgba(255,255,255,0.1) !important;
        color: #ffffff !important;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_assets():
    try:
        model = joblib.load('pcad_model.pkl')
        preprocessor = joblib.load('preprocessor.pkl')
        return model, preprocessor
    except:
        return None, None

model, preprocessor = load_assets()

# --- 2. SESSION STATE MANAGEMENT ---
if 'page' not in st.session_state:
    st.session_state.page = 'main_menu'
if 'patient_data' not in st.session_state:
    st.session_state.patient_data = {}
if 'risk_result' not in st.session_state:
    st.session_state.risk_result = {}
if 'shap_input' not in st.session_state:
    st.session_state.shap_input = None      # raw (unscaled) DataFrame row
if 'shap_scaled' not in st.session_state:
    st.session_state.shap_scaled = None     # preprocessed array row
# Simulated logged-in user (replace with real auth as needed)
if 'logged_in_user' not in st.session_state:
    st.session_state.logged_in_user = {'name': 'Dr. Ahmad Faris', 'role': 'Cardiologist'}

local_css()

# --- 3. SIDEBAR ---
with st.sidebar:
    user = st.session_state.logged_in_user
    initials = "".join([w[0].upper() for w in user['name'].split()[:2]])

    # User Card
    st.markdown(f"""
        <div class="sidebar-user-card">
            <div class="sidebar-avatar">{initials}</div>
            <p class="sidebar-username">{user['name']}</p>
            <p class="sidebar-role">{user['role']}</p>
        </div>
    """, unsafe_allow_html=True)

    # Logout Button
    if st.button("🔓  Log Out", key="logout_btn", use_container_width=True):
        st.session_state.page = 'main_menu'
        st.session_state.patient_data = {}
        st.session_state.risk_result = {}
        st.rerun()

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.markdown('<div class="nav-label">Navigation</div>', unsafe_allow_html=True)

    # Nav Buttons
    nav_items = [
        ("🏠  Main Menu",        "main_menu"),
        ("🫀  Predict PCAD",     "form"),
        ("🗃️  Patient Data List", "patient_list"),
    ]
    for label, page_key in nav_items:
        is_active = st.session_state.page == page_key
        btn_label = f"**{label}**" if is_active else label
        if st.button(btn_label, key=f"nav_{page_key}", use_container_width=True):
            st.session_state.page = page_key
            st.rerun()

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:11px; color:#4a637a; text-align:center;">PCAD Risk System v2.0</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE: MAIN MENU (Dashboard Info)
# ─────────────────────────────────────────────
if st.session_state.page == 'main_menu':
    st.markdown("## 🏠 Main Menu — Dashboard")
    st.markdown("Welcome back, **{}**. Here's an overview of the PCAD Risk Assessment System.".format(
        st.session_state.logged_in_user['name']))
    st.markdown("<br>", unsafe_allow_html=True)

    # Summary Metrics
    m1, m2, m3, m4 = st.columns(4)
    metrics = [
        ("🫀", "3", "Total Biomarkers Tracked"),
        ("📋", "11", "Input Parameters"),
        ("⚠️", "3", "Risk Categories"),
        ("🏥", "MySQL", "Patient DB Backend"),
    ]
    for col, (icon, val, label) in zip([m1, m2, m3, m4], metrics):
        with col:
            st.markdown(f"""
                <div class="dash-metric-card">
                    <div class="dash-metric-icon">{icon}</div>
                    <div class="dash-metric-value">{val}</div>
                    <div class="dash-metric-label">{label}</div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📌 About This System")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
            <div class="info-card">
                <h4>🎯 Purpose</h4>
                <p>This system predicts the risk level of Premature Coronary Artery Disease (PCAD) 
                using patient biomarkers and clinical data. It leverages a trained machine learning 
                model to classify patients into Low, Medium, or High risk categories.</p>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("""
            <div class="info-card">
                <h4>🔬 Key Biomarkers Assessed</h4>
                <p><b>CRP</b> — C-Reactive Protein (inflammation marker)<br>
                <b>IL-6</b> — Interleukin-6 (cytokine inflammatory marker)<br>
                <b>VCAM-1</b> — Vascular Cell Adhesion Molecule<br>
                <b>Glutathione</b> — Antioxidant stress indicator</p>
            </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
            <div class="info-card">
                <h4>📊 Risk Classification</h4>
                <p>
                <span style="color:#28a745; font-weight:700;">● LOW RISK</span> — Score ~15: Biomarkers within healthy range.<br><br>
                <span style="color:#ffa500; font-weight:700;">● MEDIUM RISK</span> — Score ~50: Some biomarkers elevated; lifestyle changes advised.<br><br>
                <span style="color:#ff4b4b; font-weight:700;">● HIGH RISK</span> — Score ~85: Multiple markers elevated; urgent cardiology consult recommended.
                </p>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("""
            <div class="info-card">
                <h4>🚀 Quick Actions</h4>
                <p>Use the sidebar to navigate between pages:</p>
                <p>→ <b>Predict PCAD</b> to enter new patient data<br>
                → <b>Patient Data List</b> to browse records from the database</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("""
        <div class="disclaimer-box">
            <b>⚕️ Medical Disclaimer:</b> This system is a clinical decision-support tool only. 
            All risk predictions must be reviewed and confirmed by a qualified cardiologist before 
            any clinical action is taken.
        </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE: PREDICT PCAD (Patient Data Entry)
# ─────────────────────────────────────────────
elif st.session_state.page == 'form':
    st.markdown("## 🫀 Predict PCAD — Patient Data Entry")
    st.write("Enter the patient's clinical data below to generate a risk classification.")

    with st.form("prediction_form"):
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

        submit = st.form_submit_button("Predict Risk Level", use_container_width=True, type="primary")

    if submit:
        input_data = pd.DataFrame(
            [[age, bmi, gender, smoking, crp, il6, vcam, glutathione, lipid, renal, liver]],
            columns=['Age', 'BMI', 'Gender', 'Smoking_Status', 'CRP', 'IL_6',
                     'VCAM_1', 'Glutathione', 'Lipid_Profile', 'Renal_Profile', 'Liver_Profile']
        )

        if model is not None and preprocessor is not None:
            data_scaled = preprocessor.transform(input_data)
            prediction = model.predict(data_scaled)[0]
            st.session_state.shap_input  = input_data
            st.session_state.shap_scaled = data_scaled
        else:
            # Demo fallback when model files are missing
            st.warning("⚠️ Model files not found. Showing demo result.")
            prediction = 'High'

        res_map = {
            'High':   {'label': 'HIGH RISK',   'color': '#ff4b4b', 'degree': '300deg', 'score': 85},
            2:        {'label': 'HIGH RISK',   'color': '#ff4b4b', 'degree': '300deg', 'score': 85},
            'Medium': {'label': 'MEDIUM RISK', 'color': '#ffa500', 'degree': '180deg', 'score': 50},
            1:        {'label': 'MEDIUM RISK', 'color': '#ffa500', 'degree': '180deg', 'score': 50},
            'Low':    {'label': 'LOW RISK',    'color': '#28a745', 'degree': '60deg',  'score': 15},
            0:        {'label': 'LOW RISK',    'color': '#28a745', 'degree': '60deg',  'score': 15},
        }

        st.session_state.patient_data = {
            'name': name, 'age': age, 'bmi': bmi,
            'smoking': "Smoker" if smoking == 1 else "Non-Smoker",
            'crp': crp, 'il6': il6, 'vcam': vcam, 'glutathione': glutathione
        }
        st.session_state.risk_result = res_map.get(prediction, res_map['Low'])
        st.session_state.page = 'result'
        st.rerun()


# ─────────────────────────────────────────────
# PAGE: RESULT DASHBOARD
# ─────────────────────────────────────────────
elif st.session_state.page == 'result':
    data = st.session_state.patient_data
    res  = st.session_state.risk_result

    st.markdown("## ❤️ CAD Risk Assessment Results")
    st.markdown("<br>", unsafe_allow_html=True)

    # Patient Banner
    st.markdown(f"""
        <div class="patient-banner">
            <div class="heartbeat-icon">∿</div>
            <div class="banner-item"><label>PATIENT NAME</label><span>{data['name']}</span></div>
            <div class="banner-item"><label>AGE</label><span>{data['age']} years</span></div>
            <div class="banner-item"><label>BMI</label><span>{data['bmi']} kg/m²</span></div>
            <div class="banner-item"><label>SMOKING</label><span>{data['smoking']}</span></div>
        </div>
    """, unsafe_allow_html=True)

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
            status_text  = "High"   if is_bad else "Normal"
            status_class = "status-high" if is_bad else "status-normal"
            desc = "Levels are within range."
            if label == "CRP"        and is_bad: desc = "Indicates systemic inflammation."
            elif label == "VCAM-1"   and is_bad: desc = "Endothelial dysfunction detected."
            elif label == "Glutathione" and is_bad: desc = "Depleted antioxidants increase stress."
            elif not is_bad: desc = "Within healthy clinical range."
            return f"""
                <div class="bio-card">
                    <div class="bio-title">{label}</div>
                    <div class="bio-value">{value} <span style="font-size:12px; color:#888;">{unit}</span></div>
                    <div class="bio-desc">{desc}</div>
                    <div class="{status_class}">{status_text}</div>
                </div>
            """

        r1a, r1b = st.columns(2)
        with r1a: st.markdown(get_bio_html("CRP",    data['crp'],         "mg/L",   3.0),          unsafe_allow_html=True)
        with r1b: st.markdown(get_bio_html("IL-6",   data['il6'],         "pg/mL",  5.0),          unsafe_allow_html=True)
        r2a, r2b = st.columns(2)
        with r2a: st.markdown(get_bio_html("VCAM-1", data['vcam'],        "ng/mL",  500),          unsafe_allow_html=True)
        with r2b: st.markdown(get_bio_html("Glutathione", data['glutathione'], "mmol/L", 4.0, True), unsafe_allow_html=True)

    st.markdown("""
        <div class="disclaimer-box">
            <b>⚕️ Medical Disclaimer:</b> This assessment is for informational purposes only.
            It should not replace professional medical diagnosis. Please consult with a cardiologist.
        </div>
    """, unsafe_allow_html=True)

    # ── SHAP ANALYSIS SECTION ─────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 🔍 AI Explanation — Why This Risk Level?")
    st.markdown("SHAP (SHapley Additive exPlanations) shows **which features pushed the prediction** toward or away from the predicted risk class.")

    if not SHAP_AVAILABLE:
        st.warning("⚠️ SHAP library not installed. Run `pip install shap` and restart.")

    elif model is None or preprocessor is None:
        st.info("ℹ️ SHAP analysis requires the trained model files (`pcad_model.pkl` and `preprocessor.pkl`).")

    elif st.session_state.shap_scaled is None:
        st.info("ℹ️ Submit patient data first to see the SHAP explanation.")

    else:
        feature_names = [
            'Age', 'BMI', 'Gender', 'Smoking Status',
            'CRP', 'IL-6', 'VCAM-1', 'Glutathione',
            'Lipid Profile', 'Renal Profile', 'Liver Profile'
        ]

        @st.cache_resource
        def get_shap_explainer(_mdl):
            try:
                return shap.TreeExplainer(_mdl), "tree"
            except Exception:
                try:
                    return shap.Explainer(_mdl), "generic"
                except Exception as e:
                    return None, str(e)

        with st.spinner("Calculating SHAP values..."):
            try:
                explainer, exp_type = get_shap_explainer(model)

                if explainer is None:
                    st.warning(f"Could not build SHAP explainer: {exp_type}")
                else:
                    raw_vals      = st.session_state.shap_input.values[0]
                    pred_label    = res.get("label", "LOW RISK")
                    class_idx_map = {"LOW RISK": 0, "MEDIUM RISK": 1, "HIGH RISK": 2}
                    class_idx     = class_idx_map.get(pred_label, 0)
                    n_classes     = len(class_idx_map)

                    if exp_type == "tree":
                        raw_shap = explainer.shap_values(st.session_state.shap_scaled)
                    else:
                        exp_obj  = explainer(st.session_state.shap_scaled)
                        raw_shap = exp_obj.values

                    # normalise to 1-D float array for the predicted class
                    if isinstance(raw_shap, list):
                        elem = np.array(raw_shap[class_idx], dtype=float)
                        sv   = elem.flatten() if elem.ndim == 1 else elem[0]
                    else:
                        arr = np.array(raw_shap, dtype=float)
                        if arr.ndim == 1:
                            sv = arr
                        elif arr.ndim == 2:
                            sv = arr[0]
                        elif arr.ndim == 3:
                            if arr.shape[0] == n_classes:
                                sv = arr[class_idx, 0, :]
                            elif arr.shape[2] == n_classes:
                                sv = arr[0, :, class_idx]
                            else:
                                sv = arr[0, 0, :]
                        else:
                            sv = arr.flatten()

                    sv = sv.astype(float)

                # ── Tab layout ───────────────────────────────────────────────
                    tab1, tab2, tab3 = st.tabs(["Waterfall Chart", "Bar Chart", "Feature Table"])

                    # TAB 1: Waterfall
                    with tab1:
                        st.markdown("**Waterfall chart** — each bar shows how much a feature increased (red) or decreased (blue) the risk score.")

                        sorted_idx  = np.argsort(np.abs(sv))[::-1]
                        top_n       = min(11, len(sorted_idx))
                        idx_top     = sorted_idx[:top_n][::-1]

                        feat_labels = [f"{feature_names[i]}\n= {raw_vals[i]:.2f}" for i in idx_top]
                        values      = sv[idx_top]
                        colors      = ['#ff4b4b' if v > 0 else '#4b8bff' for v in values]

                        fig_wf, ax_wf = plt.subplots(figsize=(9, 5))
                        bars = ax_wf.barh(range(len(values)), values, color=colors, edgecolor='white', height=0.6)
                        ax_wf.set_yticks(range(len(values)))
                        ax_wf.set_yticklabels(feat_labels, fontsize=9)
                        ax_wf.axvline(0, color='black', linewidth=0.8)
                        ax_wf.set_xlabel("SHAP Value (impact on model output)", fontsize=10)
                        ax_wf.set_title(f"Feature Contributions - {pred_label}", fontsize=12, fontweight='bold', pad=12)
                        for bar, val in zip(bars, values):
                            offset = 0.003 if val >= 0 else -0.003
                            ha     = 'left' if val >= 0 else 'right'
                            ax_wf.text(val + offset, bar.get_y() + bar.get_height() / 2,
                                       f"{val:+.3f}", va='center', ha=ha, fontsize=8)
                        red_patch  = mpatches.Patch(color='#ff4b4b', label='Increases risk')
                        blue_patch = mpatches.Patch(color='#4b8bff', label='Decreases risk')
                        ax_wf.legend(handles=[red_patch, blue_patch], loc='lower right', fontsize=9)
                        fig_wf.tight_layout()
                        st.pyplot(fig_wf)
                        plt.close(fig_wf)

                    # TAB 2: Bar chart
                    with tab2:
                        st.markdown("**Bar chart** — absolute SHAP value per feature; longer bar = more influential.")
                        abs_sv     = np.abs(sv)
                        sorted_abs = np.argsort(abs_sv)
                        feat_bar   = [feature_names[i] for i in sorted_abs]
                        vals_bar   = abs_sv[sorted_abs]
                        fig_bar, ax_bar = plt.subplots(figsize=(9, 5))
                        bar_colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(vals_bar)))
                        ax_bar.barh(range(len(vals_bar)), vals_bar, color=bar_colors, edgecolor='white', height=0.6)
                        ax_bar.set_yticks(range(len(vals_bar)))
                        ax_bar.set_yticklabels(feat_bar, fontsize=10)
                        ax_bar.set_xlabel("Mean |SHAP Value|", fontsize=10)
                        ax_bar.set_title("Feature Importance (SHAP)", fontsize=12, fontweight='bold', pad=12)
                        for i, v in enumerate(vals_bar):
                            ax_bar.text(v + 0.001, i, f"{v:.3f}", va='center', fontsize=8)
                        fig_bar.tight_layout()
                        st.pyplot(fig_bar)
                        plt.close(fig_bar)

                    # TAB 3: Table
                    with tab3:
                        st.markdown("**Detailed breakdown** of SHAP values per feature.")
                        shap_df = pd.DataFrame({
                            'Feature':       feature_names,
                            'Patient Value': [f"{v:.2f}" for v in raw_vals],
                            'SHAP Value':    [round(float(v), 4) for v in sv],
                            'Direction':     ['Up - Increases Risk' if v > 0 else 'Down - Decreases Risk' for v in sv],
                            '|Impact|':      [round(abs(float(v)), 4) for v in sv],
                        }).sort_values('|Impact|', ascending=False).reset_index(drop=True)

                        def highlight_direction(val):
                            if 'Up' in str(val):   return 'color: #c0392b; font-weight: 600'
                            if 'Down' in str(val): return 'color: #27ae60; font-weight: 600'
                            return ''

                        styled_shap = shap_df.style.map(highlight_direction, subset=['Direction'])
                        st.dataframe(styled_shap, use_container_width=True, hide_index=True)
                        csv_shap = shap_df.to_csv(index=False).encode('utf-8')
                        st.download_button("Download SHAP Table", csv_shap, "shap_analysis.csv", "text/csv")

            except Exception as e:
                st.error(f"SHAP calculation failed: `{e}`")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Update Patient Data", type="primary", use_container_width=True):
        st.session_state.page = 'form'
        st.rerun()


# ─────────────────────────────────────────────
# PAGE: PATIENT DATA LIST (MySQL)
# ─────────────────────────────────────────────
elif st.session_state.page == 'patient_list':
    st.markdown("## 🗃️ Patient Data List")
    st.markdown("Patient records loaded from the MySQL database.")

    # ── MySQL Connection (credentials pulled from Streamlit Secrets) ──────────
    # Set these in: Streamlit Cloud → App → Settings → Secrets
    # (or locally in .streamlit/secrets.toml)
    #
    #   db_host     = "your_host"
    #   db_port     = 3306
    #   db_name     = "pcad_db"
    #   db_user     = "root"
    #   db_password = "your_actual_password"
    #
    DB_CONFIG = {
        "host":     st.secrets["db_host"],
        "port":     st.secrets["db_port"],
        "database": st.secrets["db_name"],
        "user":     st.secrets["db_user"],
        "password": st.secrets["db_password"],
    }

    @st.cache_data(ttl=60)            # refresh cache every 60 s
    def fetch_patients():
        """
        Returns a DataFrame of all patient records.
        Adjust the query / column names to match your actual table schema.
        """
        import mysql.connector
        conn = mysql.connector.connect(**DB_CONFIG)
        query = """
            SELECT
    					p.id_pesakit AS `ID`,
    					p.nama_pesakit AS `Name`,
    					p.umur AS `Age`,
    					p.jantina AS `Gender`,
   					rk.bmi AS `BMI`,
    					rk.status_merokok AS `Smoking`,
    					rk.crp_level AS `CRP`,
    					rk.il6_level AS `IL-6`,
   				 	rk.vcam1_level AS `VCAM-1`,
    					rk.glutathione AS `Glutathione`,
    					la.keputusan_risiko AS `Risk Level`,
   					rk.tarikh_analisis AS `Date`
	FROM pesakit p
	INNER JOIN rekod_kesihatan rk
    				ON p.id_pesakit = rk.id_pesakit
	INNER JOIN laporan_analisis la
    				ON rk.id_rekod = la.id_rekod
	ORDER BY rk.tarikh_analisis DESC
        """
        df = pd.read_sql(query, conn)
        conn.close()
        return df

    # ── UI ────────────────────────────────────────────────────────────────────
    try:
        import mysql.connector
        driver_available = True
    except ImportError:
        driver_available = False

    col_status, col_refresh = st.columns([4, 1])

    with col_status:
        if driver_available:
            st.markdown('<span class="db-status-badge db-connected">● MySQL Driver Installed</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="db-status-badge db-disconnected">● mysql-connector-python not installed</span>', unsafe_allow_html=True)
            st.info("Run `pip install mysql-connector-python` and restart the app.")

    with col_refresh:
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    if driver_available:
        try:
            df_patients = fetch_patients()

            if df_patients.empty:
                st.info("No patient records found in the database.")
            else:
                # ── Search / Filter Bar ──────────────────────────────────────
                col_search, col_filter = st.columns([3, 1])
                with col_search:
                    search_term = st.text_input("🔍 Search by patient name", placeholder="Type a name…")
                with col_filter:
                    risk_filter = st.selectbox("Filter by Risk Level", ["All", "LOW RISK", "MEDIUM RISK", "HIGH RISK"])

                filtered = df_patients.copy()
                if search_term:
                    filtered = filtered[filtered["Name"].str.contains(search_term, case=False, na=False)]
                if risk_filter != "All":
                    filtered = filtered[filtered["Risk Level"] == risk_filter]

                st.markdown(f"**{len(filtered)} record(s) found**")

                # ── Colour-code Risk Level column ────────────────────────────
                def colour_risk(val):
                    colour_map = {
                        "HIGH RISK":   "background-color:#ffe0e0; color:#c0392b; font-weight:600",
                        "MEDIUM RISK": "background-color:#fff3e0; color:#e67e22; font-weight:600",
                        "LOW RISK":    "background-color:#e8f8f0; color:#27ae60; font-weight:600",
                    }
                    return colour_map.get(str(val).upper(), "")

                styled = filtered.style.map(colour_risk, subset=["Risk Level"])
                st.dataframe(styled, use_container_width=True, hide_index=True)

                # ── Download as CSV ──────────────────────────────────────────
                csv_data = filtered.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download as CSV",
                    data=csv_data,
                    file_name="pcad_patients.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.error(f"❌ Could not connect to MySQL database.\n\n**Error:** `{e}`")
            st.markdown("""
                **Troubleshooting checklist:**
                - Is MySQL running and reachable from this machine?
                - Are the credentials in Streamlit Secrets correct?
                - Does the `pcad_db` database and `patients` table exist?
                - Is port 3306 open / not blocked by a firewall?
            """)
    else:
        st.markdown("""
            <div class="info-card">
                <h4>🔌 MySQL Integration Setup</h4>
                <p>
                1. Install the driver: <code>pip install mysql-connector-python</code><br>
                2. Add your credentials to Streamlit Secrets (Settings → Secrets):<br>
                <code>db_host, db_port, db_name, db_user, db_password</code><br>
                3. Make sure your <code>patients</code> table has the expected columns (see the SQL query in the code).<br>
                4. Restart the Streamlit app.
                </p>
            </div>
        """, unsafe_allow_html=True)
