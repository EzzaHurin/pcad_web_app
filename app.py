import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import bcrypt

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# --- 1. CONFIG & ASSETS ---
st.set_page_config(page_title="PCAD Risk Assessment", layout="wide")

def local_css():
    st.markdown("""
    <style>
    .stApp { background-color: #FFF5F7; }
    [data-testid="stAppViewContainer"] { background-color: #FFF5F7; }
    [data-testid="stHeader"] { background-color: #FFF5F7; }

    .login-card {
        background: white; border-radius: 16px; padding: 36px 32px 28px 32px;
        text-align: center; box-shadow: 0 8px 24px rgba(0,0,0,0.06);
        max-width: 380px; margin: 40px auto 24px auto;
    }
    .login-heart-icon { font-size: 36px; margin-bottom: 8px; }
    .login-title { font-size: 19px; font-weight: 800; color: #00000; line-height: 1.3; margin: 8px 0 4px 0; }
    .login-subtitle { font-size: 14px; color: #888; margin-top: 6px; }
    .login-form-container {
        max-width: 460px; margin: 0 auto; background: rgba(255,255,255,0.5);
        padding: 24px 28px; border-radius: 12px;
    }
    div[data-testid="stForm"] {
        max-width: 460px; margin: 0 auto; background: rgba(255,255,255,0.5);
        border-radius: 12px; border: none; padding: 24px 28px;
    }
    div[data-testid="stForm"] button { background-color: #ff4b4b; color: white; border: none; font-weight: 600; }
    div[data-testid="stForm"] button:hover { background-color: #e63e3e; color: white; }

    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #ff7aa2 0%, #ff4d7d 45%, #ff2d6f 75%, #e91e63 100%);
    }
    [data-testid="stSidebar"] * { color: #e0e6f0 !important; }

    .sidebar-user-card {
        background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.12);
        border-radius: 12px; padding: 16px; margin-bottom: 8px; text-align: center;
    }
    .sidebar-avatar {
        width: 56px; height: 56px; border-radius: 50%;
        background: linear-gradient(135deg, #ff4b4b, #ff9a9e);
        display: flex; align-items: center; justify-content: center;
        font-size: 24px; margin: 0 auto 10px auto; box-shadow: 0 4px 12px rgba(255,75,75,0.4);
    }
    .sidebar-username { font-weight: 700; font-size: 15px; color: #ffffff !important; margin: 0; }
    .sidebar-role { font-size: 11px; color: #9aafc7 !important; margin: 2px 0 0 0; }
    .sidebar-divider { border: none; border-top: 1px solid rgba(255,255,255,0.1); margin: 12px 0; }
    .nav-label {
        font-size: 10px; font-weight: 700; letter-spacing: 1.5px;
        color: #ffffff !important; text-transform: uppercase; padding: 4px 0 8px 0;
    }

    .patient-banner {
        background: linear-gradient(135deg, #f0f4ff 0%, #e8f0fe 100%);
        padding: 20px 24px; border-radius: 12px; display: flex;
        justify-content: space-between; align-items: center;
        margin-bottom: 20px; border: 1px solid #d0dcf5;
    }
    .heartbeat-icon { font-size: 30px; color: #ff4b4b; }
    .banner-item label { font-size: 11px; color: #666; display: block; text-transform: uppercase; letter-spacing: 0.5px; }
    .banner-item span { font-weight: bold; font-size: 18px; color: #1a1f36; }

    .risk-card {
        background: white; padding: 30px; border-radius: 15px;
        text-align: center; box-shadow: 0 4px 20px rgba(0,0,0,0.08); border: 1px solid #f0f0f0;
    }
    .donut-outer {
        margin: 20px auto; width: 150px; height: 150px; border-radius: 50%;
        background: conic-gradient(var(--risk-color) var(--risk-degree), #eee 0deg);
        display: flex; align-items: center; justify-content: center;
    }
    .donut-inner {
        width: 120px; height: 120px; background: white; border-radius: 50%;
        display: flex; flex-direction: column; align-items: center; justify-content: center;
    }
    .risk-score-text { font-size: 32px; font-weight: bold; }
    .risk-label-box { padding: 10px; border-radius: 8px; font-weight: bold; margin-top: 10px; }

    .bio-card {
        background: #f9f9f9; padding: 15px; border-radius: 10px;
        border-left: 5px solid #ddd; margin-bottom: 10px;
    }
    .bio-title { font-size: 14px; color: #555; font-weight: 600; }
    .bio-value { font-size: 20px; font-weight: bold; }
    .bio-desc { font-size: 12px; color: #777; }
    .status-high { color: #ff4b4b; font-weight: bold; }
    .status-normal { color: #28a745; font-weight: bold; }
    .disclaimer-box {
        background-color: #fff3cd; padding: 15px; border-radius: 8px;
        font-size: 13px; margin-top: 20px; border-left: 4px solid #ffc107;
    }

    .dash-metric-card {
        background: white; border-radius: 12px; padding: 20px; text-align: center;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06); border: 1px solid #f0f0f0; transition: transform 0.2s;
    }
    .dash-metric-card:hover { transform: translateY(-2px); }
    .dash-metric-icon { font-size: 36px; margin-bottom: 8px; }
    .dash-metric-value { font-size: 28px; font-weight: 800; color: #1a1f36; }
    .dash-metric-label { font-size: 13px; color: #888; margin-top: 4px; }

    .info-card {
        background: white; border-radius: 12px; padding: 20px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06); border: 1px solid #f0f0f0; margin-bottom: 16px;
    }
    .info-card h4 { margin: 0 0 8px 0; color: #1a1f36; }
    .info-card p { margin: 0; font-size: 14px; color: #555; line-height: 1.6; }

    .db-status-badge { display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; }
    .db-connected { background: #d4edda; color: #155724; }
    .db-disconnected { background: #f8d7da; color: #721c24; }

    div[data-testid="stSidebarUserContent"] .stButton button {
        width: 100%; text-align: left; background: transparent; border: none;
        border-radius: 8px; padding: 10px 14px; font-size: 14px; font-weight: 500;
        color: #c8d6ea !important; transition: background 0.2s; cursor: pointer;
    }
    div[data-testid="stSidebarUserContent"] .stButton button:hover {
        background: rgba(255,255,255,0.1) !important; color: #ffffff !important;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_assets():
    import os
    app_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(app_dir, 'pcad_model.pkl')
    preproc_path = os.path.join(app_dir, 'preprocessor.pkl')
    try:
        model = joblib.load(model_path)
        preprocessor = joblib.load(preproc_path)
        return model, preprocessor, None
    except Exception as e:
        return None, None, str(e)

model, preprocessor, load_error = load_assets()

# --- 2. SESSION STATE MANAGEMENT ---
if 'page' not in st.session_state:
    st.session_state.page = 'main_menu'
if 'patient_data' not in st.session_state:
    st.session_state.patient_data = {}
if 'risk_result' not in st.session_state:
    st.session_state.risk_result = {}
if 'shap_input' not in st.session_state:
    st.session_state.shap_input = None
if 'shap_scaled' not in st.session_state:
    st.session_state.shap_scaled = None
if 'db_save_status' not in st.session_state:
    st.session_state.db_save_status = None
if 'delete_status' not in st.session_state:
    st.session_state.delete_status = None
if 'is_authenticated' not in st.session_state:
    st.session_state.is_authenticated = False
if 'logged_in_user' not in st.session_state:
    st.session_state.logged_in_user = None
if 'login_error' not in st.session_state:
    st.session_state.login_error = None


# ── Authentication Function ──────────────────────────────────────────────────
def authenticate_user(identifier, password):
    import mysql.connector
    conn = None
    try:
        conn = mysql.connector.connect(
            host=st.secrets["db_host"],
            port=st.secrets["db_port"],
            database=st.secrets["db_name"],
            user=st.secrets["db_user"],
            password=st.secrets["db_password"],
            connection_timeout=10,
        )
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT id_staf, nama_staf, emel, kata_laluan FROM Pengguna "
            "WHERE nama_staf = %s OR emel = %s LIMIT 1",
            (identifier, identifier)
        )
        row = cursor.fetchone()
        cursor.close()

        if row is None:
            return False, "Invalid username or password."

        stored_hash = row["kata_laluan"]
        try:
            password_matches = bcrypt.checkpw(password.encode("utf-8"), stored_hash.encode("utf-8"))
        except (ValueError, AttributeError):
            return False, "Account password is not set up correctly. Please contact admin."

        if not password_matches:
            return False, "Invalid username or password."

        return True, {
            "id_staf": row["id_staf"],
            "name": row["nama_staf"],
            "role": "Staff",
        }

    except Exception as e:
        return False, f"Login failed: {e}"
    finally:
        if conn and conn.is_connected():
            conn.close()


local_css()

# ─────────────────────────────────────────────
# LOGIN PAGE
# ─────────────────────────────────────────────
if not st.session_state.is_authenticated:
    st.markdown("""
        <div class="login-card">
            <div class="login-heart-icon">❤️</div>
            <div class="login-title">PCAD Risk Detection<br>and Classification<br>System</div>
            <div class="login-subtitle">Log In</div>
        </div>
    """, unsafe_allow_html=True)

    if st.session_state.login_error:
        st.error(st.session_state.login_error)
        st.session_state.login_error = None

    with st.form("login_form"):
        username = st.text_input("Username", placeholder="Enter staff name or email")
        password = st.text_input("Password", type="password", placeholder="Enter password")
        login_submit = st.form_submit_button("Log In", use_container_width=True)

    if login_submit:
        if not username or not password:
            st.session_state.login_error = "Please enter both username and password."
            st.rerun()
        else:
            success, result = authenticate_user(username.strip(), password)
            if success:
                st.session_state.is_authenticated = True
                st.session_state.logged_in_user = result
                st.session_state.page = 'main_menu'
                st.rerun()
            else:
                st.session_state.login_error = result
                st.rerun()

    st.stop()


# --- 3. SIDEBAR (only shown when authenticated) ---
with st.sidebar:
    user = st.session_state.logged_in_user
    initials = "".join([w[0].upper() for w in user['name'].split()[:2]])

    st.markdown(f"""
        <div class="sidebar-user-card">
            <div class="sidebar-avatar">{initials}</div>
            <p class="sidebar-username">{user['name']}</p>
            <p class="sidebar-role">{user['role']}</p>
        </div>
    """, unsafe_allow_html=True)

    if st.button("🔓  Log Out", key="logout_btn", use_container_width=True):
        st.session_state.is_authenticated = False
        st.session_state.logged_in_user = None
        st.session_state.page = 'main_menu'
        st.session_state.patient_data = {}
        st.session_state.risk_result = {}
        st.rerun()

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.markdown('<div class="nav-label">Navigation</div>', unsafe_allow_html=True)

    nav_items = [
        ("🏠  Homepage",        "main_menu"),
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
    st.markdown('<div style="font-size:11px; color:#fffff; text-align:center;">by Ezzahurin Ajemal</div>', unsafe_allow_html=True)
# ── END SIDEBAR ──


# ─────────────────────────────────────────────
# PAGE: HOMEPAGE
# ─────────────────────────────────────────────
if st.session_state.page == 'main_menu':

    # ══════════════════════════════════════════
    # SECTION 1: ABOUT THIS SYSTEM
    # ══════════════════════════════════════════
    st.markdown("## 📌 About This System")
    st.markdown(
        f"Welcome back, **{st.session_state.logged_in_user['name']}**. "
        "Here's an overview of the PCAD Risk Prediction and Classification System."
    )
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
        <div class="info-card">
            <h4>🎯 Purpose</h4>
            <p>
            This system predicts the risk level of Premature Coronary Artery Disease (PCAD)
            using patient biomarkers and clinical data. It leverages a trained machine learning
            model to classify patients into <b>Low</b>, <b>Medium</b>, or <b>High</b> risk categories.
            </p>
        </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
            <div class="info-card">
                <h4>🧪 Key Biomarkers Assessed</h4>
                <p>
                <b>CRP</b> — C-Reactive Protein (inflammation marker)<br>
                <b>IL-6</b> — Interleukin-6 (cytokine inflammatory marker)<br>
                <b>VCAM-1</b> — Vascular Cell Adhesion Molecule<br>
                <b>Glutathione</b> — Antioxidant stress indicator
                </p>
            </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
            <div class="info-card">
                <h4>📊 Risk Classification</h4>
                <p>
                    <span style="color:#28a745; font-weight:700;">● LOW RISK</span><br>
                    Biomarkers within healthy range.<br>
                    <span style="color:#ffa500; font-weight:700;">● MEDIUM RISK</span><br>
                    Some biomarkers are elevated and may require monitoring.<br>
                    <span style="color:#ff4b4b; font-weight:700;">● HIGH RISK</span><br>
                    Multiple biomarkers are elevated, indicating a higher likelihood of PCAD.
                </p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    # ══════════════════════════════════════════
    # SECTION 2: DATA EXPLORATION
    # ══════════════════════════════════════════
    st.markdown("## 📊 Data Exploration")
    st.markdown("Biomarker distributions from the patient dataset.")
    st.markdown("<br>", unsafe_allow_html=True)

    import os
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
    DEFAULT_DATA_PATH = os.path.join(APP_DIR, "data", "patients.csv")

    @st.cache_data(ttl=300)
    def load_dashboard_data(path):
        return pd.read_csv(path)

    dashboard_df = None
    if os.path.exists(DEFAULT_DATA_PATH):
        dashboard_df = load_dashboard_data(DEFAULT_DATA_PATH)
        st.caption(f"📁 Data loaded from `data/patients.csv` — {len(dashboard_df)} records")
    else:
        st.warning("⚠️ No patient data file found. Please ensure `patients.csv` is placed in the `data/` folder in the same directory as `app.py`.")

    if dashboard_df is not None and not dashboard_df.empty:
        df = dashboard_df.copy()

        if "PCAD_Status" in df.columns:
            df["PCAD_Label"] = df["PCAD_Status"].map({1: "Positive", 0: "Negative"})

        biomarker_cols = [c for c in ["CRP", "IL_6", "VCAM_1", "Glutathione"] if c in df.columns]

        if biomarker_cols:
            st.markdown("**Distribution of Key Biomarkers**")

            row1_cols = st.columns(2)
            biomarker_colors = {
                "CRP":         "#e74c3c",
                "IL_6":        "#3498db",
                "VCAM_1":      "#2ecc71",
                "Glutathione": "#f39c12",
            }
            biomarker_labels = {
                "CRP":         "CRP (mg/L)",
                "IL_6":        "IL-6 (pg/mL)",
                "VCAM_1":      "VCAM-1 (ng/mL)",
                "Glutathione": "Glutathione (μmol/L)",
            }

            for idx, biomarker in enumerate(biomarker_cols):
                col = row1_cols[idx % 2]
                with col:
                    st.markdown(f"**{biomarker_labels.get(biomarker, biomarker)} Distribution**")
                    fig, ax = plt.subplots(figsize=(5, 3.5))

                    if "PCAD_Label" in df.columns:
                        for label, color in [("Negative", "#28a745"), ("Positive", "#ff4b4b")]:
                            subset = df[df["PCAD_Label"] == label][biomarker].dropna()
                            ax.hist(subset, bins=15, alpha=0.6, label=label, color=color, edgecolor='white')
                        ax.legend(title="PCAD Status", fontsize=8)
                    else:
                        ax.hist(df[biomarker].dropna(), bins=15,
                                color=biomarker_colors.get(biomarker, "#4b8bff"), edgecolor='white')

                    ax.set_xlabel(biomarker_labels.get(biomarker, biomarker), fontsize=9)
                    ax.set_ylabel("Number of Patients", fontsize=9)
                    ax.set_title(f"{biomarker_labels.get(biomarker, biomarker)} Distribution",
                                 fontsize=10, fontweight='bold')
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                    stats = df[biomarker].describe()
                    st.caption(
                        f"Mean: {stats['mean']:.2f} | Std: {stats['std']:.2f} | "
                        f"Min: {stats['min']:.2f} | Max: {stats['max']:.2f}"
                    )

                if idx % 2 == 1 and idx < len(biomarker_cols) - 1:
                    st.markdown("<br>", unsafe_allow_html=True)
                    row1_cols = st.columns(2)

        else:
            st.warning("⚠️ Biomarker columns (CRP, IL_6, VCAM_1, Glutathione) not found in the dataset.")

        with st.expander("📋 View Raw Data Table"):
            st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    # ══════════════════════════════════════════
    # SECTION 3: MODEL PERFORMANCE
    # ══════════════════════════════════════════
    st.markdown("## ⚙️ Model Performance")
    st.markdown("Evaluation metrics of the trained PCAD risk classification model for High Risk.")
    st.markdown("<br>", unsafe_allow_html=True)

    model_metrics = {
        "Accuracy":  0.95,
        "Precision": 0.96,
        "Recall":    0.95,
        "F1 Score":  0.95,
    }

    metric_colors = ["#28a745", "#3498db", "#e74c3c", "#f39c12"]
    metric_icons  = ["✅", "🎯", "🔍", "⚖️"]
    metric_names  = list(model_metrics.keys())
    metric_values = list(model_metrics.values())

    perf_cols = st.columns(4)
    for col, name, value, color, icon in zip(perf_cols, metric_names, metric_values,
                                              metric_colors, metric_icons):
        with col:
            st.markdown(f"""
                <div class="dash-metric-card" style="border-top: 4px solid {color};">
                    <div class="dash-metric-icon">{icon}</div>
                    <div class="dash-metric-value" style="color:{color};">{value*100:.1f}%</div>
                    <div class="dash-metric-label">{name}</div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
        <div class="info-card">
            <h4>📖 Metric Definitions</h4>
            <p>
            <b>Accuracy</b> — Overall proportion of correct predictions (TP + TN) / Total.<br>
            <b>Precision</b> — Of all patients predicted PCAD Positive, how many truly are? Minimises false alarms.<br>
            <b>Recall (Sensitivity)</b> — Of all true PCAD Positive patients, how many did the model catch? Minimises missed cases.<br>
            <b>F1 Score</b> — Harmonic mean of Precision and Recall; balanced metric for imbalanced classes.
            </p>
        </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE: PREDICT PCAD (Patient Data Entry)
# ─────────────────────────────────────────────
elif st.session_state.page == 'form':
    
    st.markdown("""
        <style>
        div[data-testid="stForm"] {
            max-width: 100% !important;
            width: 100% !important;
            padding: 0 !important;
            background: transparent !important;
            border-radius: 0 !important;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("## 🫀 Predict PCAD — Patient Data Entry")
    st.write("Enter the patient's clinical data below to generate a risk classification.")

    DB_CONFIG = {
        "host":               st.secrets["db_host"],
        "port":               st.secrets["db_port"],
        "database":           st.secrets["db_name"],
        "user":               st.secrets["db_user"],
        "password":           st.secrets["db_password"],
        "connection_timeout": 10,
        "autocommit":         False,
    }

    def get_or_create_default_staff(cursor):
        cursor.execute("SELECT id_staf FROM Pengguna ORDER BY id_staf LIMIT 1")
        row = cursor.fetchone()
        if row:
            return row[0]
        cursor.execute(
            "INSERT INTO Pengguna (nama_staf, emel, kata_laluan) VALUES (%s, %s, %s)",
            ("Default Staff", "default.staff@ukm.edu.my", "changeme")
        )
        return cursor.lastrowid

    def get_biomarker_status(value, threshold, is_reverse=False):
        is_bad = (value < threshold) if is_reverse else (value > threshold)
        if is_reverse:
            return "Low" if is_bad else "Normal"
        return "High" if is_bad else "Normal"

    def save_prediction_to_db(name, age, gender, bmi, smoking, lipid, renal, liver,
                               crp, vcam1, il6, glutathione, id_kluster, keputusan_risiko,
                               id_staf=None):
        import mysql.connector
        conn = None
        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            cursor = conn.cursor()

            if id_staf is None:
                id_staf = get_or_create_default_staff(cursor)

            jantina = "Lelaki" if gender == 1 else "Perempuan"
            cursor.execute(
                "INSERT INTO Pesakit (nama_pesakit, umur, jantina) VALUES (%s, %s, %s)",
                (name, age, jantina)
            )
            id_pesakit = cursor.lastrowid

            cursor.execute(
                """INSERT INTO Rekod_Kesihatan
                   (id_staf, id_pesakit, umur, bmi, status_merokok, profil_lipid,
                    profil_renal, profil_hati, crp_level, vcam1_level, il6_level, glutathione)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (id_staf, id_pesakit, age, bmi, smoking, lipid, renal, liver,
                 crp, vcam1, il6, glutathione)
            )
            id_rekod = cursor.lastrowid

            status_crp         = get_biomarker_status(crp, 3.0)
            status_vcam1       = get_biomarker_status(vcam1, 500)
            status_il6         = get_biomarker_status(il6, 5.0)
            status_glutathione = get_biomarker_status(glutathione, 4.0, is_reverse=True)

            cursor.execute(
                """INSERT INTO Laporan_Analisis
                   (id_rekod, id_kluster, keputusan_risiko,
                    skor_crp, skor_vcam1, skor_il6, skor_glutathione)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (id_rekod, id_kluster, keputusan_risiko,
                 status_crp, status_vcam1, status_il6, status_glutathione)
            )

            conn.commit()

            cursor.execute("SELECT COUNT(*) FROM Pesakit WHERE id_pesakit = %s", (id_pesakit,))
            verify_count = cursor.fetchone()[0]
            cursor.close()

            return True, (f"Patient ID: {id_pesakit}, Record ID: {id_rekod} | "
                          f"Verified rows in Pesakit: {verify_count}")

        except Exception as e:
            if conn:
                conn.rollback()
            return False, str(e)
        finally:
            if conn and conn.is_connected():
                conn.close()

    with st.form("prediction_form"):
        name = st.text_input("Patient Full Name")

        col1, col2 = st.columns(2)
        with col1:
            age       = st.number_input("Age", min_value=1, max_value=120)
            bmi       = st.number_input("BMI", min_value=10.0, max_value=50.0)
            gender    = st.selectbox("Gender", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
            smoking   = st.selectbox("Smoking Status", options=[1, 0], format_func=lambda x: "Smoker" if x == 1 else "Non-Smoker")
            crp       = st.number_input("CRP Level (mg/L)")
            il6       = st.number_input("IL-6 Level (pg/mL)")
        with col2:
            vcam        = st.number_input("VCAM-1 Level (ng/mL)")
            glutathione = st.number_input("Glutathione (mmol/L)")
            lipid       = st.number_input("Lipid Profile")
            renal       = st.number_input("Renal Profile")
            liver       = st.number_input("Liver Profile")

        submit = st.form_submit_button("Predict Risk Level", use_container_width=True, type="primary")

    if submit:
        input_data = pd.DataFrame(
            [[age, bmi, gender, smoking, crp, il6, vcam, glutathione, lipid, renal, liver]],
            columns=['Age', 'BMI', 'Gender', 'Smoking_Status', 'CRP', 'IL_6',
                     'VCAM_1', 'Glutathione', 'Lipid_Profile', 'Renal_Profile', 'Liver_Profile']
        )

        if model is not None and preprocessor is not None:
            data_scaled = preprocessor.transform(input_data)
            prediction  = model.predict(data_scaled)[0]
            st.session_state.shap_input  = input_data
            st.session_state.shap_scaled = data_scaled
        else:
            st.warning(f"⚠️ Model files not found. Showing demo result. Error: `{load_error}`")
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

        id_kluster_map   = {'Low': 0, 'Medium': 1, 'High': 2, 0: 0, 1: 1, 2: 2}
        risiko_label_map = {'Low': 'Low', 'Medium': 'Medium', 'High': 'High', 0: 'Low', 1: 'Medium', 2: 'High'}
        id_kluster       = id_kluster_map.get(prediction, 0)
        keputusan_risiko = risiko_label_map.get(prediction, 'Low')

        try:
            success, msg = save_prediction_to_db(
                name=name, age=age, gender=gender, bmi=bmi, smoking=smoking,
                lipid=lipid, renal=renal, liver=liver,
                crp=crp, vcam1=vcam, il6=il6, glutathione=glutathione,
                id_kluster=id_kluster, keputusan_risiko=keputusan_risiko
            )
            st.session_state.db_save_status = ("success", msg) if success else ("error", msg)
        except Exception as e:
            st.session_state.db_save_status = ("error", str(e))

        st.session_state.page = 'result'
        st.rerun()


# ─────────────────────────────────────────────
# PAGE: RESULT DASHBOARD
# ─────────────────────────────────────────────
elif st.session_state.page == 'result':
    data = st.session_state.patient_data
    res  = st.session_state.risk_result

    st.markdown("## ❤️ CAD Risk Assessment Results")

    if st.session_state.db_save_status is not None:
        status, msg = st.session_state.db_save_status
        if status == "success":
            st.success(f"✅ Patient record saved to database. {msg}")
        else:
            st.error(f"❌ Failed to save to database: {msg}")
        st.session_state.db_save_status = None

    st.markdown("<br>", unsafe_allow_html=True)

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
        label_parts = res['label'].split(' ')
        label_line1 = label_parts[0] if len(label_parts) > 0 else res['label']
        label_line2 = ' '.join(label_parts[1:]) if len(label_parts) > 1 else ''

        st.markdown(f"""
            <div class="risk-card">
                <div style="font-weight:600; color:#555;">PREMATURE CAD RISK</div>
                <div class="donut-outer" style="--risk-color: {res['color']}; --risk-degree: {res['degree']};">
                    <div class="donut-inner">
                        <div class="risk-score-text" style="color: {res['color']}; font-size: 22px; line-height: 1.2;">{label_line1}</div>
                        <div style="color: {res['color']}; font-size: 16px; font-weight: bold;">{label_line2}</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    with right_col:
        st.markdown('<h6>🕒 BIOMARKER ANALYSIS</h6>', unsafe_allow_html=True)

        def get_bio_html(label, value, unit, threshold, is_reverse=False):
            is_bad = (value < threshold) if is_reverse else (value > threshold)
            if is_reverse:
                status_text = "Low" if is_bad else "Normal"
            else:
                status_text = "High" if is_bad else "Normal"
            status_class = "status-high" if is_bad else "status-normal"

            desc = "Levels are within range."
            if label == "CRP"           and is_bad: desc = "Indicates systemic inflammation."
            elif label == "VCAM-1"      and is_bad: desc = "Endothelial dysfunction detected."
            elif label == "Glutathione" and is_bad: desc = "Depleted antioxidants increase oxidative stress."
            elif not is_bad:                        desc = "Within healthy clinical range."

            return f"""
                <div class="bio-card">
                    <div class="bio-title">{label}</div>
                    <div class="bio-value">{value} <span style="font-size:12px; color:#888;">{unit}</span></div>
                    <div class="bio-desc">{desc}</div>
                    <div class="{status_class}">{status_text}</div>
                </div>
            """

        r1a, r1b = st.columns(2)
        with r1a: st.markdown(get_bio_html("CRP",         data['crp'],         "mg/L",   3.0),       unsafe_allow_html=True)
        with r1b: st.markdown(get_bio_html("IL-6",        data['il6'],         "pg/mL",  5.0),       unsafe_allow_html=True)
        r2a, r2b = st.columns(2)
        with r2a: st.markdown(get_bio_html("VCAM-1",      data['vcam'],        "ng/mL",  500),       unsafe_allow_html=True)
        with r2b: st.markdown(get_bio_html("Glutathione", data['glutathione'], "mmol/L", 4.0, True), unsafe_allow_html=True)

        st.caption(
            "ℹ️ Note: Glutathione works in reverse — a **Low** result (below the healthy threshold) "
            "is the concerning sign, since glutathione is an antioxidant that *protects* against "
            "cardiovascular stress. Lower levels mean weaker antioxidant defence."
        )

    st.markdown("""
        <div class="disclaimer-box">
            <b>⚕️ Medical Disclaimer:</b> This assessment is for informational purposes only.
            It should not replace professional medical diagnosis. Please consult with a cardiologist.
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    
    # ── SHAP ANALYSIS SECTION ─────────────────────────────────────────────────
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

                    tab1, tab2, tab3 = st.tabs(["Waterfall Chart", "Bar Chart", "Feature Table"])

                    # TAB 1: Waterfall
                    with tab1:
                        st.markdown("**Waterfall chart** — each bar shows how much a feature increased (red) or decreased (blue) the risk score.")

                        sorted_idx  = np.argsort(np.abs(sv))[::-1]
                        top_n       = min(11, len(sorted_idx))
                        idx_top     = sorted_idx[:top_n][::-1]
                        feat_labels = [f"{feature_names[i]}  =  {raw_vals[i]:.2f}" for i in idx_top]
                        values      = sv[idx_top]
                        colors      = ['#ff4b4b' if v > 0 else '#4b8bff' for v in values]

                        # centre chart using columns: padding | chart | padding
                        _, chart_col, _ = st.columns([0.5, 9, 0.5])
                        with chart_col:
                            fig_wf, ax_wf = plt.subplots(figsize=(7, 3.5))
                            fig_wf.patch.set_facecolor('#fafafa')
                            ax_wf.set_facecolor('#fafafa')

                            bars = ax_wf.barh(range(len(values)), values,
                                              color=colors, edgecolor='white', height=0.55)
                            ax_wf.set_yticks(range(len(values)))
                            ax_wf.set_yticklabels(feat_labels, fontsize=8)
                            ax_wf.axvline(0, color='#333333', linewidth=0.8)
                            ax_wf.set_xlabel("SHAP Value (impact on model output)", fontsize=9)
                            ax_wf.set_title(f"Feature Contributions — {pred_label}",
                                            fontsize=10, fontweight='bold', pad=10)
                            ax_wf.tick_params(axis='x', labelsize=8)
                            ax_wf.spines[['top', 'right']].set_visible(False)

                            for bar, val in zip(bars, values):
                                offset = 0.002 if val >= 0 else -0.002
                                ha     = 'left' if val >= 0 else 'right'
                                ax_wf.text(val + offset, bar.get_y() + bar.get_height() / 2,
                                           f"{val:+.3f}", va='center', ha=ha, fontsize=7)

                            red_patch  = mpatches.Patch(color='#ff4b4b', label='Increases risk')
                            blue_patch = mpatches.Patch(color='#4b8bff', label='Decreases risk')
                            ax_wf.legend(handles=[red_patch, blue_patch],
                                         loc='lower right', fontsize=8, framealpha=0.7)
                            fig_wf.tight_layout(pad=1.5)
                            st.pyplot(fig_wf, use_container_width=True)
                            plt.close(fig_wf)

                    # TAB 2: Bar chart
                    with tab2:
                        st.markdown("**Bar chart** — absolute SHAP value per feature; longer bar = more influential.")

                        abs_sv     = np.abs(sv)
                        sorted_abs = np.argsort(abs_sv)
                        feat_bar   = [feature_names[i] for i in sorted_abs]
                        vals_bar   = abs_sv[sorted_abs]

                        _, chart_col2, _ = st.columns([0.5, 9, 0.5])
                        with chart_col2:
                            fig_bar, ax_bar = plt.subplots(figsize=(7, 3.5))
                            fig_bar.patch.set_facecolor('#fafafa')
                            ax_bar.set_facecolor('#fafafa')

                            bar_colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(vals_bar)))
                            ax_bar.barh(range(len(vals_bar)), vals_bar,
                                        color=bar_colors, edgecolor='white', height=0.55)
                            ax_bar.set_yticks(range(len(vals_bar)))
                            ax_bar.set_yticklabels(feat_bar, fontsize=9)
                            ax_bar.set_xlabel("Mean |SHAP Value|", fontsize=9)
                            ax_bar.set_title("Feature Importance (SHAP)",
                                             fontsize=10, fontweight='bold', pad=10)
                            ax_bar.tick_params(axis='x', labelsize=8)
                            ax_bar.spines[['top', 'right']].set_visible(False)

                            for i, v in enumerate(vals_bar):
                                ax_bar.text(v + 0.001, i, f"{v:.3f}",
                                            va='center', fontsize=7.5)

                            fig_bar.tight_layout(pad=1.5)
                            st.pyplot(fig_bar, use_container_width=True)
                            plt.close(fig_bar)

                    # TAB 3: Table
                    with tab3:
                        st.markdown("**Detailed breakdown** of SHAP values per feature.")

                        shap_df = pd.DataFrame({
                            'Feature':       feature_names,
                            'Patient Value': [f"{v:.2f}" for v in raw_vals],
                            'SHAP Value':    [round(float(v), 4) for v in sv],
                            'Direction':     ['⬆ Increases Risk' if v > 0 else ('⬇ Decreases Risk' if v < 0 else '— No Influence') for v in sv],
                            '|Impact|':      [round(abs(float(v)), 4) for v in sv],
                        }).sort_values('|Impact|', ascending=False).reset_index(drop=True)

                        def highlight_direction(val):
                            if '⬆' in str(val): return 'color: #c0392b; font-weight: 600'
                            if '⬇' in str(val): return 'color: #27ae60; font-weight: 600'
                            if '—' in str(val):  return 'color: #888888; font-weight: 600'
                            return ''

                        # centre table with columns
                        _, tbl_col, _ = st.columns([0.5, 9, 0.5])
                        with tbl_col:
                            styled_shap = shap_df.style.map(highlight_direction, subset=['Direction'])
                            st.dataframe(styled_shap, use_container_width=True, hide_index=True, height=420)

                            csv_shap = shap_df.to_csv(index=False).encode('utf-8')
                            st.download_button("⬇️ Download SHAP Table", csv_shap,
                                               "shap_analysis.csv", "text/csv")

            except Exception as e:
                st.error(f"SHAP calculation failed: `{e}`")


# ─────────────────────────────────────────────
# PAGE: PATIENT DATA LIST (MySQL)
# ─────────────────────────────────────────────
elif st.session_state.page == 'patient_list':
    st.markdown("## 🗃️ Patient Data List")
    st.markdown("Patient records loaded from the MySQL database.")

    DB_CONFIG = {
        "host":               st.secrets["db_host"],
        "port":               st.secrets["db_port"],
        "database":           st.secrets["db_name"],
        "user":               st.secrets["db_user"],
        "password":           st.secrets["db_password"],
        "connection_timeout": 10,
        "connect_timeout":    10,
        "autocommit":         True,
        "use_pure":           True,
    }

    @st.cache_data(ttl=60)
    def fetch_patients():
        import mysql.connector
        conn = mysql.connector.connect(**DB_CONFIG)
        query = """
            SELECT
                p.id_pesakit        AS `ID`,
                p.nama_pesakit      AS `Name`,
                p.umur              AS `Age`,
                p.jantina           AS `Gender`,
                rk.id_rekod         AS `Record_ID`,
                rk.bmi              AS `BMI`,
                rk.status_merokok   AS `Smoking`,
                rk.crp_level        AS `CRP`,
                rk.il6_level        AS `IL-6`,
                rk.vcam1_level      AS `VCAM-1`,
                rk.glutathione      AS `Glutathione`,
                la.keputusan_risiko AS `Risk Level`,
                la.skor_crp         AS `CRP Status`,
                la.skor_vcam1       AS `VCAM-1 Status`,
                la.skor_il6         AS `IL-6 Status`,
                la.skor_glutathione AS `Glutathione Status`,
                rk.tarikh_rekod     AS `Date`
            FROM Pesakit p
            JOIN Rekod_Kesihatan rk  ON rk.id_pesakit = p.id_pesakit
            JOIN Laporan_Analisis la ON la.id_rekod   = rk.id_rekod
            ORDER BY rk.tarikh_rekod DESC
        """
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            df = pd.read_sql(query, conn)
        conn.close()
        return df

    def delete_patient_record(id_pesakit, id_rekod):
        import mysql.connector
        conn = None
        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            cursor = conn.cursor()

            cursor.execute("DELETE FROM Laporan_Analisis WHERE id_rekod = %s", (id_rekod,))
            cursor.execute("DELETE FROM Rekod_Kesihatan WHERE id_rekod = %s", (id_rekod,))

            cursor.execute("SELECT COUNT(*) FROM Rekod_Kesihatan WHERE id_pesakit = %s", (id_pesakit,))
            remaining = cursor.fetchone()[0]
            if remaining == 0:
                cursor.execute("DELETE FROM Pesakit WHERE id_pesakit = %s", (id_pesakit,))

            conn.commit()
            cursor.close()
            return True, "Record deleted successfully."
        except Exception as e:
            if conn:
                conn.rollback()
            return False, str(e)
        finally:
            if conn and conn.is_connected():
                conn.close()

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

    if "delete_status" in st.session_state and st.session_state.delete_status is not None:
        status, msg = st.session_state.delete_status
        if status == "success":
            st.success(f"✅ {msg}")
        else:
            st.error(f"❌ Failed to delete: {msg}")
        st.session_state.delete_status = None

    st.markdown("<br>", unsafe_allow_html=True)

    if driver_available:
        try:
            df_patients = fetch_patients()

            if df_patients.empty:
                st.info("No patient records found in the database.")
            else:
                col_search, col_filter = st.columns([3, 1])
                with col_search:
                    search_term = st.text_input(
                        "🔍 Search by patient name",
                        placeholder="Type any part of a name…",
                        help="Matches if the name contains this text anywhere (case-insensitive)."
                    )
                with col_filter:
                    risk_filter = st.selectbox("Filter by Risk Level", ["All", "Low", "Medium", "High"])

                filtered = df_patients.copy()

                if search_term and search_term.strip():
                    term = search_term.strip()
                    filtered = filtered[filtered["Name"].str.contains(term, case=False, na=False, regex=False)]

                if risk_filter != "All":
                    filtered = filtered[filtered["Risk Level"].str.contains(risk_filter, case=False, na=False, regex=False)]

                st.markdown(f"**{len(filtered)} record(s) found**")

                float_cols = ["BMI", "CRP", "IL-6", "VCAM-1", "Glutathione"]
                display_df = filtered.copy()
                for col in float_cols:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].astype(float).round(2)

                def colour_risk(val):
                    val_upper = str(val).upper()
                    if "HIGH"   in val_upper: return "background-color:#ffe0e0; color:#c0392b; font-weight:600"
                    if "MEDIUM" in val_upper: return "background-color:#fff3e0; color:#e67e22; font-weight:600"
                    if "LOW"    in val_upper: return "background-color:#e8f8f0; color:#27ae60; font-weight:600"
                    return ""

                def colour_status(val):
                    val_upper = str(val).upper()
                    if val_upper in ("HIGH", "LOW"): return "color:#c0392b; font-weight:600"
                    if val_upper == "NORMAL":        return "color:#27ae60; font-weight:600"
                    return ""

                status_cols = [c for c in ["CRP Status", "VCAM-1 Status", "IL-6 Status", "Glutathione Status"]
                               if c in display_df.columns]

                styled = display_df.style.map(colour_risk, subset=["Risk Level"]) \
                                         .format({c: "{:.2f}" for c in float_cols if c in display_df.columns})
                if status_cols:
                    styled = styled.map(colour_status, subset=status_cols)

                st.dataframe(styled, use_container_width=True, hide_index=True)

                csv_data = display_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download as CSV",
                    data=csv_data,
                    file_name="pcad_patients.csv",
                    mime="text/csv",
                )

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("#### 🗑️ Delete a Record")
                st.caption("Select a patient record below to permanently remove it from the database.")

                filtered = filtered.reset_index(drop=True)
                options = [
                    f"{row['ID']} — {row['Name']} ({pd.to_datetime(row['Date']).strftime('%Y-%m-%d')})"
                    for _, row in filtered.iterrows()
                ]

                if options:
                    del_col1, del_col2 = st.columns([4, 1])
                    with del_col1:
                        selected_option = st.selectbox("Select record to delete", options, key="delete_select")
                    with del_col2:
                        st.markdown("<br>", unsafe_allow_html=True)
                        confirm_delete = st.button("🗑️ Delete", type="primary", use_container_width=True)

                    if confirm_delete:
                        selected_idx = options.index(selected_option)
                        selected_row = filtered.iloc[selected_idx]
                        success, msg = delete_patient_record(
                            id_pesakit=int(selected_row["ID"]),
                            id_rekod=int(selected_row["Record_ID"])
                        )
                        st.session_state.delete_status = ("success" if success else "error", msg)
                        st.cache_data.clear()
                        st.rerun()

        except Exception as e:
            st.error(f"❌ Could not connect to MySQL database.\n\n**Error:** `{e}`")
            st.markdown("""
                **Troubleshooting checklist:**
                - Is MySQL running and reachable from this machine?
                - Are the credentials in Streamlit Secrets correct?
                - Does the database and `Pesakit`/`Rekod_Kesihatan`/`Laporan_Analisis` tables exist?
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
                3. Make sure your <code>Pesakit</code>, <code>Rekod_Kesihatan</code>, and <code>Laporan_Analisis</code> tables exist.<br>
                4. Restart the Streamlit app.
                </p>
            </div>
        """, unsafe_allow_html=True)
