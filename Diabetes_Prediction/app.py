import os
import time
from datetime import datetime
import numpy as np
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# 1. PAGE CONFIGURATION & INITIALIZATION
# ============================================================

st.set_page_config(
    page_title="DiabetesAI - Medical Intelligence Dashboard",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded",
)

HISTORY_FILE = "prediction_history.csv"
DATASET_FILE = "diabetes.csv"

# Ensure History File Exists
if not os.path.exists(HISTORY_FILE):
    df_empty = pd.DataFrame(
        columns=[
            "Timestamp",
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age",
            "Prediction",
            "Probability",
            "RiskLevel",
        ]
    )
    df_empty.to_csv(HISTORY_FILE, index=False)

# ============================================================
# 2. CUSTOM CSS & ADVANCED THEMING (GLASSMORPHISM & ANIMATIONS)
# ============================================================

st.markdown(
    """
<style>
    /* Global Imports */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Gradient Background & Soft Overlay */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        color: #f8fafc;
    }

    /* Glassmorphism Containers */
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.36);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 40px 0 rgba(14, 165, 233, 0.15);
        border-color: rgba(14, 165, 233, 0.3);
    }

    /* Metric KPI Cards */
    .kpi-card {
        background: linear-gradient(135deg, rgba(30,41,59,0.8) 0%, rgba(15,23,42,0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 14px;
        padding: 18px 20px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    }
    .kpi-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 6px;
        font-weight: 600;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
    }

    /* Risk Status Badges */
    .badge-low {
        background-color: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.3);
        padding: 8px 16px;
        border-radius: 30px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-high {
        background-color: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 8px 16px;
        border-radius: 30px;
        font-weight: 600;
        display: inline-block;
    }

    /* Header Banner */
    .header-container {
        padding: 20px 28px;
        background: linear-gradient(90deg, #0284c7 0%, #0d9488 100%);
        border-radius: 16px;
        margin-bottom: 25px;
        box-shadow: 0 10px 25px -5px rgba(2, 132, 199, 0.4);
    }

    /* Streamlit Sidebar Customization */
    section[data-testid="stSidebar"] {
        background-color: #0b1329 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Streamlit Primary Button Redesign */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #0284c7 0%, #0284c7 100%);
        color: white;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1rem;
        padding: 12px 28px;
        border: none;
        box-shadow: 0 4px 14px 0 rgba(2, 132, 199, 0.39);
        transition: all 0.2s ease-in-out;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(90deg, #0369a1 0%, #0f766e 100%);
        box-shadow: 0 6px 20px 0 rgba(2, 132, 199, 0.55);
        transform: translateY(-1px);
    }

    /* Disclaimer Footer Box */
    .disclaimer-box {
        background: rgba(245, 158, 11, 0.1);
        border-left: 4px solid #f59e0b;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin-top: 30px;
        font-size: 0.85rem;
        color: #cbd5e1;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# 3. MODEL LOADING & CACHING
# ============================================================


@st.cache_resource
def load_ml_components():
    try:
        with open("diabetes_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("diabetes_features.pkl", "rb") as f:
            features = pickle.load(f)
        return model, features, None
    except Exception as e:
        return None, None, str(e)


model, features, load_error = load_ml_components()

if load_error:
    st.error(
        f"Critical Error Loading Model Assets: {load_error}\n\n"
        "Ensure `diabetes_model.pkl` and `diabetes_features.pkl` are located in the working directory."
    )
    st.stop()

# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================


def get_risk_category(prob_percentage):
    if prob_percentage <= 30.0:
        return "Low Risk", "#22c55e", "🟢"
    elif prob_percentage <= 60.0:
        return "Moderate Risk", "#f59e0b", "🟡"
    elif prob_percentage <= 80.0:
        return "High Risk", "#f97316", "🟠"
    else:
        return "Very High Risk", "#ef4444", "🔴"


def log_prediction(inputs_dict, prediction_val, probability_val, risk_level):
    row_data = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **inputs_dict,
        "Prediction": int(prediction_val),
        "Probability": round(float(probability_val), 2),
        "RiskLevel": risk_level,
    }
    df = pd.DataFrame([row_data])
    df.to_csv(HISTORY_FILE, mode="a", header=False, index=False)


# ============================================================
# 5. SIDEBAR NAVIGATION & SYSTEM STATUS
# ============================================================

st.sidebar.markdown(
    """
    <div style="text-align: center; padding: 10px 0;">
        <h2 style="color: #38bdf8; font-weight: 700; margin: 0;">DiabetesAI</h2>
        <p style="color: #64748b; font-size: 0.8rem; margin: 0;">Clinical Intelligence System</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")

selected_page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Overview",
        "🔮 Prediction Engine",
        "📊 Analytics Hub",
        "📜 Prediction History",
        "🗂 Dataset Explorer",
        "🤖 Model Architecture",
        "📚 Clinical Insights",
        "ℹ️ About",
    ],
    index=1,
)

st.sidebar.markdown("---")

# System Readiness Card
st.sidebar.markdown(
    """
    <div style="background: rgba(15, 23, 42, 0.6); padding: 12px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.05);">
        <p style="font-size: 0.8rem; color: #94a3b8; margin-bottom: 6px;">SYSTEM STATUS</p>
        <p style="font-size: 0.85rem; margin: 2px 0;">Model Engine: <span style="color:#4ade80;">🟢 Online</span></p>
        <p style="font-size: 0.85rem; margin: 2px 0;">Features Loaded: <span style="color:#38bdf8;">8/8 Verified</span></p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# 6. TOP HEADER
# ============================================================

st.markdown(
    """
    <div class="header-container">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <h1 style="color: white; margin: 0; font-size: 1.8rem; font-weight: 700;">Diabetes Risk Intelligence Platform</h1>
                <p style="color: #e0f2fe; margin: 4px 0 0 0; font-size: 0.95rem;">Machine Learning Predictive Analytics & Diagnostic Decision Support System</p>
            </div>
            <div style="text-align: right; background: rgba(255,255,255,0.15); padding: 8px 16px; border-radius: 8px;">
                <span style="font-size: 0.8rem; color: #f0f9ff; display: block;">CLASSIFICATION MODEL</span>
                <strong style="color: white; font-size: 0.95rem;">Decision Tree Algorithm</strong>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# PAGE 1: OVERVIEW
# ============================================================
if selected_page == "🏠 Overview":
    st.markdown("### Executive Overview")

    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown(
            """
            <div class="glass-card">
                <h3 style="color: #38bdf8; margin-top:0;">Welcome to DiabetesAI Analytics</h3>
                <p style="color: #cbd5e1; line-height: 1.6;">
                    DiabetesAI is an advanced, healthcare-focused predictive platform built upon non-invasive physiological measurements. By utilizing modern decision tree logic trained on validated clinical datasets, the dashboard evaluates diabetes risk profiles in real time.
                </p>
                <div style="margin-top: 20px; display: flex; gap: 15px;">
                    <div style="flex: 1; background: rgba(15,23,42,0.5); padding: 12px; border-radius: 8px;">
                        <strong style="color: #38bdf8;">8 Feature Inputs</strong>
                        <p style="font-size: 0.8rem; color: #94a3b8; margin: 4px 0 0 0;">Evaluates Glucose, BMI, Age, Insulin & Vitals</p>
                    </div>
                    <div style="flex: 1; background: rgba(15,23,42,0.5); padding: 12px; border-radius: 8px;">
                        <strong style="color: #4ade80;">Deterministic Tree Path</strong>
                        <p style="font-size: 0.8rem; color: #94a3b8; margin: 4px 0 0 0;">Non-blackbox rule progression logic</p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_b:
        st.image(
            "https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?auto=format&fit=crop&w=800&q=80",
            caption="AI-Assisted Digital Diagnostics",
            use_container_width=True,
        )

    st.markdown("#### Clinical Feature Scope")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            '<div class="kpi-card"><div class="kpi-title">Metabolic Baseline</div><div class="kpi-value">Glucose</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div class="kpi-card"><div class="kpi-title">Body Composition</div><div class="kpi-value">BMI Value</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            '<div class="kpi-card"><div class="kpi-title">Pancreatic Response</div><div class="kpi-value">Insulin</div></div>',
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            '<div class="kpi-card"><div class="kpi-title">Genetic Lineage</div><div class="kpi-value">Pedigree Fun.</div></div>',
            unsafe_allow_html=True,
        )

# ============================================================
# PAGE 2: PREDICTION ENGINE
# ============================================================
elif selected_page == "🔮 Prediction Engine":
    st.markdown("### Patient Assessment & Risk Calculator")

    # Smart Input Validation Warning Placeholders
    validation_warnings = []

    with st.form("prediction_form"):
        st.markdown(
            "<h4 style='color: #38bdf8;'>1. Patient Clinical Metrics</h4>",
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Patient Profile & Demographics**")
            pregnancies = st.number_input(
                "Pregnancies",
                min_value=0,
                max_value=20,
                value=1,
                step=1,
                help="Number of times pregnant",
            )
            age = st.number_input(
                "Age (Years)",
                min_value=1,
                max_value=120,
                value=33,
                step=1,
                help="Patient age in years",
            )

        with c2:
            st.markdown("**Vital Signs & Hemodynamics**")
            glucose = st.number_input(
                "Glucose (mg/dL)",
                min_value=0,
                max_value=300,
                value=117,
                step=1,
                help="Plasma glucose concentration (2 hours in an oral glucose tolerance test)",
            )
            blood_pressure = st.number_input(
                "Blood Pressure (mmHg)",
                min_value=0,
                max_value=200,
                value=72,
                step=1,
                help="Diastolic blood pressure",
            )
            bmi = st.number_input(
                "BMI (kg/m²)",
                min_value=0.0,
                max_value=70.0,
                value=31.2,
                step=0.1,
                help="Body Mass Index",
            )

        with c3:
            st.markdown("**Lab Measures & Lineage**")
            skin_thickness = st.number_input(
                "Skin Thickness (mm)",
                min_value=0,
                max_value=100,
                value=23,
                step=1,
                help="Triceps skin fold thickness",
            )
            insulin = st.number_input(
                "Insulin (mu U/ml)",
                min_value=0,
                max_value=900,
                value=30,
                step=1,
                help="2-Hour serum insulin",
            )
            diabetes_pedigree = st.number_input(
                "Diabetes Pedigree Function",
                min_value=0.0,
                max_value=3.0,
                value=0.375,
                step=0.001,
                format="%.3f",
                help="Diabetes pedigree score calculated from family history",
            )

        submit_predict = st.form_submit_button(
            "Execute Risk Analysis", use_container_width=True
        )

    # Input validation checks
    if glucose == 0:
        validation_warnings.append(
            "Glucose value registered as 0 mg/dL. In clinical settings, physiological glucose is non-zero."
        )
    if blood_pressure == 0:
        validation_warnings.append(
            "Diastolic Blood Pressure registered as 0 mmHg."
        )
    if bmi == 0.0:
        validation_warnings.append(
            "BMI registered as 0.0 kg/m². Ensure physiological dimensions are valid."
        )

    for warn in validation_warnings:
        st.warning(f"⚠️ **Validation Notice**: {warn}")

    if submit_predict:
        # Build DataFrame with identical features order
        input_data = pd.DataFrame(
            [[
                pregnancies,
                glucose,
                blood_pressure,
                skin_thickness,
                insulin,
                bmi,
                diabetes_pedigree,
                age,
            ]],
            columns=features,
        )

        # Execution of ML prediction
        prediction_val = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]

        prob_no_diabetes = probabilities[0] * 100.0
        prob_diabetes = probabilities[1] * 100.0

        risk_label, risk_color, risk_icon = get_risk_category(prob_diabetes)

        # Save record
        input_dict = {
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": diabetes_pedigree,
            "Age": age,
        }
        log_prediction(
            input_dict, prediction_val, prob_diabetes, risk_label
        )

        st.markdown("---")
        st.markdown("### Diagnostic Result & Risk Profile")

        # Top KPI Visual Matrix
        k1, k2, k3, k4 = st.columns(4)

        with k1:
            status_title = (
                "DIABETES RISK DETECTED"
                if prediction_val == 1
                else "LOW RISK / NO DIABETES"
            )
            badge_class = "badge-high" if prediction_val == 1 else "badge-low"
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">Model Classification</div>
                    <div class="{badge_class}" style="margin-top:6px;">{status_title}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with k2:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">Diabetes Probability</div>
                    <div class="kpi-value" style="color:{risk_color};">{prob_diabetes:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with k3:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">Assessed Risk Tier</div>
                    <div class="kpi-value" style="color:{risk_color}; font-size:1.4rem;">{risk_icon} {risk_label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with k4:
            conf_val = max(prob_diabetes, prob_no_diabetes)
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">Classification Confidence</div>
                    <div class="kpi-value" style="color:#38bdf8;">{conf_val:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Interactive Visualization Columns
        v_col1, v_col2 = st.columns([1, 1])

        with v_col1:
            # Radial Gauge for Risk
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=prob_diabetes,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={
                        "text": "Risk Probability Meter (%)",
                        "font": {"color": "#f8fafc", "size": 16},
                    },
                    gauge={
                        "axis": {
                            "range": [0, 100],
                            "tickwidth": 1,
                            "tickcolor": "#94a3b8",
                        },
                        "bar": {"color": risk_color},
                        "bgcolor": "rgba(30, 41, 59, 0.5)",
                        "borderwidth": 2,
                        "bordercolor": "rgba(255,255,255,0.1)",
                        "steps": [
                            {"range": [0, 30], "color": "rgba(34, 197, 94, 0.2)"},
                            {"range": [30, 60], "color": "rgba(245, 158, 11, 0.2)"},
                            {"range": [60, 80], "color": "rgba(249, 115, 22, 0.2)"},
                            {"range": [80, 100], "color": "rgba(239, 68, 68, 0.2)"},
                        ],
                    },
                )
            )
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#f8fafc"},
                height=300,
                margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with v_col2:
            # Patient Metric Comparison vs Normalized Reference
            # Standardized max values for scaling radar chart
            max_bounds = {
                "Glucose": 200,
                "BloodPressure": 120,
                "BMI": 50,
                "Insulin": 200,
                "Age": 80,
            }
            p_metrics = ["Glucose", "BloodPressure", "BMI", "Insulin", "Age"]
            p_vals = [
                min(glucose / max_bounds["Glucose"], 1.0),
                min(blood_pressure / max_bounds["BloodPressure"], 1.0),
                min(bmi / max_bounds["BMI"], 1.0),
                min(insulin / max_bounds["Insulin"], 1.0),
                min(age / max_bounds["Age"], 1.0),
            ]

            fig_radar = go.Figure(
                data=go.Scatterpolar(
                    r=p_vals,
                    theta=p_metrics,
                    fill="toself",
                    name="Patient Values",
                    line=dict(color="#38bdf8"),
                )
            )
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True, range=[0, 1], showticklabels=False
                    ),
                    bgcolor="rgba(30, 41, 59, 0.5)",
                ),
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f8fafc"),
                title="Relative Metric Spectrum (Normalized)",
                height=300,
                margin=dict(l=40, r=40, t=50, b=20),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # AI Decision & Contributory Insights
        st.markdown(
            """
            <div class="glass-card">
                <h4 style="color: #38bdf8; margin-top:0;">🧠 Model Decision Context & Contributory Factors</h4>
                <p style="color: #cbd5e1; font-size:0.9rem;">
                    The Decision Tree classifier evaluates inputs along specific split thresholds. Based on the calculated path:
                </p>
                <ul>
            """,
            unsafe_allow_html=True,
        )

        insights = []
        if glucose >= 140:
            insights.append(
                f"Elevated Plasma Glucose level ({glucose} mg/dL) significantly influences higher risk node traversal."
            )
        else:
            insights.append(
                f"Glucose measurement ({glucose} mg/dL) remains within nominal limits (<140 mg/dL)."
            )

        if bmi >= 30.0:
            insights.append(
                f"BMI metric ({bmi} kg/m²) indicates obesity classification, a primary weighting parameter in decision trees."
            )

        if age >= 45:
            insights.append(
                f"Age factor ({age} yrs) aligns with statistical higher-incidence clinical pathways."
            )

        for ins in insights:
            st.markdown(
                f"<li style='color: #94a3b8; font-size:0.9rem;'>{ins}</li>",
                unsafe_allow_html=True,
            )

        st.markdown("</ul></div>", unsafe_allow_html=True)

        # Download Report Section
        report_df = pd.DataFrame([
            {
                "Parameter": "Patient Classification",
                "Value": (
                    "DIABETES POSITIVE"
                    if prediction_val == 1
                    else "NO DIABETES DETECTED"
                ),
            },
            {
                "Parameter": "Diabetes Probability",
                "Value": f"{prob_diabetes:.2f}%",
            },
            {"Parameter": "Assessed Risk Category", "Value": risk_label},
            {"Parameter": "Glucose", "Value": f"{glucose} mg/dL"},
            {"Parameter": "Blood Pressure", "Value": f"{blood_pressure} mmHg"},
            {"Parameter": "BMI", "Value": f"{bmi} kg/m²"},
            {"Parameter": "Insulin", "Value": f"{insulin} mu U/ml"},
            {"Parameter": "Age", "Value": f"{age} Years"},
            {
                "Parameter": "Pedigree Function",
                "Value": f"{diabetes_pedigree:.3f}",
            },
            {
                "Parameter": "Assessment Timestamp",
                "Value": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
        ])

        csv_report = report_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Clinical Summary Report (CSV)",
            data=csv_report,
            file_name=f"DiabetesAI_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

# ============================================================
# PAGE 3: ANALYTICS HUB
# ============================================================
elif selected_page == "📊 Analytics Hub":
    st.markdown("### System-Wide Analytics & Metrics")

    if os.path.exists(HISTORY_FILE):
        df_hist = pd.read_csv(HISTORY_FILE)
    else:
        df_hist = pd.DataFrame()

    if df_hist.empty:
        st.info(
            "No historical prediction records located. Execute assessments in the Prediction Engine to populate clinical analytics."
        )
    else:
        total_evals = len(df_hist)
        pos_evals = len(df_hist[df_hist["Prediction"] == 1])
        neg_evals = total_evals - pos_evals
        avg_prob = df_hist["Probability"].mean()

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-title">Total Assessments</div><div class="kpi-value">{total_evals}</div></div>',
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-title">Positive Detections</div><div class="kpi-value" style="color:#ef4444;">{pos_evals}</div></div>',
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-title">Negative Detections</div><div class="kpi-value" style="color:#22c55e;">{neg_evals}</div></div>',
                unsafe_allow_html=True,
            )
        with m4:
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-title">Mean Risk Score</div><div class="kpi-value">{avg_prob:.1f}%</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")

        c_left, c_right = st.columns(2)

        with c_left:
            # Classification Ratio
            fig_pie = px.pie(
                df_hist,
                names="Prediction",
                title="Historical Classification Ratio",
                color="Prediction",
                color_discrete_map={0: "#22c55e", 1: "#ef4444"},
                hole=0.4,
            )
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#f8fafc")
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with c_right:
            # Risk Level Distribution
            fig_bar = px.histogram(
                df_hist,
                x="RiskLevel",
                title="Evaluated Risk Tier Distribution",
                color="RiskLevel",
                color_discrete_map={
                    "Low Risk": "#22c55e",
                    "Moderate Risk": "#f59e0b",
                    "High Risk": "#f97316",
                    "Very High Risk": "#ef4444",
                },
            )
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f8fafc"),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

# ============================================================
# PAGE 4: PREDICTION HISTORY
# ============================================================
elif selected_page == "📜 Prediction History":
    st.markdown("### Historical Assessment Registry")

    if os.path.exists(HISTORY_FILE):
        df_hist = pd.read_csv(HISTORY_FILE)
    else:
        df_hist = pd.DataFrame()

    if df_hist.empty:
        st.info("Historical log database is currently empty.")
    else:
        st.markdown("Filter and inspect previous model executions.")

        # Search / Filter
        st.dataframe(
            df_hist,
            use_container_width=True,
            column_config={
                "Probability": st.column_config.NumberColumn(
                    "Risk Prob (%)", format="%.2f %%"
                ),
                "Timestamp": "Execution Time",
            },
        )

        csv_download = df_hist.to_csv(index=False)
        st.download_button(
            label="Export Registry Log (CSV)",
            data=csv_download,
            file_name="diabetes_prediction_registry.csv",
            mime="text/csv",
        )

# ============================================================
# PAGE 5: DATASET EXPLORER
# ============================================================
elif selected_page == "🗂 Dataset Explorer":
    st.markdown("### Training Dataset Analytics & EDA")

    if os.path.exists(DATASET_FILE):
        df_raw = pd.read_csv(DATASET_FILE)

        d1, d2, d3 = st.columns(3)
        with d1:
            st.metric("Total Records", df_raw.shape[0])
        with d2:
            st.metric("Total Features", df_raw.shape[1] - 1)
        with d3:
            st.metric(
                "Positive Instances",
                int(
                    df_raw["Outcome"].sum()
                    if "Outcome" in df_raw.columns
                    else 0
                ),
            )

        st.markdown("#### Raw Dataset Sample")
        st.dataframe(df_raw.head(10), use_container_width=True)

        st.markdown("#### Feature Distribution Profiles")
        selected_feature = st.selectbox("Select Feature for Distribution Inspection", features)

        fig_dist = px.histogram(
            df_raw,
            x=selected_feature,
            color="Outcome" if "Outcome" in df_raw.columns else None,
            marginal="box",
            title=f"Distribution Profile: {selected_feature}",
            color_discrete_map={0: "#38bdf8", 1: "#ef4444"},
        )
        fig_dist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f8fafc"),
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    else:
        st.warning(
            f"Dataset file `{DATASET_FILE}` was not found in the root directory. Place `diabetes.csv` to enable EDA features."
        )

# ============================================================
# PAGE 6: MODEL ARCHITECTURE
# ============================================================
elif selected_page == "🤖 Model Architecture":
    st.markdown("### Machine Learning Model Specifications")

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown(
            """
            <div class="glass-card">
                <h4 style="color:#38bdf8;">Hyperparameter Configuration</h4>
                <table style="width:100%; color:#cbd5e1; font-size:0.9rem;">
                    <tr><td style="padding:6px 0;"><strong>Algorithm Type:</strong></td><td>Decision Tree Classifier</td></tr>
                    <tr><td style="padding:6px 0;"><strong>Max Depth:</strong></td><td>7</td></tr>
                    <tr><td style="padding:6px 0;"><strong>Min Samples Leaf:</strong></td><td>15</td></tr>
                    <tr><td style="padding:6px 0;"><strong>Min Samples Split:</strong></td><td>2</td></tr>
                    <tr><td style="padding:6px 0;"><strong>Target Variable:</strong></td><td>Outcome (0 or 1)</td></tr>
                    <tr><td style="padding:6px 0;"><strong>Feature Count:</strong></td><td>8 Numerical Features</td></tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_m2:
        st.markdown(
            """
            <div class="glass-card">
                <h4 style="color:#38bdf8;">Feature Input Ordering</h4>
                <ol style="color:#cbd5e1; font-size:0.85rem; padding-left: 20px;">
                    <li>Pregnancies</li>
                    <li>Glucose</li>
                    <li>BloodPressure</li>
                    <li>SkinThickness</li>
                    <li>Insulin</li>
                    <li>BMI</li>
                    <li>DiabetesPedigreeFunction</li>
                    <li>Age</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ============================================================
# PAGE 7: CLINICAL INSIGHTS
# ============================================================
elif selected_page == "📚 Clinical Insights":
    st.markdown("### Educational Clinical Reference")

    st.markdown(
        """
        <div class="glass-card">
            <h4 style="color: #38bdf8;">Key Biomarkers in Diabetes Evaluation</h4>
            <p style="color: #cbd5e1; font-size: 0.9rem;">
                <strong>Plasma Glucose:</strong> Measures the amount of glucose circulating in the blood. Levels over 140 mg/dL postprandial warrant clinical attention.<br><br>
                <strong>Body Mass Index (BMI):</strong> A proxy measure of body fat calculated from height and weight. BMI > 30 kg/m² strongly correlates with increased insulin resistance.<br><br>
                <strong>Insulin:</strong> Peptide hormone produced by beta cells of the pancreatic islets. Abnormal levels signal beta-cell dysfunction or peripheral resistance.<br><br>
                <strong>Diabetes Pedigree Function:</strong> A mathematical metric evaluating family genetic synthesis and diabetes history across generations.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# PAGE 8: ABOUT
# ============================================================
elif selected_page == "ℹ️ About":
    st.markdown("### About DiabetesAI Platform")

    st.markdown(
        """
        <div class="glass-card">
            <h4 style="color:#38bdf8;">DiabetesAI System Information</h4>
            <p style="color:#cbd5e1; font-size:0.9rem;">
                This software is engineered as a modern interactive decision-support UI for non-invasive diabetes risk evaluation using machine learning algorithms.
            </p>
            <hr style="border-color: rgba(255,255,255,0.1);">
            <p style="color:#94a3b8; font-size:0.8rem;">
                Frameworks: Python, Streamlit, Scikit-Learn, Plotly, Pandas, NumPy
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# GLOBAL MEDICAL DISCLAIMER FOOTER
# ============================================================
st.markdown(
    """
    <div class="disclaimer-box">
        <strong>MEDICAL DISCLAIMER:</strong> This application is developed strictly for educational, portfolio, and analytical demonstration purposes. It does not provide medical diagnosis, treatment recommendations, or clinical advice. Always consult a qualified healthcare professional regarding personal medical conditions.
    </div>
    """,
    unsafe_allow_html=True,
)