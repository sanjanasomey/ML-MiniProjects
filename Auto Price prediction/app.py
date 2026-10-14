import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & THEME SETUP
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AutoPrice AI — Vehicle Price Intelligence",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Light Theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Main Container Light Gradient */
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #edf2f7 50%, #e2e8f0 100%);
        color: #0f172a;
    }

    /* Fixed Sidebar Styling */
    [data-testid="stSidebar"] {
        background: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
        box-shadow: 4px 0 24px rgba(0, 0, 0, 0.02);
    }
    
    [data-testid="stSidebar"] * {
        color: #1e293b;
    }

    /* Sidebar Brand Section */
    .sidebar-brand {
        padding: 24px 16px 16px 16px;
        text-align: center;
        border-bottom: 1px solid #f1f5f9;
        margin-bottom: 20px;
    }
    .brand-title {
        font-size: 1.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #2563eb, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
    }
    .brand-sub {
        font-size: 0.75rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-top: 2px;
    }

    /* Navigation Radio Modernization */
    [data-testid="stSidebar"] .stRadio > div {
        gap: 6px;
    }
    [data-testid="stSidebar"] .stRadio label {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 10px 14px;
        font-weight: 600;
        color: #475569;
        transition: all 0.25s ease;
        cursor: pointer;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        background: #eff6ff;
        border-color: #bfdbfe;
        color: #2563eb;
    }
    [data-testid="stSidebar"] .stRadio label[data-checked="true"] {
        background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
        color: #ffffff !important;
        border-color: #2563eb !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25);
    }

    /* System Status Card */
    .status-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 16px;
        margin-top: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03);
    }
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        background-color: #10b981;
        border-radius: 50%;
        box-shadow: 0 0 10px #10b981;
        margin-right: 6px;
    }

    /* Glassmorphic & Modern Cards */
    .ui-card {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(226, 232, 240, 0.8);
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.04);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        margin-bottom: 20px;
    }
    .ui-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 16px 36px rgba(15, 23, 42, 0.08);
    }

    /* Hero Section Component */
    .hero-wrapper {
        background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%);
        border: 1px solid #e2e8f0;
        border-radius: 24px;
        padding: 40px;
        margin-bottom: 28px;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.04);
        position: relative;
        overflow: hidden;
    }
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.2;
        letter-spacing: -1px;
    }
    .hero-subtitle {
        color: #475569;
        font-size: 1.1rem;
        margin-top: 12px;
        line-height: 1.6;
    }

    /* Metric Cards */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 20px;
        text-align: left;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.02);
    }
    .metric-title {
        font-size: 0.75rem;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    .metric-val {
        font-size: 1.75rem;
        font-weight: 800;
        color: #0f172a;
        margin-top: 6px;
    }

    /* Process Flow Steps */
    .step-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        position: relative;
    }
    .step-num {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        background: linear-gradient(135deg, #2563eb, #7c3aed);
        color: #ffffff;
        font-weight: 800;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 12px auto;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }

    /* Price Output Box */
    .price-display-card {
        background: linear-gradient(135deg, #eff6ff 0%, #f5f3ff 100%);
        border: 2px solid #bfdbfe;
        border-radius: 24px;
        padding: 36px;
        text-align: center;
        box-shadow: 0 12px 32px rgba(37, 99, 235, 0.1);
        margin-top: 24px;
        animation: fadeIn 0.8s ease-in-out;
    }
    .price-amount {
        font-size: 3.2rem;
        font-weight: 900;
        background: linear-gradient(135deg, #1d4ed8, #6d28d9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 8px 0;
    }

    /* Analytics Metric Visibility Fix */
    [data-testid="stMetric"] {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 18px !important;
        padding: 18px 20px !important;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.05) !important;
    }

    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] *,
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] *,
    [data-testid="stMetricDelta"],
    [data-testid="stMetricDelta"] * {
        opacity: 1 !important;
        visibility: visible !important;
    }

    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] * {
        color: #475569 !important;
        font-weight: 700 !important;
    }

    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] * {
        color: #0f172a !important;
        font-weight: 800 !important;
    }

    [data-testid="stMetricDelta"],
    [data-testid="stMetricDelta"] * {
        color: #64748b !important;
    }

    /* Analytics heading and supporting text */
    .analytics-subtitle {
        color: #475569 !important;
    }

    /* Modern Buttons */
    .stButton > button {
        border-radius: 14px !important;
        font-weight: 700 !important;
        padding: 12px 28px !important;
        transition: all 0.3s ease !important;
    }

    /* Hide standard Streamlit header & footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. FILE LOADING & BACKEND INTEGRATION
# -----------------------------------------------------------------------------
MODEL_FILE = "random_forest_model.pkl"
FEATURE_FILE = "features.pkl"
DATASET_FILE = "autos_dataset.csv"
HISTORY_FILE = "prediction_history.csv"

# Fallback default features if pickle file is unavailable during initialization
DEFAULT_FEATURES = [
    "symboling", "wheel-base", "length", "width", "height", "curb-weight",
    "engine-size", "compression-ratio", "horsepower", "peak-rpm",
    "city-mpg", "highway-mpg", "num-of-cylinders"
]

@st.cache_resource
def load_ml_assets():
    model, features, error = None, None, None
    try:
        if os.path.exists(MODEL_FILE):
            with open(MODEL_FILE, "rb") as f:
                model = pickle.load(f)
        else:
            error = f"Model file '{MODEL_FILE}' not found."

        if os.path.exists(FEATURE_FILE):
            with open(FEATURE_FILE, "rb") as f:
                features = pickle.load(f)
        else:
            features = DEFAULT_FEATURES

    except Exception as e:
        error = str(e)

    return model, features, error

model, features, load_error = load_ml_assets()

def save_to_history(input_data, predicted_price):
    """Save predictions without breaking an existing history CSV schema."""
    record = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **input_data,
        "Predicted Price": round(float(predicted_price), 2)
    }
    df_new = pd.DataFrame([record])

    # If an old history file exists with a different schema, migrate it first.
    if os.path.exists(HISTORY_FILE):
        try:
            df_old = pd.read_csv(HISTORY_FILE)

            # Common old-schema variants for the prediction column.
            aliases = {
                "predicted_price": "Predicted Price",
                "predicted price": "Predicted Price",
                "prediction": "Predicted Price",
                "price": "Predicted Price"
            }
            rename_map = {}
            for col in df_old.columns:
                key = str(col).strip().lower()
                if key in aliases and aliases[key] not in df_old.columns:
                    rename_map[col] = aliases[key]
                    break
            if rename_map:
                df_old = df_old.rename(columns=rename_map)

            # Keep old rows only if they contain a usable prediction column.
            if "Predicted Price" in df_old.columns:
                df_old["Predicted Price"] = pd.to_numeric(
                    df_old["Predicted Price"], errors="coerce"
                )
                df_old = df_old.dropna(subset=["Predicted Price"])
                df_out = pd.concat([df_old, df_new], ignore_index=True, sort=False)
            else:
                # Existing file is unrelated/invalid for prediction history.
                # Start a clean prediction history rather than appending incompatible columns.
                df_out = df_new

            df_out.to_csv(HISTORY_FILE, index=False)
        except Exception:
            # If the old CSV is unreadable, replace it with a valid history file.
            df_new.to_csv(HISTORY_FILE, index=False)
    else:
        df_new.to_csv(HISTORY_FILE, index=False)

# -----------------------------------------------------------------------------
# 3. SIDEBAR NAVIGATION & SYSTEM STATUS
# -----------------------------------------------------------------------------
st.sidebar.markdown("""
<div class="sidebar-brand">
    <div style="font-size: 2.2rem; margin-bottom: 4px;">🚘</div>
    <div class="brand-title">AutoPrice AI</div>
    <div class="brand-sub">Vehicle Price Intelligence</div>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "🏠  Dashboard",
        "💰  Predict Price",
        "📊  Analytics",
        "🚗  Vehicle Gallery",
        "🗃️  Dataset Explorer",
        "🌲  AI Model",
        "ℹ️  About Project"
    ]
)

# Sidebar System Status Card
st.sidebar.markdown(f"""
<div class="status-card">
    <div style="font-size: 0.72rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px;">
        System Status
    </div>
    <div style="display: flex; align-items: center; margin-top: 6px; font-weight: 700; color: #0f172a;">
        <span class="status-indicator" style="background-color: {'#10b981' if model else '#ef4444'}; box-shadow: 0 0 10px {'#10b981' if model else '#ef4444'};"></span>
        {'AI MODEL ONLINE' if model else 'MODEL OFFLINE'}
    </div>
    <div style="font-size: 0.78rem; color: #64748b; margin-top: 6px;">
        Random Forest Regressor<br>
        <strong>{len(features)}</strong> Active Inputs
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 4. PAGE 1: DASHBOARD PAGE
# -----------------------------------------------------------------------------
if page == "🏠  Dashboard":
    # Hero Section
    c_hero1, c_hero2 = st.columns([1.3, 1])
    with c_hero1:
        st.markdown("""
        <div class="hero-wrapper">
            <span style="background: rgba(37, 99, 235, 0.1); color: #2563eb; padding: 6px 14px; border-radius: 999px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px;">
                Intelligent Platform
            </span>
            <h1 class="hero-title" style="margin-top: 14px;">Discover the True Value of Every Drive</h1>
            <p class="hero-subtitle">
                AI-powered automobile price prediction using intelligent machine learning and vehicle analytics.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with c_hero2:
        st.image(
            "https://images.unsplash.com/photo-1617814076367-b759c7d7e738?auto=format&fit=crop&w=1000&q=80",
            caption="Futuristic Luxury Concept",
            use_container_width=True
        )

    # KPI Metrics
    dataset_rows = "—"
    if os.path.exists(DATASET_FILE):
        try:
            dataset_rows = str(len(pd.read_csv(DATASET_FILE)))
        except:
            pass

    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(f'<div class="metric-card"><div class="metric-title">🤖 AI Algorithm</div><div class="metric-val">Random Forest</div></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="metric-card"><div class="metric-title">📊 Input Features</div><div class="metric-val">{len(features)}</div></div>', unsafe_allow_html=True)
    m3.markdown(f'<div class="metric-card"><div class="metric-title">🗃️ Dataset Rows</div><div class="metric-val">{dataset_rows}</div></div>', unsafe_allow_html=True)
    m4.markdown(f'<div class="metric-card"><div class="metric-title">⚡ Prediction Engine</div><div class="metric-val" style="color: #10b981;">Live & Ready</div></div>', unsafe_allow_html=True)

    st.write("")
    st.write("")

    # How AutoPrice AI Works
    st.markdown("### How AutoPrice AI Works")
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown('<div class="step-box"><div class="step-num">1</div><h4 style="margin:0;">Vehicle Details</h4><p style="font-size: 0.82rem; color: #64748b; margin-top: 6px;">Enter specifications like horsepower, size, and MPG.</p></div>', unsafe_allow_html=True)
    with s2:
        st.markdown('<div class="step-box"><div class="step-num">2</div><h4 style="margin:0;">AI Analysis</h4><p style="font-size: 0.82rem; color: #64748b; margin-top: 6px;">Features are structured matching model parameters.</p></div>', unsafe_allow_html=True)
    with s3:
        st.markdown('<div class="step-box"><div class="step-num">3</div><h4 style="margin:0;">Random Forest</h4><p style="font-size: 0.82rem; color: #64748b; margin-top: 6px;">Decision tree ensembles aggregate predictions.</p></div>', unsafe_allow_html=True)
    with s4:
        st.markdown('<div class="step-box"><div class="step-num">4</div><h4 style="margin:0;">Get Estimate</h4><p style="font-size: 0.82rem; color: #64748b; margin-top: 6px;">Receive an instant accurate market value estimate.</p></div>', unsafe_allow_html=True)

    st.write("")
    st.write("")

    # Explore Popular Vehicles Showcase
    st.markdown("### Explore Popular Vehicles")
    g1, g2, g3 = st.columns(3)
    with g1:
        st.image("https://images.unsplash.com/photo-1503376780353-7e6692767b70?auto=format&fit=crop&w=600&q=80", use_container_width=True)
        st.markdown('<div class="ui-card"><span style="font-size:0.75rem; font-weight:700; color:#2563eb;">SPORTS</span><h4>Performance GT</h4><p style="font-size:0.85rem; color:#64748b;">High horsepower, aerodynamic body design, lightweight frame.</p><p><strong>Estimated: $55,000 - $85,000</strong></p></div>', unsafe_allow_html=True)
    with g2:
        st.image("https://images.unsplash.com/photo-1555215695-3004980ad54e?auto=format&fit=crop&w=600&q=80", use_container_width=True)
        st.markdown('<div class="ui-card"><span style="font-size:0.75rem; font-weight:700; color:#7c3aed;">LUXURY SEDAN</span><h4>Executive Touring</h4><p style="font-size:0.85rem; color:#64748b;">Extended wheelbase, high comfort specs, mid-range engine compression.</p><p><strong>Estimated: $40,000 - $65,000</strong></p></div>', unsafe_allow_html=True)
    with g3:
        st.image("https://images.unsplash.com/photo-1533473359331-0135ef1b58bf?auto=format&fit=crop&w=600&q=80", use_container_width=True)
        st.markdown('<div class="ui-card"><span style="font-size:0.75rem; font-weight:700; color:#059669;">SUV</span><h4>Urban Crossover</h4><p style="font-size:0.85rem; color:#64748b;">Higher height profile, optimized fuel economy, durable chassis.</p><p><strong>Estimated: $25,000 - $45,000</strong></p></div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. PAGE 2: PREDICT PRICE PAGE
# -----------------------------------------------------------------------------
elif page == "💰  Predict Price":
    st.markdown("""
    <div style="text-align: center; margin-bottom: 24px;">
        <h1 style="font-size: 2.2rem; font-weight: 800; color: #0f172a;">Predict Your Vehicle's Value</h1>
        <p style="color: #64748b; font-size: 1rem;">Enter your automobile specifications and let AutoPrice AI estimate its market value.</p>
    </div>
    """, unsafe_allow_html=True)

    st.image("https://images.unsplash.com/photo-1492144534655-ae79c964c9d7?auto=format&fit=crop&w=1200&q=80", use_container_width=True)
    st.write("")

    if model is None:
        st.error(f"⚠️ Unable to run predictions. Please ensure '{MODEL_FILE}' exists in the application root directory.")
        st.stop()

    # Dynamic Field Mapper based on features.pkl order
    feature_defaults = {
        "symboling": (-3, 3, 0, "Risk rating factor (-3 to 3)"),
        "wheel-base": (80.0, 130.0, 98.8, "Wheelbase length in inches"),
        "length": (140.0, 210.0, 174.0, "Overall length in inches"),
        "width": (60.0, 75.0, 65.9, "Overall width in inches"),
        "height": (45.0, 60.0, 53.7, "Overall height in inches"),
        "curb-weight": (1400, 4100, 2555, "Total weight without passengers (lbs)"),
        "engine-size": (60, 326, 130, "Engine displacement size"),
        "compression-ratio": (7.0, 23.0, 10.1, "Engine compression ratio"),
        "horsepower": (48, 288, 104, "Peak engine horsepower"),
        "peak-rpm": (4150, 6600, 5125, "Engine peak RPM"),
        "city-mpg": (13, 49, 25, "Fuel efficiency in city driving"),
        "highway-mpg": (16, 54, 30, "Fuel efficiency on highway"),
        "num-of-cylinders": (2, 12, 4, "Total engine cylinders")
    }

    input_values = {}

    st.markdown('<div class="ui-card">', unsafe_allow_html=True)
    st.subheader("⚙️ Vehicle Specifications Form")
    st.write("Fill in the fields below. Features are processed dynamically to maintain model compatibility.")
    st.write("")

    # Dynamically render inputs while preserving features.pkl array order
    c1, c2 = st.columns(2)
    for idx, feature_name in enumerate(features):
        feat_key = str(feature_name).strip().lower()
        col = c1 if idx % 2 == 0 else c2

        with col:
            if feat_key in feature_defaults:
                min_v, max_v, def_v, help_txt = feature_defaults[feat_key]
                if isinstance(def_v, float):
                    input_values[feature_name] = st.slider(
                        label=f"{str(feature_name).replace('-', ' ').title()}",
                        min_value=float(min_v),
                        max_value=float(max_v),
                        value=float(def_v),
                        help=help_txt
                    )
                else:
                    input_values[feature_name] = st.number_input(
                        label=f"{str(feature_name).replace('-', ' ').title()}",
                        min_value=int(min_v),
                        max_value=int(max_v),
                        value=int(def_v),
                        help=help_txt
                    )
            else:
                input_values[feature_name] = st.number_input(
                    label=f"{str(feature_name).replace('-', ' ').title()}",
                    value=0.0
                )

    st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    if st.button("🚀  Predict Vehicle Price", type="primary", use_container_width=True):
        try:
            # Maintain exact feature ordering as specified in features.pkl
            input_vector = [input_values[f] for f in features]
            prediction = model.predict([input_vector])[0]

            # Save prediction to history file
            save_to_history(input_values, prediction)

            st.balloons()
            st.markdown(f"""
            <div class="price-display-card">
                <span style="background: #dbeafe; color: #1e40af; font-weight: 700; padding: 6px 16px; border-radius: 999px; font-size: 0.8rem; letter-spacing: 1px;">
                    ✨ AI PREDICTION COMPLETE
                </span>
                <div style="font-size: 0.9rem; color: #64748b; margin-top: 16px; text-transform: uppercase; font-weight: 700;">
                    Estimated Market Value
                </div>
                <div class="price-amount">${float(prediction):,.2f}</div>
                <div style="font-size: 0.82rem; color: #64748b; margin-top: 8px;">
                    Calculated via Random Forest Regressor Ensemble • Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </div>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction Calculation Error: {e}")

# -----------------------------------------------------------------------------
# 6. PAGE 3: ANALYTICS PAGE
# -----------------------------------------------------------------------------
elif page == "📊  Analytics":
    st.markdown("## 📊 Analytics Dashboard")
    st.markdown('<p class="analytics-subtitle">Real-time telemetry and distribution analysis from saved prediction history.</p>', unsafe_allow_html=True)

    # Safely load and normalize prediction history.
    df_hist = pd.DataFrame()

    if os.path.exists(HISTORY_FILE):
        try:
            df_hist = pd.read_csv(HISTORY_FILE)

            # Fix old CSV files that used "price" or another common name.
            aliases = {
                "predicted_price": "Predicted Price",
                "predicted price": "Predicted Price",
                "prediction": "Predicted Price",
                "price": "Predicted Price"
            }

            if "Predicted Price" not in df_hist.columns:
                for col in df_hist.columns:
                    if str(col).strip().lower() in aliases:
                        df_hist = df_hist.rename(columns={col: "Predicted Price"})
                        break

            if "Predicted Price" in df_hist.columns:
                df_hist["Predicted Price"] = pd.to_numeric(
                    df_hist["Predicted Price"], errors="coerce"
                )
                df_hist = df_hist.dropna(subset=["Predicted Price"]).copy()

            if "Timestamp" in df_hist.columns:
                df_hist["Timestamp"] = pd.to_datetime(
                    df_hist["Timestamp"], errors="coerce"
                )

        except Exception as e:
            st.error(f"Unable to read prediction history: {e}")
            df_hist = pd.DataFrame()

    if df_hist.empty or "Predicted Price" not in df_hist.columns:
        st.markdown("""
        <div class="ui-card" style="text-align: center; padding: 60px 20px;">
            <div style="font-size: 3rem; margin-bottom: 12px;">📊</div>
            <h3>No Valid Predictions Recorded Yet</h3>
            <p style="color: #64748b; max-width: 500px; margin: 0 auto 20px auto;">
                Make a prediction from the <strong>Predict Price</strong> page.
                Analytics will automatically appear here after a valid prediction is saved.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("📈 Total Predictions", len(df_hist))
        k2.metric(
            "💰 Average Estimated Price",
            f"${df_hist['Predicted Price'].mean():,.2f}"
        )
        k3.metric(
            "🔥 Highest Estimate",
            f"${df_hist['Predicted Price'].max():,.2f}"
        )
        k4.metric(
            "📉 Lowest Estimate",
            f"${df_hist['Predicted Price'].min():,.2f}"
        )

        st.write("")
        c_left, c_right = st.columns(2)

        with c_left:
            if "Timestamp" in df_hist.columns and df_hist["Timestamp"].notna().any():
                trend_df = df_hist.dropna(subset=["Timestamp"]).sort_values("Timestamp")
                fig_trend = px.line(
                    trend_df,
                    x="Timestamp",
                    y="Predicted Price",
                    markers=True,
                    title="Prediction History Trend",
                    color_discrete_sequence=["#2563eb"]
                )
            else:
                trend_df = df_hist.reset_index()
                trend_df["Prediction No."] = trend_df.index + 1
                fig_trend = px.line(
                    trend_df,
                    x="Prediction No.",
                    y="Predicted Price",
                    markers=True,
                    title="Prediction History Trend",
                    color_discrete_sequence=["#2563eb"]
                )

            fig_trend.update_layout(
                template="plotly_white",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_trend, use_container_width=True)

        with c_right:
            fig_dist = px.histogram(
                df_hist,
                x="Predicted Price",
                nbins=max(1, min(15, len(df_hist))),
                title="Price Distribution Analysis",
                color_discrete_sequence=["#7c3aed"]
            )
            fig_dist.update_layout(
                template="plotly_white",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_dist, use_container_width=True)

        st.write("### Recent Prediction Logs")
        st.dataframe(df_hist.tail(10), use_container_width=True)

# -----------------------------------------------------------------------------
# 7. PAGE 4: VEHICLE GALLERY PAGE
# -----------------------------------------------------------------------------
elif page == "🚗  Vehicle Gallery":
    st.markdown("## 🚗 Automobile Visual Gallery")
    st.write("Explore automotive categories and performance profiles.")

    gallery_items = [
        {"title": "High Performance Sport", "cat": "🏎️ Sports", "img": "https://images.unsplash.com/photo-1503376780353-7e6692767b70?auto=format&fit=crop&w=600&q=80", "desc": "Aerodynamic chassis designed for speed and rapid throttle response."},
        {"title": "Executive Sedan", "cat": "👑 Luxury", "img": "https://images.unsplash.com/photo-1555215695-3004980ad54e?auto=format&fit=crop&w=600&q=80", "desc": "Premium interior finishings with calibrated engine noise suppression."},
        {"title": "All-Terrain SUV", "cat": "🚙 SUV", "img": "https://images.unsplash.com/photo-1533473359331-0135ef1b58bf?auto=format&fit=crop&w=600&q=80", "desc": "High ground clearance paired with robust multi-axle torque distribution."},
        {"title": "Futuristic EV Concept", "cat": "⚡ Electric", "img": "https://images.unsplash.com/photo-1563720223185-11003d516935?auto=format&fit=crop&w=600&q=80", "desc": "Zero-emission powertrain utilizing next-gen energy storage."},
        {"title": "Urban Compact", "cat": "🚗 Compact", "img": "https://images.unsplash.com/photo-1541899481282-d53bffe3c35d?auto=format&fit=crop&w=600&q=80", "desc": "Optimized fuel consumption built specifically for city commuting."},
        {"title": "Grand Tourer", "cat": "🏎️ Sports", "img": "https://images.unsplash.com/photo-1542282088-72c9c27ed0cd?auto=format&fit=crop&w=600&q=80", "desc": "Long-distance high-speed highway touring vehicle."}
    ]

    col1, col2, col3 = st.columns(3)
    for idx, item in enumerate(gallery_items):
        col = [col1, col2, col3][idx % 3]
        with col:
            st.image(item["img"], use_container_width=True)
            st.markdown(f"""
            <div class="ui-card">
                <span style="font-size: 0.75rem; font-weight: 700; color: #2563eb;">{item['cat']}</span>
                <h4 style="margin: 4px 0;">{item['title']}</h4>
                <p style="font-size: 0.85rem; color: #64748b;">{item['desc']}</p>
            </div>
            """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 8. PAGE 5: DATASET EXPLORER PAGE
# -----------------------------------------------------------------------------
elif page == "🗃️  Dataset Explorer":
    st.markdown("## 🗃️ Dataset Explorer")
    st.write("Inspect automobile specifications data used for evaluation.")

    if os.path.exists(DATASET_FILE):
        try:
            df_data = pd.read_csv(DATASET_FILE)
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Total Rows", df_data.shape[0])
            d2.metric("Total Columns", df_data.shape[1])
            d3.metric("Missing Values", int(df_data.isna().sum().sum()))
            d4.metric("Target Variable", "price")

            st.write("")
            st.markdown('<div class="ui-card">', unsafe_allow_html=True)
            st.subheader("Dataset Insights")
            st.write("This dataset contains key technical metrics for automobiles including engine specifications, dimensions, and fuel ratings used during model training.")
            st.dataframe(df_data, use_container_width=True, height=400)
            st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error loading CSV file: {e}")
    else:
        st.warning(f"⚠️ '{DATASET_FILE}' missing from project directory. Please place it in the application folder.")

# -----------------------------------------------------------------------------
# 9. PAGE 6: AI MODEL PAGE
# -----------------------------------------------------------------------------
elif page == "🌲  AI Model":
    st.markdown("## 🌲 Inside the AI Engine")
    st.write("Architecture summary of the Random Forest Regressor pipeline.")

    st.markdown("""
    <div class="ui-card" style="text-align: center; padding: 30px;">
        <h3 style="margin-bottom: 20px;">Model Decision Flow</h3>
        <div style="display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap; gap: 10px;">
            <div style="background: #eff6ff; border: 1px solid #bfdbfe; padding: 12px 20px; border-radius: 12px; font-weight: 700;">Vehicle Features</div>
            <div style="font-size: 1.5rem; color: #2563eb;">➔</div>
            <div style="background: #f5f3ff; border: 1px solid #ddd6fe; padding: 12px 20px; border-radius: 12px; font-weight: 700;">Decision Trees Ensemble</div>
            <div style="font-size: 1.5rem; color: #7c3aed;">➔</div>
            <div style="background: #ecfdf5; border: 1px solid #a7f3d0; padding: 12px 20px; border-radius: 12px; font-weight: 700;">Random Forest Averaging</div>
            <div style="font-size: 1.5rem; color: #059669;">➔</div>
            <div style="background: #fff7ed; border: 1px solid #fed7aa; padding: 12px 20px; border-radius: 12px; font-weight: 700;">Predicted Market Price</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    a1, a2 = st.columns(2)
    with a1:
        st.markdown(f"""
        <div class="ui-card">
            <h4>📦 System Artifacts</h4>
            <p><strong>Algorithm:</strong> Random Forest Regressor</p>
            <p><strong>Task:</strong> Automobile Price Regression</p>
            <p><strong>Model File:</strong> {MODEL_FILE}</p>
            <p><strong>Features Configuration:</strong> {FEATURE_FILE}</p>
            <p><strong>Target Variable:</strong> Vehicle Price (USD)</p>
        </div>
        """, unsafe_allow_html=True)

    with a2:
        st.markdown("""
        <div class="ui-card">
            <h4>Why Random Forest?</h4>
            <p>🌲 <strong>Ensemble Power:</strong> Combines predictions from multiple decision trees to minimize individual tree variance.</p>
            <p>⚡ <strong>Non-linear Fitting:</strong> Captures complex interactions between dimensions, horsepower, and fuel economy.</p>
            <p>🛡️ <strong>Robustness:</strong> High resistance to overfitting compared to single estimator trees.</p>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 10. PAGE 7: ABOUT PROJECT PAGE
# -----------------------------------------------------------------------------
elif page == "ℹ️  About Project":
    st.markdown("## ℹ️ About AutoPrice AI")
    st.write("A modern machine learning platform designed to estimate automobile prices through data-driven prediction.")

    st.write("")
    i1, i2 = st.columns(2)
    with i1:
        st.markdown("""
        <div class="ui-card">
            <h4>🎯 Project Objective</h4>
            <p style="color: #64748b;">To build an accessible price prediction platform for consumer vehicles by analyzing technical specifications and performance indicators via supervised machine learning models.</p>
        </div>
        """, unsafe_allow_html=True)
    with i2:
        st.markdown("""
        <div class="ui-card">
            <h4>🧠 Machine Learning Core</h4>
            <p style="color: #64748b;">Utilizes Scikit-Learn implementation of Random Forest Regressor trained on structured numerical automobile datasets.</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.markdown("### Project Workflow")
    st.markdown("""
    <div class="ui-card">
        <div style="display: flex; justify-content: space-between; flex-wrap: wrap; text-align: center; gap: 10px;">
            <div><strong>1. Dataset</strong><br><span style="font-size:0.8rem; color:#64748b;">Automobile Specs</span></div>
            <div>➔</div>
            <div><strong>2. Cleaning</strong><br><span style="font-size:0.8rem; color:#64748b;">Missing Imputation</span></div>
            <div>➔</div>
            <div><strong>3. Feature Selection</strong><br><span style="font-size:0.8rem; color:#64748b;">Export features.pkl</span></div>
            <div>➔</div>
            <div><strong>4. RF Training</strong><br><span style="font-size:0.8rem; color:#64748b;">Scikit-Learn Model</span></div>
            <div>➔</div>
            <div><strong>5. Export</strong><br><span style="font-size:0.8rem; color:#64748b;">random_forest_model.pkl</span></div>
            <div>➔</div>
            <div><strong>6. Interactive UI</strong><br><span style="font-size:0.8rem; color:#64748b;">Streamlit Dashboard</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
st.write("")
st.write("")
st.markdown("""
<div style="text-align: center; border-top: 1px solid #e2e8f0; padding-top: 20px; color: #94a3b8; font-size: 0.8rem;">
    AutoPrice AI Platform • Intelligent Machine Learning Automobile Analytics • Enterprise Dashboard Interface
</div>
""", unsafe_allow_html=True)