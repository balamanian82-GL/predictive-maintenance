import streamlit as st
import pandas as pd
import pickle
import numpy as np
from huggingface_hub import hf_hub_download

# ── Page config MUST be first Streamlit command ────────────────────────────
st.set_page_config(
    page_title="Engine Predictive Maintenance",
    page_icon="🔧",
    layout="wide"
)

# ── Feature list — must match training column order exactly ────────────────
FEATURE_COLS = ['Engine_RPM', 'Lub_Oil_Pressure', 'Fuel_Pressure', 'Coolant_Pressure', 'Lub_Oil_Temp', 'Coolant_Temp', 'temp_pressure_ratio']

# ── Load model from HF Hub ─────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="balamanian82/predictive-maintenance-engine-model",
        filename="best_model.pkl",
        repo_type="model"
    )
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()
OPTIMAL_THRESHOLD = 0.35

# ── Title ──────────────────────────────────────────────────────────────────
st.title("🔧 Engine Predictive Maintenance")
st.markdown(
    "Predict whether an engine requires maintenance based on "
    "real-time sensor readings using an XGBoost classifier "
    "(Val F1 = 0.76 | Recall = 0.86 at threshold 0.35)."
)
st.markdown("---")

# ── Sidebar inputs ─────────────────────────────────────────────────────────
st.sidebar.header("📊 Engine Sensor Inputs")

engine_rpm       = st.sidebar.slider("Engine RPM",                   300,   1500, 750)
lub_oil_pressure = st.sidebar.slider("Lub Oil Pressure (bar)",        1.0,    6.0, 3.5, step=0.1)
fuel_pressure    = st.sidebar.slider("Fuel Pressure (kPa)",           1.0,   20.0, 8.0, step=0.1)
coolant_pressure = st.sidebar.slider("Coolant Pressure (bar)",        1.0,    5.0, 2.5, step=0.1)
lub_oil_temp     = st.sidebar.slider("Lub Oil Temperature (°C)", 60.0, 100.0, 78.0, step=0.5)
coolant_temp     = st.sidebar.slider("Coolant Temperature (°C)", 60.0, 100.0, 78.0, step=0.5)

temp_pressure_ratio = lub_oil_temp / (coolant_pressure + 1e-6)

# ── Build input DataFrame — column names must match training exactly ───────
input_df = pd.DataFrame([{
    "Engine_RPM"          : engine_rpm,
    "Lub_Oil_Pressure"    : lub_oil_pressure,
    "Fuel_Pressure"       : fuel_pressure,
    "Coolant_Pressure"    : coolant_pressure,
    "Lub_Oil_Temp"        : lub_oil_temp,
    "Coolant_Temp"        : coolant_temp,
    "temp_pressure_ratio" : temp_pressure_ratio,
}])[FEATURE_COLS]   # reorder to match training

# ── Display inputs ─────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    st.subheader("Input Sensor Readings")
    st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)

# ── Predict ────────────────────────────────────────────────────────────────
probability  = model.predict_proba(input_df)[0][1]
pred_default = int(probability >= 0.50)
pred_optimal = int(probability >= OPTIMAL_THRESHOLD)

with col2:
    st.subheader("Prediction Results")
    st.metric("Maintenance Probability", f"{probability:.1%}")

    st.markdown("**Default threshold (0.50):**")
    if pred_default == 1:
        st.error("⚠️ MAINTENANCE REQUIRED")
    else:
        st.success("✅ ENGINE NORMAL")

    st.markdown(f"**Optimised threshold ({OPTIMAL_THRESHOLD}):**")
    if pred_optimal == 1:
        st.warning("⚡ MAINTENANCE FLAGGED (high recall mode)")
    else:
        st.info("⚡ ENGINE NORMAL (high recall mode)")

# ── Risk gauge chart ───────────────────────────────────────────────────────
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(5, 2))
colour = "tomato" if pred_optimal == 1 else "steelblue"
ax.barh(["Risk"], [probability],       color=colour,    height=0.4)
ax.barh(["Risk"], [1 - probability],   left=[probability], color="#e0e0e0", height=0.4)
ax.axvline(OPTIMAL_THRESHOLD, color="orange", linestyle="--", linewidth=1.5,
           label=f"Threshold ({OPTIMAL_THRESHOLD})")
ax.set_xlim(0, 1)
ax.set_xlabel("Maintenance Probability")
ax.set_title(f"Risk Score: {probability:.1%}")
ax.legend(fontsize=8)
st.pyplot(fig)

# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Model: XGBoost | Training: SMOTE oversampled | Dataset: Engine Sensor Readings")
