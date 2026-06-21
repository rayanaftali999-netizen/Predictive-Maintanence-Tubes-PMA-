"""
Predictive Maintenance — Streamlit Web App (Deployment Opsi B)
ABK4ABB3 Pembelajaran Mesin dan Aplikasi

Run locally:   streamlit run app.py
Deploy free:   push to a Hugging Face Space (SDK: streamlit) or Streamlit Cloud.
"""
import json, joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Predictive Maintenance", page_icon="🔧", layout="centered")

@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    with open("metrics.json") as f:
        meta = json.load(f)
    return model, scaler, meta

model, scaler, meta = load_artifacts()
FEATURES = meta["feature_names"]
TYPE_MAP = meta["type_mapping"]

st.title("🔧 Prediksi Kerusakan Mesin")
st.caption(f"Predictive Maintenance — model: {meta['winner']} "
           f"(F1={meta['metrics'][meta['winner']]['f1']})")

st.markdown("Masukkan parameter sensor mesin untuk memprediksi risiko kegagalan.")

col1, col2 = st.columns(2)
with col1:
    ptype = st.selectbox("Tipe Produk (kualitas)", ["L", "M", "H"], index=0)
    air = st.number_input("Air temperature [K]", 295.0, 305.0, 300.0, 0.1)
    proc = st.number_input("Process temperature [K]", 305.0, 315.0, 310.0, 0.1)
    rot = st.number_input("Rotational speed [rpm]", 1100, 2900, 1500, 10)
with col2:
    torque = st.number_input("Torque [Nm]", 3.0, 80.0, 40.0, 0.1)
    wear = st.number_input("Tool wear [min]", 0, 260, 100, 1)

def build_features():
    temp_diff = proc - air
    power = torque * rot * 2 * np.pi / 60.0
    wear_torque = wear * torque
    row = {
        "Type": TYPE_MAP[ptype],
        "Air temperature [K]": air,
        "Process temperature [K]": proc,
        "Rotational speed [rpm]": rot,
        "Torque [Nm]": torque,
        "Tool wear [min]": wear,
        "Temp diff [K]": temp_diff,
        "Power [W]": power,
        "Wear*Torque": wear_torque,
    }
    return pd.DataFrame([[row[f] for f in FEATURES]], columns=FEATURES)

if st.button("🔍 Prediksi", type="primary"):
    X = build_features()
    Xs = scaler.transform(X)
    pred = int(model.predict(Xs)[0])
    proba = float(model.predict_proba(Xs)[0][1])
    st.divider()
    if pred == 1:
        st.error(f"⚠️ RISIKO KEGAGALAN TERDETEKSI — probabilitas {proba:.1%}")
    else:
        st.success(f"✅ Mesin Normal — probabilitas kegagalan {proba:.1%}")
    st.progress(proba)
    st.caption("Probabilitas di atas adalah keluaran model untuk kelas 'Failure'.")

with st.expander("ℹ️ Performa Model (test set)"):
    st.json(meta["metrics"])
