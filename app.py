import streamlit as st
import numpy as np
import joblib
import random

# Load files
model = joblib.load("stress_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

st.set_page_config(page_title="Stress Monitoring", page_icon="🧠", layout="wide")

# UI styling
st.markdown("""
<style>
body { background: linear-gradient(120deg,#7f7fd5,#86a8e7,#91eae4); font-family:'Arial'; }
.card { background:white; padding:25px; border-radius:15px; width:420px; margin:auto;
        box-shadow:0 4px 20px rgba(0,0,0,0.25); }
h1 { text-align:center; font-size:40px; color:white; font-weight:bold; }
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

quotes = [
    "🌿 Relax… one breath at a time.",
    "💪 You are stronger than your stress.",
    "✨ Just breathe, everything will be okay.",
]
st.markdown("<h1>🧠 Stress Level Detection</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align:center; font-size:20px; color:white;'>{random.choice(quotes)}</p>", unsafe_allow_html=True)


# ✅ Rule-based classification (Medical Thresholds)
def rule_based(st_spo2, st_hr):
    if st_spo2 < 60 and st_hr < 60:
        return "Critical Hypoxia (Severe Stress)"
    if 60 <= st_spo2 < 80 and 60 <= st_hr < 80:
        return "High Physiological Stress"
    if 80 <= st_spo2 <= 100 and 80 <= st_hr <= 100:
        return "Normal"
    if 100 < st_hr <= 120 and 100 <= st_hr <= 120:
        return "Moderate Stress"
    if st_hr > 120 and st_spo2 > 120:
        return "Severe Stress"
    return None


def stress_suggestions(stress_class):
    stress_class = stress_class.lower()
    if "high" in stress_class or "severe" in stress_class:
        return ["🧘 Deep breathing (4–4–6 method)", "💃 Dance", "🚴 Cycling", "🏊 Swimming", "🎧 Calm music"]
    if "moderate" in stress_class:
        return ["🚶 Short walk", "🎯 Hobby time", "☀ Sunlight exposure", "📞 Talk to a friend"]
    return ["✅ You're fine!", "💧 Stay hydrated", "🙂 Stay positive"]


# UI INPUT FORM
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.subheader("📊 Enter Your Values")
    spo2 = st.number_input("SpO₂ (%)", min_value=40, max_value=100, value=97)
    hr = st.number_input("Heart Rate (BPM)", min_value=40, max_value=200, value=90)

    if st.button("🔍 Predict Stress", use_container_width=True):

        # ✅ Step-1: Medical rule-based validation
        ruled = rule_based(spo2, hr)

        if ruled:
            stress_label = ruled  # highest accuracy logic
        else:
            # ✅ Step-2: ML Prediction fallback
            scaled = scaler.transform([[spo2, hr]])
            prediction = model.predict(scaled)[0]
            stress_label = label_encoder.inverse_transform([prediction])[0]

        st.success(f"✨ Stress Result: **{stress_label}**")

        st.subheader("💡 Suggested Activities:")
        for tip in stress_suggestions(stress_label):
            st.write(f"- {tip}")

    st.markdown("</div>", unsafe_allow_html=True)
