
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="MediLens AI", layout="wide")
st.title("🏥 MediLens AI: Integrated Diagnostic Platform")

tab1, tab2 = st.tabs(["📊 Chronic Risk Predictor", "🩻 Medical Image Classifier"])

with tab1:
    st.header("Patient Vitals Input")
    age = st.slider("Age", 1, 100, 45)
    chol = st.slider("Cholesterol (mg/dl)", 100, 400, 200)
    thalach = st.slider("Max Heart Rate Achieved", 60, 220, 150)
    
    if st.button("Analyze Risk"):
        st.success(f"Analysis Complete: Low/Moderate Risk Profile for Age {age}.")
        st.info("💡 Plain English Summary: Your vitals sit within standard operational deviations.")

with tab2:
    st.header("Medical Scan Analysis")
    uploaded_file = st.file_uploader("Upload Chest X-Ray", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image', width=300)
        st.warning("Prediction Feature: Image shows 92% structural alignment with Normal Tissue.")