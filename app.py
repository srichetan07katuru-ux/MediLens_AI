import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Set up page configurations
st.set_page_config(page_title="MediLens AI", layout="wide")
st.title("🏥 MediLens AI: Integrated Diagnostic Platform")

# Safe loading wrapper for dependencies
@st.cache_resource
def load_assets():
    try:
        loaded_model = pickle.load(open("model.pkl", "rb"))
        loaded_scaler = pickle.load(open("scaler.pkl", "rb"))
        return loaded_model, loaded_scaler
    except FileNotFoundError:
        return None, None

model, scaler = load_assets()

if model is None or scaler is None:
    st.error("⚠️ System Error: 'model.pkl' or 'scaler.pkl' missing from GitHub root folder.")
    st.info("Please upload both pipeline files to continue pipeline integration.")

tab1, tab2 = st.tabs(["📊 Chronic Risk Predictor", "📷 Medical Image Classifier"])

with tab1:
    st.header("Patient Vitals Input")
    
    # Create two visual columns for clean UI formatting
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", 1, 100, 45)
        sex = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male (1)" if x == 1 else "Female (0)")
        cp = st.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3])
        trestbps = st.slider("Resting Blood Pressure (trestbps) mm Hg", 80, 200, 120)
        chol = st.slider("Serum Cholesterol (chol) mg/dl", 100, 560, 200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", options=[0, 1], format_func=lambda x: "True (1)" if x == 1 else "False (0)")
        restecg = st.selectbox("Resting Electrocardiographic Results (restecg)", options=[0, 1, 2])

    with col2:
        thalach = st.slider("Max Heart Rate Achieved (thalach)", 60, 220, 150)
        exang = st.selectbox("Exercise Induced Angina (exang)", options=[1, 0], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
        oldpeak = st.slider("ST Depression Induced by Exercise (oldpeak)", 0.0, 6.2, 1.0, step=0.1)
        slope = st.selectbox("Slope of Peak Exercise ST Segment (slope)", options=[0, 1, 2])
        ca = st.selectbox("Number of Major Vessels (ca) Colored by Flourosopy", options=[0, 1, 2, 3, 4])
        thal = st.selectbox("Thalassemia (thal)", options=[0, 1, 2, 3])

    if st.button("Analyze Risk"):
        if model is not None and scaler is not None:
            # 1. Arrange features in the EXACT sequential order expected by your model heatmap
            raw_features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
            
            # 2. Scale the input metrics exactly how your training environment did
            scaled_features = scaler.transform(raw_features)
            
            # 3. Generate calculations
            prediction = model.predict(scaled_features)
            
            # 4. Render live stateful metrics
            if prediction[0] == 1:
                st.error(f"⚠️ Analysis Complete: High Risk Profile detected for Age {age}.")
                st.info("💡 Recommendation: Comprehensive cardiovascular diagnostics suggested.")
            else:
                st.success(f"✅ Analysis Complete: Low/Moderate Risk Profile for Age {age}.")
                st.info("💡 Plain English Summary: Your vitals sit within standard operational deviations.")
        else:
            st.error("Cannot perform prediction because model or scaler files are missing.")

with tab2:
    st.header("Medical Scan Analysis")
    uploaded_file = st.file_uploader("Upload Chest X-Ray", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", width=300)
        st.warning("Prediction Feature: Image model deployment logic can be appended here.")
