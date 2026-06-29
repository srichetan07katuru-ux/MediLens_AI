import streamlit as st
import pandas as pd
import numpy as np
import pickle  # <-- 1. Add pickle to load your model

# <-- 2. Load your model file (make sure model.pkl is uploaded to your GitHub)
@st.cache_resource
def load_model():
    return pickle.load(open("model.pkl", "rb"))

try:
    model = load_model()
except Exception as e:
    st.error("Model file 'model.pkl' not found in repository!")

st.set_page_config(page_title="MediLens AI", layout="wide")
st.title("🏥 MediLens AI: Integrated Diagnostic Platform")

tab1, tab2 = st.tabs(["📊 Chronic Risk Predictor", "📷 Medical Image Classifier"])

with tab1:
    st.header("Patient Vitals Input")
    age = st.slider("Age", 1, 100, 45)
    chol = st.slider("Cholesterol (mg/dl)", 100, 400, 200)
    thalach = st.slider("Max Heart Rate Achieved", 60, 220, 150)
    
    if st.button("Analyze Risk"):
        # <-- 3. Create the input feature array matching your model's expected columns
        input_data = np.array([[age, chol, thalach]])
        
        # <-- 4. Get the actual prediction from the model
        prediction = model.predict(input_data)
        
        # <-- 5. Make the UI output dynamic based on the prediction
        if prediction[0] == 1:
            st.error(f"⚠️ Analysis Complete: High Risk Profile detected for Age {age}.")
            st.info("💡 Recommendation: Consult a medical specialist for comprehensive evaluation.")
        else:
            st.success(f"✅ Analysis Complete: Low/Moderate Risk Profile for Age {age}.")
            st.info("💡 Plain English Summary: Your vitals sit within standard operational deviations.")