import streamlit as st
import numpy as np

# 1. Load the "Brain"
model_data = np.load('stroke_model_v1.npz', allow_pickle=True)
w = model_data['weights']
b = model_data['bias']
mean = model_data['mean']
std = model_data['std']
feature_names = model_data['feature_names'].tolist()
feature_to_scale = ['age', 'avg_glucose_level', "bmi", "age_glucose", 'age_s']

st.title("🧠 Stroke Risk Predictor")
st.write("Adjust the sliders to see the stroke probability.")

# 2. Sidebar Sliders
age = st.sidebar.slider("Age", 0, 100, 50)
bmi = st.sidebar.slider("BMI", 10.0, 60.0, 25.0)
glucose = st.sidebar.slider("Avg Glucose Level", 50.0, 300.0, 100.0)
married = st.sidebar.selectbox("Ever Married?", ["No", "Yes"])
smoking = st.sidebar.selectbox("Smoking Status", ["Never Smoked", "Formerly Smoked", "Smokes", "Unknown"])

# 3. Prepare Input
patient_dict = {
    'age': age,
    'avg_glucose_level': glucose,
    'bmi': bmi,
    'age_glucose': age * glucose,
    'age_s': age ** 2,
    'ever_married': 1 if married == "Yes" else 0,
    'smoking_status_never smoked': 1 if smoking == "Never Smoked" else 0,
    'smoking_status_formerly smoked': 1 if smoking == "Formerly Smoked" else 0,
    'smoking_status_smokes': 1 if smoking == "Smokes" else 0,
}

# 4. Predict
if st.button("Predict"):
    x_input = np.array([patient_dict.get(name, 0) for name in feature_names], dtype=float)

    for i, name in enumerate(feature_names):
        if name in feature_to_scale:
            scale_idx = feature_to_scale.index(name)
            x_input[i] = (x_input[i] - mean[scale_idx]) / std[scale_idx]

    z = np.dot(x_input, w) + b
    # Added np.clip to prevent the "fried" exp error you saw in the logs
    z = np.clip(z, -500, 500)
    prob = 1 / (1 + np.exp(-z))

    st.subheader(f"Stroke Probability: {prob:.2%}")
    if prob >= 0.3:
        st.error("⚠️ HIGH RISK")
    else:
        st.success("✅ LOW RISK")
