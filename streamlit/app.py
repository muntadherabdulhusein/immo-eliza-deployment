import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained XGBoost model
model_path = r"C:\Users\munta\OneDrive\Desktop\The Projects\immo-eliza-deployment\streamlit\xgboost_model.pkl"
xgboost_model = joblib.load(model_path)

# Title and Description
st.title("Property Price Prediction App")
st.write("""
This app predicts property prices using a pre-trained XGBoost model.
Input the property features below to get a prediction.
""")

# Feature Input
st.header("Enter Property Features")

# Replace these feature names with the original ones used during training
original_features = [
    "property_type_HOUSE", "subproperty_type_VILLA",
    "garden_sqm", "fl_garden", "fl_swimming_pool",
    "primary_energy_consumption_sqm", "heating_type_GAS",
    "fl_double_glazing", "cadastral_income"
]

# User input for each feature
input_data = {}
for feature in original_features:
    input_data[feature] = st.number_input(f"Enter value for {feature}:", value=0.0)

# Convert input data to a DataFrame for prediction
input_df = pd.DataFrame([input_data])

# Prediction Button
if st.button("Predict Price"):
    try:
        # Make prediction
        prediction = xgboost_model.predict(input_df)[0]
        st.success(f"The predicted property price is: ${prediction:,.2f}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
