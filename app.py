import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit App
st.set_page_config(page_title='Customer Churn Prediction', page_icon='📊', layout='centered')

# Title and Description
st.title('📊 Customer Churn Prediction')
st.markdown(
    """
    This app predicts the likelihood of a customer churning based on their demographic and account information. 
    Provide the necessary details below to get a prediction.
    """
)

# User Input Section
st.header("Enter Customer Details")
with st.expander("ℹ️ Instructions"):
    st.write("Fill in the customer's details below. The app will provide a prediction of whether the customer is likely to churn.")

# Organize Inputs Using Columns
col1, col2 = st.columns(2)

with col1:
    geography = st.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0], help="Select the customer's country/region.")
    gender = st.selectbox('⚤ Gender', label_encoder_gender.classes_, help="Select the customer's gender.")
    age = st.slider('🎂 Age', 18, 92, help="Drag the slider to specify the customer's age.")
    credit_score = st.number_input('💳 Credit Score', min_value=0, max_value=1000, step=1, help="Enter the customer's credit score.")
    tenure = st.slider('📆 Tenure (years)', 0, 10, help="Select the customer's tenure in years.")

with col2:
    balance = st.number_input('💰 Balance', min_value=0.0, help="Enter the customer's account balance.")
    estimated_salary = st.number_input('🤑 Estimated Salary', min_value=0.0, help="Enter the customer's estimated annual salary.")
    num_of_products = st.slider('📦 Number of Products', 1, 4, help="Select the number of products the customer uses.")
    has_cr_card = st.selectbox('💳 Has Credit Card', [0, 1], format_func=lambda x: "Yes" if x else "No", help="Does the customer have a credit card?")
    is_active_member = st.selectbox('✔️ Is Active Member', [0, 1], format_func=lambda x: "Yes" if x else "No", help="Is the customer actively using their account?")

# Prepare Input Data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Prediction Section
st.header("Prediction Results")
if st.button("🔍 Predict"):
    with st.spinner("Analyzing..."):
        prediction = model.predict(input_data_scaled)
        prediction_proba = prediction[0][0]

    # Display Prediction Probability
    st.metric("Churn Probability", f"{prediction_proba:.2%}")

    # Display Result
    if prediction_proba > 0.5:
        st.error('⚠️ The customer is likely to churn.', icon="❌")
    else:
        st.success('✅ The customer is not likely to churn.', icon="✔️")

    # Provide Additional Insights
    st.subheader("What This Means:")
    if prediction_proba > 0.5:
        st.write("The customer shows a high probability of churning. Consider retention strategies like personalized offers or enhanced support.")
    else:
        st.write("The customer is engaged and likely to continue. Maintain the current level of service.")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit")
