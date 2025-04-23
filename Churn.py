import pickle
import streamlit as st
import pandas as pd

# --- Load the trained model ---
try:
    with open('expresso_churn_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file 'expresso_churn_model.pkl' not found. Make sure it's in the same directory as this app.")
    st.stop()

# --- Load feature names ---
try:
    with open('feature_names.pkl', 'rb') as file:
        feature_names = pickle.load(file)
except FileNotFoundError:
    st.error("Feature names file 'feature_names.pkl' not found. Make sure it's in the same directory as this app.")
    st.stop()

# --- Load label encoders ---
try:
    with open('label_encoders.pkl', 'rb') as file:
        label_encoders = pickle.load(file)
except FileNotFoundError:
    st.error("Label encoders file 'label_encoders.pkl' not found. Make sure it's in the same directory as this app.")
    st.stop()

st.title('Expresso Churn Predictor')
st.subheader('Enter customer features to predict churn probability')

# --- Input fields ---
tenure_options = ['K > 24 month', 'I 18-21 month', 'H 15-18 month', 'G 12-15 month',
                  'J 21-24 month', 'F 9-12 month', 'E 6-9 month', 'D 3-6 month',
                  'C 1-3 month', 'B 0-1 month', 'A < 1 month']
tenure = st.selectbox('TENURE', tenure_options)

montant = st.number_input('MONTANT', value=0.0)
frequence_rech = st.number_input('FREQUENCE_RECH', value=0.0)
revenue = st.number_input('REVENUE', value=0.0)
arpu_segment = st.number_input('ARPU_SEGMENT', value=0.0)
frequence = st.number_input('FREQUENCE', value=0.0)
data_volume = st.number_input('DATA_VOLUME', value=0.0)
on_net = st.number_input('ON_NET', value=0.0)
orange = st.number_input('ORANGE', value=0.0)
tigo = st.number_input('TIGO', value=0.0)
regularity = st.number_input('REGULARITY', value=0.0)

top_pack_options = ['No_Top_Pack', 'other', 'Data C', 'All Net 500MB Day']  # Add all values from training
top_pack = st.selectbox('TOP_PACK', top_pack_options)

freq_top_pack = st.number_input('FREQ_TOP_PACK', value=0.0)

region_options = ['Dakar', 'Thiès', 'Saint-Louis']  # Add all regions from training
region = st.selectbox('REGION', region_options)

mrg_options = ['NO', 'YES']
mrg = st.selectbox('MRG', mrg_options)

# --- Predict button ---
if st.button('Predict Churn'):
    input_data = pd.DataFrame({
        'MONTANT': [montant],
        'FREQUENCE_RECH': [frequence_rech],
        'REVENUE': [revenue],
        'ARPU_SEGMENT': [arpu_segment],
        'FREQUENCE': [frequence],
        'DATA_VOLUME': [data_volume],
        'ON_NET': [on_net],
        'ORANGE': [orange],
        'TIGO': [tigo],
        'REGULARITY': [regularity],
        'FREQ_TOP_PACK': [freq_top_pack],
        'TENURE': [tenure],
        'TOP_PACK': [top_pack],
        'REGION': [region],
        'MRG': [mrg],
        'DATA_VOLUME_MISSING': [0],  # Add or adjust based on your training setup
    })

    # --- Encode categorical features using loaded encoders ---
    for col in ['TENURE', 'TOP_PACK', 'REGION', 'MRG']:
        encoder = label_encoders[col]
        input_data[col] = encoder.transform(input_data[col])

    # --- Align column order ---
    try:
        input_data = input_data[feature_names]
    except KeyError as e:
        st.error(f"Input data does not match trained features. Missing: {e}")
        st.stop()

    # --- Make prediction ---
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[:, 1]

    # --- Output ---
    st.subheader('Prediction Result:')
    if prediction[0] == 1:
        st.warning(f'This customer is likely to churn (Probability: {probability[0]:.2f})')
    else:
        st.success(f'This customer is unlikely to churn (Probability: {probability[0]:.2f})')
