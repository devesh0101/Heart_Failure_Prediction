import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the scaler

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


## streamlit app
#st.title('Death by Heart Failure Prediction')
st.markdown(
    """
    <div style="text-align: center;">
        <h1>Death by Heart Failure Prediction</h1>
        <a href="https://github.com/devesh0101/Heart_Failure_Prediction" target="_blank">GitHub Link</a>
    </div>
    """, unsafe_allow_html=True
)



# User input
age= st.slider('Age', 1, 100)
anaemia = st.selectbox('Suffering from Anaemia', [0, 1])
creatinine_phosphokinase = st.number_input('Level of the CPK enzyme in the blood (mcg/L)') 
diabetes= st.selectbox('If the patient has diabetes', [0, 1])
ejection_fraction= st.slider('percentage of blood leaving the heart at each contraction', 0, 100) 
high_blood_pressure = st.selectbox('If the patient has hypertension ', [0, 1])
platelets= st.number_input('Platelets in the blood (kiloplatelets/mL)') 
serum_creatinine= st.number_input('Level of serum creatinine in the blood (mg/dL)')  
serum_sodium= st.number_input('level of serum sodium in the blood (mEq/L)')  
sex= st.selectbox(' Woman or Man', [0, 1]) 
smoking= st.selectbox('If the patient smokes or not', [0, 1]) 
time= st.slider('Follow-up period (days)', 1, 365) 


# Prepare the input data


input_data = pd.DataFrame({
    'age': [age],
    'anaemia': [anaemia],
    'creatinine_phosphokinase': [creatinine_phosphokinase],
    'diabetes':[diabetes],
    'ejection_fraction':[ejection_fraction],
    'high_blood_pressure': [high_blood_pressure],
    'platelets': [platelets],
    'serum_creatinine': [serum_creatinine],
    'serum_sodium': [serum_sodium],
    'sex': [sex],
    'smoking': [smoking],
    'time': [time]
})


# Scale the input data
input_data_scaled = scaler.transform(input_data)


# Predict HF
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Death Prediction: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('Most likely, the paient is no longer alive.')
else:
    st.write('Most likely, the paient is alive.')
