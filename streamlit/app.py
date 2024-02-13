import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load('knn_model.pkl')

# Urutan fitur yang diharapkan
features_order = [
    'Marital status', 'Application mode', 'Application order', 'Course',
    'Daytime/evening attendance', 'Previous qualification', "Mother's occupation",
    "Father's occupation", 'Displaced', 'Debtor', 'Tuition fees up to date', 'gender',
    'Scholarship holder', 'Age', 'Curricular units 1st sem (credited)',
    'Curricular units 1st sem (enrolled)', 'Curricular units 1st sem (evaluations)',
    'Curricular units 1st sem (approved)', 'Curricular units 1st sem (grade)',
    'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)',
    'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (grade)', 'Curricular units 2nd sem (without evaluations)', 'GDP'
]

# Streamlit UI
st.title('Student Status Prediction')

# Input form
st.write('Enter student features:')
input_features = []

for feature in features_order:
    if feature == "Curricular units 1st sem (grade)" or feature == 'GDP' or feature == "Curricular units 2nd sem (grade)":
        input_feature = st.number_input(feature, value=0.0)
    else:
        input_feature = st.number_input(feature, value=0)
    input_features.append(input_feature)

# Perform prediction when button is clicked
if st.button('Predict'):
    # Convert input features into numpy array
    input_features_array = np.array(input_features).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(input_features_array)
    
    # Display prediction result
    reverse_mapping = {0: 'Dropout', 1: 'Enrolled', 2: 'Graduate'}
    predicted_label = reverse_mapping[prediction[0]]
    
    st.write('Predicted student status:', predicted_label)
