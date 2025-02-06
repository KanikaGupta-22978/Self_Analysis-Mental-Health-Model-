import joblib
import numpy as np
import pandas as pd
import os 

# Load the pipeline
if not os.path.exists('mental_health_pipeline.pkl'):
    raise FileNotFoundError("mental_health_pipeline.pkl not found. Run model_deployment.py first.")

pipeline = joblib.load('mental_health_pipeline.pkl')

# Example input with all 23 features
symptom_vector = {
    'Age': 30,
    'Gender': 'male',
    'Country': 'United States',
    'self_employed': 'no',
    'family_history': 'yes',
    'work_interfere': 'sometimes',
    'no_employees': '26-100',
    'remote_work': 'no',
    'tech_company': 'yes',
    'benefits': 'yes',
    'care_options': 'not sure',
    'wellness_program': 'no',
    'seek_help': 'no',
    'anonymity': 'no',
    'leave': 'somewhat difficult',
    'mental_health_consequence': 'no',
    'phys_health_consequence': 'no',
    'coworkers': 'yes',
    'supervisor': 'yes',
    'mental_health_interview': 'no',
    'phys_health_interview': 'no',
    'mental_vs_physical': 'yes',
    'obs_consequence': 'no'
}

# Convert the symptom_vector into a DataFrame with one row
input_df = pd.DataFrame([symptom_vector])

# Make prediction
prediction = pipeline.predict(input_df)
print(f"Predicted mental health condition: {prediction[0]}")
