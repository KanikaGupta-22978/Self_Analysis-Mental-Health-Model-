import joblib
import streamlit as st
import pandas as pd

# Load the pipeline
pipeline = joblib.load("mental_health_pipeline.pkl")  # Ensure this file exists in your directory

st.title("Mental Health Condition Prediction")

# Collect user input for all required features
age = st.number_input("Age", min_value=1, max_value=100, value=30)
gender = st.selectbox("Gender", ["male", "female", "other"])
country = st.text_input("Country", "United States")
self_employed = st.selectbox("Are you self-employed?", ["yes", "no"])
family_history = st.selectbox("Do you have a family history of mental illness?", ["yes", "no"])
work_interfere = st.selectbox("How often does mental health interfere with work?", ["never", "rarely", "sometimes", "often"])
no_employees = st.selectbox("Number of employees at your company", ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"])
remote_work = st.selectbox("Do you work remotely?", ["yes", "no"])
tech_company = st.selectbox("Is it a tech company?", ["yes", "no"])
benefits = st.selectbox("Does your employer provide mental health benefits?", ["yes", "no", "don't know"])
care_options = st.selectbox("Are mental health care options provided?", ["yes", "no", "not sure"])
wellness_program = st.selectbox("Is there a wellness program?", ["yes", "no", "don't know"])
seek_help = st.selectbox("Does your employer encourage seeking help for mental health?", ["yes", "no"])
anonymity = st.selectbox("Is anonymity protected?", ["yes", "no", "don't know"])
leave = st.selectbox("How easy is it to take a mental health leave?", ["very easy", "somewhat easy", "somewhat difficult", "very difficult", "don't know"])
mental_health_consequence = st.selectbox("Do you fear negative consequences for mental health leave?", ["yes", "no", "maybe"])
phys_health_consequence = st.selectbox("Do you fear negative consequences for physical health leave?", ["yes", "no", "maybe"])
coworkers = st.selectbox("Would you discuss mental health with coworkers?", ["yes", "no", "some of them"])
supervisor = st.selectbox("Would you discuss mental health with a supervisor?", ["yes", "no", "some of them"])
mental_health_interview = st.selectbox("Would you bring up mental health during an interview?", ["yes", "no"])
phys_health_interview = st.selectbox("Would you bring up physical health during an interview?", ["yes", "no"])
mental_vs_physical = st.selectbox("Is mental health treated the same as physical health?", ["yes", "no", "don't know"])
obs_consequence = st.selectbox("Have you observed negative consequences for mental health issues?", ["yes", "no"])

# Prepare the input data
input_data = pd.DataFrame({
    "Age": [age],
    "Gender": [gender],
    "Country": [country],
    "self_employed": [self_employed],
    "family_history": [family_history],
    "work_interfere": [work_interfere],
    "no_employees": [no_employees],
    "remote_work": [remote_work],
    "tech_company": [tech_company],
    "benefits": [benefits],
    "care_options": [care_options],
    "wellness_program": [wellness_program],
    "seek_help": [seek_help],
    "anonymity": [anonymity],
    "leave": [leave],
    "mental_health_consequence": [mental_health_consequence],
    "phys_health_consequence": [phys_health_consequence],
    "coworkers": [coworkers],
    "supervisor": [supervisor],
    "mental_health_interview": [mental_health_interview],
    "phys_health_interview": [phys_health_interview],
    "mental_vs_physical": [mental_vs_physical],
    "obs_consequence": [obs_consequence]
})

# Prediction
if st.button("Predict"):
    try:
        prediction = pipeline.predict(input_data)[0]
        if prediction == 1:
            st.success("Prediction: Possible mental health condition detected.")
        else:
            st.success("Prediction: No mental health condition detected.")
    except ValueError as e:
        st.error(f"Error in prediction: {e}")
