# -*- coding: utf-8 -*-

!pip install -U crewai litellm pandas streamlit pyngrok

import os
import pandas as pd
from getpass import getpass
from crewai import Agent, Task, Crew, Process, LLM
import streamlit as st

os.environ["GROQ_API_KEY"] = getpass("Enter your Groq API Key: ")

llm = LLM(model="groq/llama-3.1-8b-instant",
    temperature=0.1)

collector = Agent(
    role="Patient Collector",
    goal="Organize patient vitals",
    backstory="Expert healthcare data assistant",
    llm=llm,
    verbose=False
)

doctor = Agent(
    role="Doctor AI",
    goal="Analyze patient data and identify health risks",
    backstory="Clinical AI assistant",
    llm=llm,
    verbose=False
)

def analyze_patients(dataframe):
    all_results = []
    for index, row in dataframe.iterrows():
        # Create a detailed string from the current patient's data
        patient_info = ", ".join([f"{col}: {row[col]}" for col in dataframe.columns])

        # Define the task for the collector agent using the current patient's data
        collect_vitals_task = Task(
            description=f"Collect patient vitals and health information from the following input data: {patient_info}. Extract all relevant details such as age, gender, blood pressure, heart rate, temperature, medical history, medications, allergies, chief complaint, social history, height, weight, BMI. Prioritize accuracy and completeness based on the provided row data.",
            agent=collector,
            expected_output="A structured list of collected patient vitals and health information, clearly outlining all details extracted from the input data. Example: 'Patient Information: Age: 35, Gender: Male, Vitals: BP: 120/80, HR: 72, Temp: 98.6F, Medical History: Hypertension, Allergies: Penicillin, etc.'"
        )

        # Define the task for the doctor agent using the output of the collector
        analyze_health_task = Task(
            description="Analyze the collected patient vitals and health information to identify potential health risks and suggest preliminary advice. Formulate a comprehensive report based *only* on the provided vitals and information. Do not invent new data.",
            agent=doctor,
            expected_output="A detailed report for this specific patient, listing identified health risks, the reasoning behind each risk, and concrete preliminary advice and recommendations."
        )

        # Create and kickoff a new crew for each patient
        patient_crew = Crew(
            agents=[collector, doctor],
            tasks=[collect_vitals_task, analyze_health_task],
            process=Process.sequential
        )
        patient_analysis_result = patient_crew.kickoff()
        all_results.append(patient_analysis_result)

    return all_results

st.title("Health Risk Identifier App")

uploaded_file = st.file_uploader("Upload Health Risk Dataset", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file) # Use uploaded_file directly for pandas
    st.write("Patient Data:")
    st.dataframe(df)

    st.write("Analyzing health risks...")
    result = analyze_patients(df)

    st.subheader("Analysis Results:")
    for i, res in enumerate(result):
        st.write(f"--- Patient {i+1} ---")
        st.write(res)
    st.write("Health Risk Results:")
    st.json(result)

from pyngrok import ngrok
import os

ngrok.kill()

port = 8501

NGROK_AUTH_TOKEN = getpass("Enter your ngrok Auth Token: ")
ngrok.set_auth_token(NGROK_AUTH_TOKEN)
public_url = ngrok.connect(port)
print("Streamlit app URL:", public_url)