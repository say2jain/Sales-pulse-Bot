
import streamlit as st
import pandas as pd
import openai
import matplotlib.pyplot as plt
from gtts import gTTS
import os
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, ClientSettings
import speech_recognition as sr
import tempfile

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("Sales data.csv")
    df["Booking Date"] = pd.to_datetime(df["Booking Date"], errors='coerce')
    df["Net Sale Value (AED)"] = pd.to_numeric(df["Net Sale Value (AED)"], errors='coerce')
    df["Total Saleable Area (Sqft)"] = pd.to_numeric(df["Total Saleable Area (Sqft)"], errors='coerce')
    return df[df["Unit Type"].notna()]  # Filter out missing unit types

df = load_data()

# Text to speech
def speak_text(text):
    tts = gTTS(text)
    tts.save("response.mp3")
    audio_file = open("response.mp3", "rb")
    st.audio(audio_file.read(), format="audio/mp3")

# AI response
def get_ai_response(question, sample_data):
    prompt = f"""
You are an AI sales analyst. Answer the user's question based on the sales data provided. Suggest or describe relevant charts if applicable.

Sample Data:
{sample_data}

Question: {question}
"""
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return completion.choices[0].message.content.strip()

# Intent-based charting
def render_chart(question):
    st.subheader("📊 Suggested Chart")
    if "nationality" in question.lower():
        data = df.groupby("P1 Nationality")["Net Sale Value (AED)"].sum().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots()
        data.plot(kind="bar", ax=ax)
        ax.set_title("Top Nationalities by Sales")
        st.pyplot(fig)

    elif "unit type" in question.lower():
        data = df.groupby("Unit Type")["Net Sale Value (AED)"].mean()
        fig, ax = plt.subplots()
        data.plot(kind="bar", ax=ax)
        ax.set_title("Avg Net Sale Value by Unit Type")
        st.pyplot(fig)

    elif "project" in question.lower() or "master" in question.lower():
        data = df.groupby("Project Name")["Net Sale Value (AED)"].sum().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots()
        data.plot(kind="bar", ax=ax)
        ax.set_title("Top Projects by Total Sales")
        st.pyplot(fig)

    elif "trend" in question.lower() or "monthly" in question.lower():
        trend = df.set_index("Booking Date").resample("M")["Net Sale Value (AED)"].sum()
        fig, ax = plt.subplots()
        trend.plot(ax=ax)
        ax.set_title("Monthly Sales Trend")
        st.pyplot(fig)
    else:
        st.info("No specific chart identified for this query.")

# Streamlit UI
st.set_page_config(layout="wide")
st.title("🎙️ AI Voice Bot – Real Estate Sales Analyst")

# Text input fallback
st.markdown("#### Speak or type your question:")

user_question = st.text_input("Try: 'Show sales by nationality' or 'Compare unit types'")

if user_question:
    st.markdown(f"**You asked:** {user_question}")
    answer = get_ai_response(user_question, df.sample(20).to_string())
    st.success(answer)
    speak_text(answer)
    render_chart(user_question)
