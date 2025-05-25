
import streamlit as st
import pandas as pd
import openai
import speech_recognition as sr
from gtts import gTTS
import os
import matplotlib.pyplot as plt

# Set OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else "sk-..."

# Load and clean data
@st.cache_data
def load_data():
    df = pd.read_csv("Sales data.csv")
    df["Booking Date"] = pd.to_datetime(df["Booking Date"], errors='coerce')
    df["Net Sale Value (AED)"] = pd.to_numeric(df["Net Sale Value (AED)"], errors='coerce')
    df["Total Saleable Area (Sqft)"] = pd.to_numeric(df["Total Saleable Area (Sqft)"], errors='coerce')
    return df

df = load_data()

# Voice input function
def transcribe_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Please speak.")
        audio = r.listen(source)
    try:
        return r.recognize_google(audio)
    except sr.UnknownValueError:
        return "Sorry, I could not understand your voice."
    except sr.RequestError as e:
        return f"Could not request results; {e}"

# Ask GPT with structured context
def ask_ai(question, context):
    prompt = f"""
You are a smart real estate data assistant. You are analyzing structured sales data.

Field meanings:
- Master Development: Top-level location hierarchy.
- Project Name: Subdivision of master plan.
- Unit Type: Property type (e.g. 2 BR, 3 BR). Use this, not ARC Unit Type.
- Total Saleable Area (Sqft): Area of unit sold.
- Net Sale Value (AED): Final sale price.
- Deal Type: e.g., Direct Sales or Client with Broker.

Use this sample data:
{context}

Answer this question in a detailed yet clear manner:
{question}
"""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# Text to speech
def speak(text):
    tts = gTTS(text)
    tts.save("response.mp3")
    os.system("start response.mp3" if os.name == "nt" else "afplay response.mp3")

# Charts
def show_charts(df):
    st.subheader("📊 Smart Visualizations")

    # Unit Type Distribution
    st.markdown("**Unit Type Distribution**")
    unit_type_counts = df["Unit Type"].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(unit_type_counts, labels=unit_type_counts.index, autopct="%1.1f%%")
    ax1.axis("equal")
    st.pyplot(fig1)

    # Nationality Sales
    st.markdown("**Top Nationalities by Net Sales**")
    top_nat = df.groupby("P1 Nationality")["Net Sale Value (AED)"].sum().sort_values(ascending=False).head(10)
    fig2, ax2 = plt.subplots()
    top_nat.plot(kind="bar", ax=ax2)
    st.pyplot(fig2)

    # Sale Trend
    st.markdown("**Monthly Sales Trend**")
    trend = df.set_index("Booking Date").resample("M")["Net Sale Value (AED)"].sum()
    fig3, ax3 = plt.subplots()
    trend.plot(ax=ax3)
    st.pyplot(fig3)

# Streamlit UI
st.set_page_config(layout="wide")
st.title("🏠 Real Estate Voice AI Bot")

col1, col2 = st.columns(2)

with col1:
    st.header("🎙️ Talk to the Data")
    if st.button("Ask via Voice"):
        user_query = transcribe_audio()
        st.write(f"**You asked:** {user_query}")
        context_data = df.sample(20).to_string()
        answer = ask_ai(user_query, context_data)
        st.success(answer)
        speak(answer)

with col2:
    show_charts(df)
