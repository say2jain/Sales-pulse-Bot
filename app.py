
import streamlit as st
import pandas as pd
import openai
import matplotlib.pyplot as plt
from gtts import gTTS
import io
import base64
import traceback
from openai import OpenAI

# Init GPT client
client = OpenAI()

st.set_page_config(layout="wide")
st.title("🤖 AI Voice Bot – Continuous Executive Assistant")

# Load Data
st.sidebar.header("📂 Upload Sales Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Using uploaded data")
else:
    df = pd.read_csv("Sales data.csv")
    st.info("📌 Using default sales data")

df.columns = df.columns.str.strip()
if "Booking Date" in df.columns:
    df["Booking Date"] = pd.to_datetime(df["Booking Date"], errors='coerce')

# Chat history
if "history" not in st.session_state:
    st.session_state.history = []

# TTS output
def speak_response(text):
    tts = gTTS(text)
    tts.save("temp.mp3")
    audio_file = open("temp.mp3", "rb")
    st.audio(audio_file.read(), format="audio/mp3")

# Ask GPT
def get_response(question, schema):
    prompt = f"""
You're a data assistant. Based on the schema and a question:
1. Answer in business terms.
2. Then provide matplotlib chart code.

Schema:
{schema}

Question: {question}

Format:
<response>
<code>
```python
# your chart code here
```
"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.header("💬 Ask me anything about the data")
    question = st.text_input("Your question", key="q")

    if question:
        st.session_state.history.append({"role": "user", "content": question})
        schema = df.dtypes.astype(str).to_string()
        try:
            output = get_response(question, schema)
            if "```python" in output:
                response_text, chart_code = output.split("```python")
                chart_code = chart_code.replace("```", "").strip()
            else:
                response_text = output
                chart_code = ""

            st.session_state.history.append({"role": "assistant", "content": response_text.strip()})
            speak_response(response_text.strip())

        except Exception as e:
            st.session_state.history.append({"role": "assistant", "content": "Error getting response."})
            st.error("⚠️ GPT failed: " + str(e))

    # Show conversation history
    for msg in st.session_state.history[::-1]:
        st.markdown(f"**{msg['role'].capitalize()}:** {msg['content']}")

with col2:
    st.header("📊 Smart Visuals")
    if question and 'chart_code' in locals():
        try:
            local_env = {"df": df, "plt": plt, "st": st}
            exec(chart_code, local_env)
        except Exception:
            st.warning("⚠️ GPT chart failed. Here's a fallback:")
            if "Booking Date" in df.columns:
                trend = df.set_index("Booking Date").resample("M")["Net Sale Value (AED)"].sum()
                fig, ax = plt.subplots()
                trend.plot(ax=ax)
                ax.set_title("Fallback: Monthly Sales Trend")
                st.pyplot(fig)
