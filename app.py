
import streamlit as st
import pandas as pd
import openai
import matplotlib.pyplot as plt
import io
from gtts import gTTS
import os
import traceback

# Set up page
st.set_page_config(layout="wide")
st.title("🤖 AI Voice Bot – Dynamic Real Estate Analyst")

# Load OpenAI Key
openai.api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else "sk-..."

# Upload data
st.sidebar.header("📁 Upload Sales Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully.")
else:
    df = pd.read_csv("Sales data.csv")
    st.info("📌 Using default sales data.")

# Clean and prep data
df.columns = df.columns.str.strip()
if "Booking Date" in df.columns:
    df["Booking Date"] = pd.to_datetime(df["Booking Date"], errors='coerce')

st.sidebar.markdown("### Sample of Loaded Data")
st.sidebar.dataframe(df.head(5))

# Text-to-speech
def speak_text(text):
    tts = gTTS(text)
    tts.save("response.mp3")
    audio_file = open("response.mp3", "rb")
    st.audio(audio_file.read(), format="audio/mp3")

# GPT function for both answering and generating chart code
def get_gpt_response(question, context_schema):
    prompt = f"""
You are a data analyst AI. Given a question and the structure of a dataset, do two things:
1. Provide a brief natural language answer or insight.
2. Generate a Python matplotlib chart code snippet (within a function) based on the question.

Respond in this format strictly:
<response>
<code>
```python
# your code here
```

Here is the dataset schema:
{context_schema}

Question: {question}
"""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# UI for asking questions
st.markdown("## 🎙️ Ask Your Question About the Data")
user_input = st.text_input("E.g., What are top nationalities by net sale value?")

if user_input:
    st.markdown(f"**You asked:** {user_input}")
    # Schema sample
    context = df.dtypes.astype(str).to_string()
    with st.spinner("🧠 Thinking..."):
        gpt_output = get_gpt_response(user_input, context)

    # Split response and code
    try:
        response_part, code_part = gpt_output.split("```python")
        chart_code = code_part.replace("```", "").strip()
    except:
        st.error("Could not parse the GPT response. Here's what it returned:")
        st.code(gpt_output)
        response_part = gpt_output
        chart_code = ""

    # Show natural language answer
    st.success(response_part.strip())
    speak_text(response_part.strip())

    # Execute chart code
    if chart_code:
        st.markdown("### 📊 AI-Generated Visualization")
        try:
            local_env = {"df": df, "plt": plt, "st": st}
            exec(chart_code, local_env)
        except Exception as e:
            st.error("⚠️ Error running AI-generated chart code:")
            st.code(traceback.format_exc())
