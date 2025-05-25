
import streamlit as st
import pandas as pd
import openai
import matplotlib.pyplot as plt
from gtts import gTTS
import os
import traceback
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase

st.set_page_config(layout="wide")
st.title("🎙️ Executive AI Voice Bot – Real Estate Insights")

openai.api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else "sk-..."

st.sidebar.header("📂 Upload Sales Data")
uploaded_file = st.sidebar.file_uploader("Upload your CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Custom data uploaded")
else:
    df = pd.read_csv("Sales data.csv")
    st.info("Using default data")

df.columns = df.columns.str.strip()
if "Booking Date" in df.columns:
    df["Booking Date"] = pd.to_datetime(df["Booking Date"], errors='coerce')

def speak_text(text):
    tts = gTTS(text)
    tts.save("response.mp3")
    audio_file = open("response.mp3", "rb")
    st.audio(audio_file.read(), format="audio/mp3")

def get_gpt_response(question, context_schema):
    prompt = f"""
You are a senior real estate data analyst bot. Based on the dataset schema and a natural language question, do two things:
1. Return a short and useful business insight.
2. Write a matplotlib chart code to visualize the answer.

Use this response format only:
<response>
<code>
```python
# your code here
```

Schema:
{context_schema}

Question:
{question}
"""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

col1, col2 = st.columns([1, 1])

with col1:
    st.header("🧠 Ask Your Question")
    question = st.text_input("Type your question (e.g., 'Show top projects by sales')")
    st.markdown("🎤 **Voice input coming soon** via browser mic")

    if question:
        st.markdown(f"**You asked:** {question}")
        with st.spinner("Analyzing your data..."):
            context = df.dtypes.astype(str).to_string()
            gpt_output = get_gpt_response(question, context)

        try:
            response_text, code_block = gpt_output.split("```python")
            code_snippet = code_block.replace("```", "").strip()
        except:
            response_text = gpt_output
            code_snippet = ""

        st.success(response_text.strip())
        speak_text(response_text.strip())

with col2:
    st.header("📊 Visual Insight")
    if question and code_snippet:
        try:
            exec_env = {"df": df, "plt": plt, "st": st}
            exec(code_snippet, exec_env)
        except Exception as e:
            st.error("GPT chart code failed. Here's a fallback chart:")
            st.code(traceback.format_exc())
            if "Booking Date" in df.columns:
                fallback = df.set_index("Booking Date").resample("M")["Net Sale Value (AED)"].sum()
                fig, ax = plt.subplots()
                fallback.plot(ax=ax)
                ax.set_title("Fallback: Monthly Sales Trend")
                st.pyplot(fig)
    elif question:
        st.info("Waiting for chart generation...")

st.markdown("---")
st.caption("Powered by GPT-4 | Built for Executive Dashboards")
