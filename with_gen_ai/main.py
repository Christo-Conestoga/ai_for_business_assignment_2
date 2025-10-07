# patient_actor_ui.py

import os
from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st

# ✅ Load environment variables from .env
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("❌ OPENAI_API_KEY not found. Please set it in your .env file.")
    st.stop()

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# System prompt for patient persona
system_prompt = """
You are acting as a patient in a medical training simulation.
Stay consistent with the following case details:

- Complaint: Headache for 2 days
- Location: above right eye
- Severity: sharp, worse in mornings
- Associated symptoms: nausea, light sensitivity

The student is a medical trainee. Respond realistically and conversationally
like a patient would, without giving diagnoses or medical jargon.
"""

# Streamlit Page Config
st.set_page_config(page_title="Patient Actor Chatbot", page_icon="🧠", layout="centered")

st.title("🧠 Patient Actor Chatbot")
st.caption("Simulate realistic patient conversations for medical training.")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": system_prompt}
    ]
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar
with st.sidebar:
    st.header("📝 Scenario")
    st.write("**Case:** Headache for 2 days")
    st.write("**Symptoms:** sharp pain, nausea, light sensitivity")
    if st.button("🔄 Reset Conversation"):
        st.session_state.messages = [{"role": "system", "content": system_prompt}]
        st.session_state.chat_history = []
        st.experimental_rerun()

# Chat container
chat_container = st.container()

# Display previous messages
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        with chat_container.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with chat_container.chat_message("assistant"):
            st.markdown(msg["content"])

# Input box
prompt = st.chat_input("Type your question to the patient...")

# Handle user input
if prompt:
    # Add user message
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate patient response
    with chat_container.chat_message("assistant"):
        with st.spinner("Patient is responding..."):
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=st.session_state.messages
            )
            patient_reply = response.choices[0].message.content.strip()
            st.markdown(patient_reply)

    # Save assistant message
    st.session_state.chat_history.append({"role": "assistant", "content": patient_reply})
    st.session_state.messages.append({"role": "assistant", "content": patient_reply})
