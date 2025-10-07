# uncomment this code to download the model
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import os

# MODEL_NAME = "microsoft/phi-2"
# SAVE_PATH = "./models/phi-2"

# os.makedirs(SAVE_PATH, exist_ok=True)
# print(f"🔽 Downloading {MODEL_NAME}...")

# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# tokenizer.save_pretrained(SAVE_PATH)
# model.save_pretrained(SAVE_PATH)

# print(f"Model saved to {SAVE_PATH}")



import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os


MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "phi-2")


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, local_files_only=True, torch_dtype=torch.float32)


SYSTEM_PROMPT = """
You are acting as a patient in a medical training simulation.
Stay consistent with the following case details:

- Complaint: Headache for 2 days
- Location: above right eye
- Severity: sharp, worse in mornings
- Associated symptoms: nausea, light sensitivity

The student is a medical trainee. Respond realistically and conversationally
like a patient would, without giving diagnoses or medical jargon.
"""


st.set_page_config(page_title="Patient Actor Chatbot", page_icon="🧠")
st.title("🧠 Patient Actor Chatbot (Local Model)")
st.caption(f"Model loaded from `{MODEL_PATH}`")


if "history" not in st.session_state:
    st.session_state.history = [{"role": "system", "content": SYSTEM_PROMPT}]


for msg in st.session_state.history[1:]:
    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
        st.markdown(msg["content"])


prompt = st.chat_input("Ask the patient something...")
if prompt:
    st.session_state.history.append({"role": "user", "content": prompt})

  
    input_text = "\n".join(
        [f"{m['role'].upper()}: {m['content']}" for m in st.session_state.history]
    )

    inputs = tokenizer(input_text, return_tensors="pt")

    with st.spinner("Patient is thinking..."):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=120,
                do_sample=True,
                top_p=0.9,
                temperature=0.8
            )

    raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "ASSISTANT:" in raw_output:
        reply = raw_output.split("ASSISTANT:")[-1].strip()
    else:
        reply = raw_output[len(input_text):].strip()

    st.session_state.history.append({"role": "assistant", "content": reply})

    with st.chat_message("assistant"):
        st.markdown(reply)
