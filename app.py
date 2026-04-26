import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

st.set_page_config(page_title="Chittagonian to Bangla Translator", page_icon="🌍", layout="centered")

st.title("🌍 Chittagonian to Standard Bangla Translator")
st.markdown("Welcome! Enter text in **Chittagonian** below, and the AI will translate it to **Standard Bangla**.")

@st.cache_resource
def load_model():
    # This automatically downloads your model from Hugging Face!
    model_name = "Tanjid7/chittagonian-translator-mt5"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

with st.spinner("Loading AI Model (This takes a minute on startup)..."):
    tokenizer, model = load_model()

user_input = st.text_area("Enter Chittagonian Text:", height=150, placeholder="e.g., অনারা কেন আছন?")

if st.button("Translate 🚀"):
    if user_input.strip() == "":
        st.warning("Please enter some text to translate.")
    else:
        with st.spinner("Translating..."):
            input_ids = tokenizer(user_input, return_tensors="pt").input_ids
            outputs = model.generate(input_ids, max_length=128, num_beams=5, early_stopping=True)
            translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            st.success("Translation Complete!")
            st.info(f"**Standard Bangla:** {translation}")

st.markdown("---")
st.caption("Built with Fine-Tuned mT5 | Developed for Thesis Project")
