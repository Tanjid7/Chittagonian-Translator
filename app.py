%%writefile app.py
import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, MarianMTModel, MarianTokenizer
import torch

# --- Page Config ---
st.set_page_config(
    page_title="Chittagonian AI Translator",
    page_icon="🌊",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for extra design ---
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        color: #1E88E5;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0px;
    }
    .sub-title {
        text-align: center;
        color: #555555;
        margin-bottom: 30px;
    }
    .translation-box {
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
    }
    .bangla-box {
        background-color: #E8F5E9;
        border-left: 5px solid #4CAF50;
    }
    .english-box {
        background-color: #FFF3E0;
        border-left: 5px solid #FF9800;
    }
    .stTextArea textarea {
        font-size: 1.2rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Force CPU to avoid VRAM Out of Memory
device = 'cpu'

@st.cache_resource
def load_models():
    try:
        # 1. Load Chittagonian -> Bangla Model (from Hugging Face!)
        # CHANGE "Tanjid7" if your Hugging Face username is different!
        path_mt5 = 'Tanjid7/chittagonian-translator-mt5' 
        tokenizer_mt5 = AutoTokenizer.from_pretrained(path_mt5)
        model_mt5 = AutoModelForSeq2SeqLM.from_pretrained(path_mt5).to(device)

        # 2. Load Bangla -> English Model (Helsinki)
        path_en = 'Helsinki-NLP/opus-mt-bn-en'
        tokenizer_en = MarianTokenizer.from_pretrained(path_en)
        model_en = MarianMTModel.from_pretrained(path_en).to(device)

        return tokenizer_mt5, model_mt5, tokenizer_en, model_en
    except Exception as e:
        st.error(f'Error loading models: {e}')
        return None, None, None, None

tokenizer_mt5, model_mt5, tokenizer_en, model_en = load_models()

# --- Sidebar ---
with st.sidebar:
    st.markdown("## ⚙️ Settings & Info")
    st.info('''
    **Translation Pipeline:**
    1. Chittagonian → Standard Bangla
    2. Standard Bangla → English
    ''')
    st.markdown('---')
    st.subheader('💡 Try these phrases:')
    st.code('অ্যাঁই ভাত ন হাইয়্যুম')
    st.code('তোঁয়ার নাম কি?')
    st.code('অ্যাঁই তোঁয়ারে ভালোবাসি')


# --- Main UI ---
st.markdown('<p class="main-title">🌊 Chittagonian AI Translator</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Instantly translate regional dialect to Standard Bangla and English.</p>', unsafe_allow_html=True)

# Input Area
st.markdown("#### 🗣️ Enter Chittagonian Text:")
input_text = st.text_area('', placeholder='এখানে চাটগাঁইয়া ভাষা লিখুন...', height=100, label_visibility="collapsed")

# Translation Options Toggle
st.markdown("#### 🛠️ Translation Options")
col_opt1, col_opt2 = st.columns(2)
with col_opt1:
    to_bangla = st.checkbox("🇧🇩 Translate to Standard Bangla", value=True, disabled=True) # Always True
with col_opt2:
    to_english = st.checkbox("🌍 Also Translate to English", value=True)

st.markdown("<br>", unsafe_allow_html=True)

# Buttons
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    translate_btn = st.button('🚀 Translate', use_container_width=True)
with col2:
    clear_btn = st.button('🗑️ Clear', use_container_width=True)

if clear_btn:
    st.rerun()

st.markdown('---')

# --- Translation Logic ---
if translate_btn:
    if input_text.strip() == '':
        st.warning('⚠️ Please enter some text first!')
    elif model_mt5 is not None and model_en is not None:
        with st.spinner('🔄 Analyzing dialect & translating... (Running on CPU)'):

            # Step 1: Chittagonian -> Bangla
            inputs_mt5 = tokenizer_mt5(input_text, return_tensors='pt', max_length=64, truncation=True).to(device)
            outputs_mt5 = model_mt5.generate(
                inputs_mt5.input_ids,
                max_length=64, num_beams=5,
                repetition_penalty=2.5, early_stopping=True
            )
            bangla_translation = tokenizer_mt5.decode(outputs_mt5[0], skip_special_tokens=True)

            # Display Bangla Output
            st.markdown("##### 🇧🇩 Standard Bangla Result:")
            st.markdown(f'<div class="translation-box bangla-box"><span style="font-size: 1.2rem; color: #2E7D32;">{bangla_translation}</span></div>', unsafe_allow_html=True)

            # Step 2: Bangla -> English (If checked by user)
            if to_english:
                inputs_en = tokenizer_en(bangla_translation, return_tensors='pt', max_length=128, truncation=True).to(device)
                outputs_en = model_en.generate(
                    inputs_en.input_ids,
                    max_length=128, num_beams=4, early_stopping=True
                )
                english_translation = tokenizer_en.decode(outputs_en[0], skip_special_tokens=True)

                # Display English Output
                st.markdown("##### 🌍 English Result:")
                st.markdown(f'<div class="translation-box english-box"><span style="font-size: 1.2rem; color: #E65100;">{english_translation}</span></div>', unsafe_allow_html=True)

            st.toast('Translation Complete!', icon='✅')
