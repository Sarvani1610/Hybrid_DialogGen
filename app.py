"""
Streamlit chat UI for the Hybrid LSTM-Transformer dialogue model.

Run locally:
    streamlit run app.py

Deploy free online: see README.md (Hugging Face Spaces / Streamlit Cloud).
"""

import os
import streamlit as st
import tensorflow as tf

from src import config, data
from src.model import build_model
from src.inference import greedy_decode

st.set_page_config(page_title="Hybrid LSTM-Transformer Chatbot", page_icon="💬")


@st.cache_resource(show_spinner="Loading model...")
def load_model_and_tokenizer():
    if not os.path.exists(config.TOKENIZER_PATH) or not os.path.exists(
        config.MODEL_WEIGHTS_PATH
    ):
        return None, None
    tok = data.load_tokenizer()
    vocab_size = min(config.VOCAB_SIZE, len(tok.word_index) + 1)
    model = build_model(vocab_size)
    model.load_weights(config.MODEL_WEIGHTS_PATH)
    return model, tok


st.title("💬 Hybrid LSTM-Transformer Dialogue Bot")
st.caption(
    "Transformer encoder + LSTM decoder with attention. "
    "Train it with `python -m src.train` before chatting, or run the "
    "quick `--source sample` smoke test."
)

model, tok = load_model_and_tokenizer()

if model is None:
    st.warning(
        "No trained checkpoint found yet.\n\n"
        "Run this first, from the repo root:\n\n"
        "```bash\npython -m src.train --source sample --epochs 5\n```\n\n"
        "(or `--source cornell` once you've downloaded the Cornell Movie-Dialogs "
        "Corpus into `data/` -- see README.md), then reload this page."
    )
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Say something...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = greedy_decode(model, tok, user_input)
            reply = reply if reply else "..."
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})

with st.sidebar:
    st.header("About")
    st.write(
        "Encoder: Transformer self-attention blocks.\n\n"
        "Decoder: Stacked LSTM blocks + Luong attention.\n\n"
        "Trained with masked cross-entropy, evaluated with BLEU."
    )
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()
