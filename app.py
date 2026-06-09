import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="BART Text Summarizer")

st.title("📝 Text Summarization using BART")

@st.cache_resource
def load_model():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_model()

text = st.text_area("Enter your text here")

max_len = st.slider("Max Length", 50, 300, 130)
min_len = st.slider("Min Length", 10, 100, 30)

if st.button("Summarize"):
    if text.strip():
        result = summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)
        st.subheader("Summary")
        st.write(result[0]["summary_text"])
    else:
        st.warning("Please enter some text")
