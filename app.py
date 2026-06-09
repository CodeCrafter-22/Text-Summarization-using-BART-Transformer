import streamlit as st
from transformers import pipeline

st.title("Text Summarization App")

@st.cache_resource
def load_model():
    return pipeline(
        "summarization",
        model="facebook/bart-large-cnn"
    )

summarizer = load_model()

text = st.text_area("Enter text")

if st.button("Summarize"):
    if text:
        result = summarizer(text, max_length=130, min_length=30, do_sample=False)
        st.write(result[0]["summary_text"])
    else:
        st.warning("Enter text first")
