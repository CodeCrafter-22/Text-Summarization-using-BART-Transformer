import streamlit as st
from transformers import pipeline

st.title("Text Summarization using BART")

summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)

text = st.text_area("Enter text")

if st.button("Summarize"):
    summary = summarizer(
        text,
        max_length=100,
        min_length=30,
        do_sample=False
    )
    st.write(summary[0]["summary_text"])
