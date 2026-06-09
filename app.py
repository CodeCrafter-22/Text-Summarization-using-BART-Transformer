import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="BART Summarizer")

st.title("📝 Text Summarization App")

@st.cache_resource
def load_model():
    return pipeline(
        "summarization",
        model="facebook/bart-large-cnn"
    )

summarizer = load_model()

text = st.text_area("Enter your text")

if st.button("Summarize"):
    if len(text.strip()) == 0:
        st.warning("Please enter text")
    else:
        with st.spinner("Generating summary..."):
            result = summarizer(text, max_length=130, min_length=30, do_sample=False)
            st.success(result[0]["summary_text"])

st.write("App is running...")
