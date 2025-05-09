import streamlit as st

def framework_selector():
    st.sidebar.subheader("Agent Framework")
    return st.sidebar.radio("Select Framework", [
        "openai",
        "pydantic",
        "langgraph",
    ])
