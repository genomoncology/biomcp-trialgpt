import os

import streamlit as st
from openai import OpenAI


# Function to get OpenAI API key from session state with fallback to environment variable
def get_openai_api_key():
    """Get OpenAI API key from session state or environment variable."""
    return st.session_state.get("openai_api_key", "") or os.getenv("OPENAI_API_KEY", "")


def create_chat_completion(api_key=None, **kwargs):
    """
    Wrapper for OpenAI chat completion supporting both legacy and new client.
    Args:
        api_key: Optional API key to use. If not provided, will use the key from session state or env var.
        **kwargs: Arguments to pass to the OpenAI API
    Returns:
        The response from the OpenAI API
    """
    # Get API key, prioritizing the provided key, then session state, then env var
    openai_api_key = api_key or get_openai_api_key()

    if not openai_api_key:
        msg = "OpenAI API key not set. Please provide an API key in the sidebar."
        raise ValueError(msg)

    client = OpenAI(api_key=openai_api_key)
    return client.chat.completions.create(**kwargs)
