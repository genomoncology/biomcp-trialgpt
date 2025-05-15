from typing import List, Optional

import streamlit as st


def framework_selector(available_options: Optional[List[str]] = None):
    """
    Display a radio button selector for the agent framework.
    
    Args:
        available_options: Optional list of available frameworks to show.
                          If None, all frameworks will be shown.
    
    Returns:
        The selected framework as a string.
    """
    st.sidebar.subheader("Agent Framework")
    default_options = [
        "pydantic",
        "langgraph",
    ]

    options = available_options if available_options else default_options

    if not options:
        st.sidebar.error("No frameworks available. Please check your API keys.")
        return None

    return st.sidebar.radio("Select Framework", options)
