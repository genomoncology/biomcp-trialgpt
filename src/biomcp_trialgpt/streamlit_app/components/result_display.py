from typing import List

import streamlit as st


def display_results(results: List[dict], framework: str):
    st.header(f"Results from {framework} agent")
    if not results:
        st.warning("No trials found.")
        return
    for trial in results:
        st.subheader(trial.get("title", "Untitled"))
        st.write(f"NCT ID: {trial.get('nct_id', '')}")
        st.write(f"Score: {trial.get('score', '')}")
        if traceback := trial.get("explanation"):
            with st.expander("Explanation"):
                st.write(traceback)
