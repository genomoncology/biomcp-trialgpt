import streamlit as st


def patient_input_form():
    st.sidebar.subheader("Patient Presentation")
    presentation = st.sidebar.text_area("Enter patient clinical note:", height=200)
    llm_model = st.sidebar.selectbox(
        "Extraction Model", ["google-gemini-2.5-pro-preview-03-25", "gpt-o4-mini", "anthropic-claude-3-7-sonnet-latest"]
    )
    # Trial filter controls
    recruiting_status = st.sidebar.selectbox(
        "Recruiting Status",
        ["", "OPEN", "CLOSED", "ANY"],
        index=0,
    )
    # Optional date filters
    use_min = st.sidebar.checkbox("Filter by Min Date")
    min_date = st.sidebar.date_input("Min Date") if use_min else None
    use_max = st.sidebar.checkbox("Filter by Max Date")
    max_date = st.sidebar.date_input("Max Date") if use_max else None
    phase = st.sidebar.selectbox(
        "Phase",
        ["", "Phase 1", "Phase 2", "Phase 3", "Phase 4", "N/A"],
        index=0,
    )
    submitted = st.sidebar.button("Submit")
    return presentation, llm_model, recruiting_status, min_date, max_date, phase, submitted
