import streamlit as st

from biomcp_trialgpt.streamlit_app.services.note_extractor import get_api_keys


def patient_input_form():
    st.sidebar.subheader("Patient Presentation")
    presentation = st.sidebar.text_area("Enter patient clinical note:", height=200)

    # Get available API keys
    openai_key, anthropic_key, google_key = get_api_keys()

    # Create a list of available models based on API keys
    available_models = []
    if google_key:
        available_models.append("google-gemini-2.5-pro-preview-03-25")
    if openai_key:
        available_models.append("gpt-o4-mini")
    if anthropic_key:
        available_models.append("anthropic-claude-3-7-sonnet-latest")

    # Display a message if no models are available
    if not available_models:
        st.sidebar.warning("No API keys are set. Please add at least one API key in the API Keys section below.")
        available_models = ["google-gemini-2.5-pro-preview-03-25", "gpt-o4-mini", "anthropic-claude-3-7-sonnet-latest"]

    llm_model = st.sidebar.selectbox("Extraction Model", available_models)

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


def api_keys_form():
    """
    Display and manage API keys in the sidebar.
    This function allows users to view and modify API keys that were initially
    set via environment variables. Any changes made will only persist for the
    current browser session and will be reset when the page is refreshed.
    """
    st.sidebar.subheader("API Keys")

    # Initialize session state for API keys if they don't exist
    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = st.secrets.get("OPENAI_API_KEY", "")
    if "anthropic_api_key" not in st.session_state:
        st.session_state.anthropic_api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    if "gemini_api_key" not in st.session_state:
        st.session_state.gemini_api_key = st.secrets.get("GEMINI_API_KEY", "")

    # Display masked input fields for API keys
    openai_key = st.sidebar.text_input(
        "OpenAI API Key",
        value=st.session_state.openai_api_key,
        type="password",
        help="Enter your OpenAI API key. Changes will only persist for this browser session.",
    )
    anthropic_key = st.sidebar.text_input(
        "Anthropic API Key",
        value=st.session_state.anthropic_api_key,
        type="password",
        help="Enter your Anthropic API key. Changes will only persist for this browser session.",
    )
    gemini_key = st.sidebar.text_input(
        "Google Gemini API Key",
        value=st.session_state.gemini_api_key,
        type="password",
        help="Enter your Google Gemini API key. Changes will only persist for this browser session.",
    )

    # Update session state with new values
    st.session_state.openai_api_key = openai_key
    st.session_state.anthropic_api_key = anthropic_key
    st.session_state.gemini_api_key = gemini_key

    return {"openai_api_key": openai_key, "anthropic_api_key": anthropic_key, "gemini_api_key": gemini_key}
