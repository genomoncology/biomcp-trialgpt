import streamlit as st
from biomcp_trialgpt.streamlit_app.components.input_form import patient_input_form
from biomcp_trialgpt.streamlit_app.components.framework_selector import framework_selector
from biomcp_trialgpt.streamlit_app.components.result_display import display_results
from biomcp_trialgpt.streamlit_app.services.biomcp_client import BioMCPClient
from biomcp_trialgpt.streamlit_app.services.agent_openai import run_openai_agent
from biomcp_trialgpt.streamlit_app.services.agent_pydantic import run_pydantic_agent
from biomcp_trialgpt.streamlit_app.services.agent_langgraph import run_langgraph_agent
from biomcp_trialgpt.streamlit_app.services.note_extractor import parse_clinical_note

def main():
    st.title("BioMCP TrialGPT Comparative Evaluation")
    st.sidebar.header("Patient-to-Trial Matching")

    presentation, llm_model, recruiting_status, min_date, max_date, phase, submitted = patient_input_form()
    # Always display framework selector
    framework = framework_selector()
    if not submitted:
        return
    patient_data, prompt, resp_content = parse_clinical_note(presentation, llm_model)
    # Debug: show extraction step details
    with st.expander("Step 1: Clinical Note Extraction"):
        st.write("**Model:**", llm_model)
        st.markdown("**Prompt**")
        st.code(prompt, language="text")
        st.markdown("**Raw Response**")
        st.text(resp_content)
        st.markdown("**Parsed JSON**")
        st.json(patient_data)

    # Step 2: Trial Retrieval via BioMCP
    client = BioMCPClient()
    with st.expander("Step 2: Trial Retrieval via BioMCP"):
        st.markdown("**Parameters sent to BioMCP**")
        params = {
            "conditions": patient_data.get("conditions", []),
            "terms": patient_data.get("terms", []),
            "interventions": patient_data.get("interventions", []),
            "recruiting_status": recruiting_status,
            "min_date": min_date.isoformat() if min_date else None,
            "max_date": max_date.isoformat() if max_date else None,
            "phase": phase,
        }
        st.json(params)
        trials_list = client.retrieve_trials(**params)
        st.markdown("**Raw Response**")
        st.json(trials_list)

    if patient_data and framework:
        with st.spinner(f"Running {framework} agent..."):
            if framework == "openai":
                results = run_openai_agent(
                    client,
                    patient_data,
                    recruiting_status,
                    min_date,
                    max_date,
                    phase,
                )
            elif framework == "pydantic":
                results = run_pydantic_agent(
                    client,
                    patient_data,
                    recruiting_status,
                    min_date,
                    max_date,
                    phase,
                )
            elif framework == "langgraph":
                results = run_langgraph_agent(
                    client,
                    patient_data,
                    recruiting_status,
                    min_date,
                    max_date,
                    phase,
                )
            else:
                st.error("Unknown framework selected")
                return
        display_results(results, framework)

if __name__ == "__main__":
    main()
