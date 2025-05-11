import streamlit as st
import os
import json
from biomcp_trialgpt.streamlit_app.components.input_form import patient_input_form
from biomcp_trialgpt.streamlit_app.components.framework_selector import framework_selector
from typing import Dict, Any, Callable, Optional, List, Tuple
import datetime

# Set page configuration
st.set_page_config(page_title="BioMCP Trial Matching", layout="wide")

# Try to import agent modules, but handle missing environment variables gracefully
try:
    from biomcp_trialgpt.streamlit_app.services.agent_openai import run_openai_agent
    openai_available = True
except ValueError as e:
    if "OPENAI_API_KEY" in str(e):
        openai_available = False
    else:
        raise

try:
    from biomcp_trialgpt.streamlit_app.services.agent_pydantic import run_pydantic_agent
    pydantic_available = True
except ValueError as e:
    if "OPENAI_API_KEY" in str(e) or "ANTHROPIC_API_KEY" in str(e) or "GOOGLE_API_KEY" in str(e):
        pydantic_available = False
    else:
        raise

try:
    from biomcp_trialgpt.streamlit_app.services.agent_langgraph import run_langgraph_agent
    langgraph_available = True
except ValueError as e:
    if "OPENAI_API_KEY" in str(e) or "ANTHROPIC_API_KEY" in str(e) or "GOOGLE_API_KEY" in str(e):
        langgraph_available = False
    else:
        raise

def main():
    st.title("BioMCP TrialGPT Comparative Evaluation")
    st.sidebar.header("Patient-to-Trial Matching")

    # Only show available frameworks for selection
    available_frameworks = []
    if openai_available:
        available_frameworks.append("openai")
    if pydantic_available:
        available_frameworks.append("pydantic")
    if langgraph_available:
        available_frameworks.append("langgraph")

    if not available_frameworks:
        st.error("""
        No agent frameworks are available. Please set the required environment variables:
        - OPENAI_API_KEY
        - ANTHROPIC_API_KEY (for Pydantic and LangGraph)
        - GOOGLE_API_KEY (for Pydantic and LangGraph)
        """)
        return

    presentation, llm_model, recruiting_status, min_date, max_date, phase, submitted = patient_input_form()
    # Always display framework selector
    framework = framework_selector(available_options=available_frameworks)
    if not submitted:
        return

    # Initialize all expanders as closed
    step1_expander = st.expander("Step 1: Clinical Note Extraction", expanded=False)
    step2_expander = st.expander("Step 2: Trial Retrieval", expanded=False)
    step3_expander = st.expander("Step 3: Eligibility Matching", expanded=False)
    step4_expander = st.expander("Step 4: Trial Scoring", expanded=False)
    
    # Create a progress indicator
    progress = st.progress(0)

    # Map the selected framework to the corresponding agent function
    agent_map = {
        "openai": run_openai_agent if openai_available else None,
        "pydantic": run_pydantic_agent if pydantic_available else None,
        "langgraph": run_langgraph_agent if langgraph_available else None
    }

    agent_function = agent_map.get(framework)

    if agent_function is None:
        st.error(f"The {framework} agent is not available. Please check your environment variables.")
        return

    # Run the agent pipeline progressively
    with st.spinner(f"Running {framework} agent..."):
        logs = run_progressive_pipeline(
            agent_function, 
            presentation, 
            llm_model, 
            recruiting_status, 
            min_date, 
            max_date, 
            phase,
            step1_expander,
            step2_expander,
            step3_expander,
            step4_expander,
            progress
        )

    # Display final ranked trials outside the collapsibles
    display_ranked_trials(logs, st.empty())

def run_progressive_pipeline(
    agent_function: Callable,
    presentation: str,
    llm_model: str,
    recruiting_status: str,
    min_date: Any,
    max_date: Any,
    phase: str,
    step1_expander: st.expander,
    step2_expander: st.expander,
    step3_expander: st.expander,
    step4_expander: st.expander,
    progress: st.progress
) -> Dict[str, Any]:
    """
    Run the agent pipeline progressively, updating the UI after each step.
    
    Args:
        agent_function: The agent function to run (openai, pydantic, or langgraph)
        presentation: The patient presentation text
        llm_model: The LLM model to use
        recruiting_status: The trial recruiting status filter
        min_date: The minimum date filter
        max_date: The maximum date filter
        phase: The trial phase filter
        step1_expander: The expander for step 1
        step2_expander: The expander for step 2
        step3_expander: The expander for step 3
        step4_expander: The expander for step 4
        progress: The progress bar
        
    Returns:
        Dictionary containing the results of all steps
    """
    # Run Step 1: Clinical note extraction
    step1 = agent_function(
        presentation, llm_model, recruiting_status, min_date, max_date, phase, 
        step="step1"
    )
    display_step1_results(step1, step1_expander)
    step1_expander.expanded = True
    progress.progress(25)

    # Run Step 2: Trial retrieval
    step2 = agent_function(
        presentation, llm_model, recruiting_status, min_date, max_date, phase,
        step="step2", step1_data=step1
    )
    display_step2_results(step2, step2_expander)
    step2_expander.expanded = True
    progress.progress(50)

    # Run Step 3: Eligibility matching
    step3 = agent_function(
        presentation, llm_model, recruiting_status, min_date, max_date, phase,
        step="step3", step1_data=step1, step2_data=step2
    )
    display_step3_results(step3, step3_expander)
    step3_expander.expanded = True
    progress.progress(75)

    # Run Step 4: Trial scoring & ranking
    step4 = agent_function(
        presentation, llm_model, recruiting_status, min_date, max_date, phase,
        step="step4", step1_data=step1, step2_data=step2, step3_data=step3
    )
    display_step4_results(step4, step4_expander)
    step4_expander.expanded = True
    progress.progress(100)

    # Combine all steps for the complete logs
    return {
        "step1": step1,
        "step2": step2,
        "step3": step3,
        "step4": step4
    }

def display_step1_results(step1: Dict[str, Any], expander: st.expander) -> None:
    """Display the results of step 1 in the provided expander."""
    with expander:
        # Check if step1 is a tuple (from direct parse_clinical_note call)
        if isinstance(step1, tuple) and len(step1) == 3:
            data, prompt, response = step1
            st.markdown("**Prompt**")
            st.code(prompt, language="text")
            st.markdown("**Response**")
            st.code(response, language="text")
            st.markdown("**Parsed JSON**")
            st.json(data)
        # Check if step1 is a dictionary with expected keys
        elif isinstance(step1, dict):
            if "model" in step1:
                st.write("**Model:**", step1.get("model", "Unknown"))
            if "prompt" in step1:
                st.markdown("**Prompt**")
                st.code(step1.get("prompt", ""), language="text")
            if "response" in step1:
                st.markdown("**Response**")
                st.code(step1.get("response", ""), language="text")
            if "data" in step1:
                st.markdown("**Parsed JSON**")
                st.json(step1.get("data", {}))
        else:
            st.error("Unexpected data format for Step 1 results")
            st.json(step1)

def display_step2_results(step2: Dict[str, Any], expander: st.expander) -> None:
    """Display the results of step 2 in the provided expander."""
    with expander:
        if isinstance(step2, dict):
            if "params" in step2:
                st.markdown("**Parameters sent to BioMCP**")
                st.json(step2.get("params", {}))
            if "response" in step2:
                st.markdown("**Raw Response**")
                st.json(step2.get("response", []))
        else:
            st.error("Unexpected data format for Step 2 results")
            st.json(step2)

def display_step3_results(step3: Dict[str, Any], expander: st.expander) -> None:
    """Display the results of step 3 in the provided expander."""
    with expander:
        if isinstance(step3, dict):
            for trial_id, r in step3.items():
                st.markdown(f"**Trial {trial_id}**")
                if isinstance(r, dict) and "inclusion" in r and "exclusion" in r:
                    inc_p, inc_r = r["inclusion"]
                    exc_p, exc_r = r["exclusion"]
                    st.markdown("**Inclusion Prompt**")
                    st.code(inc_p, language="text")
                    st.markdown("**Inclusion Response**")
                    st.code(inc_r, language="json")
                    st.markdown("**Exclusion Prompt**")
                    st.code(exc_p, language="text")
                    st.markdown("**Exclusion Response**")
                    st.code(exc_r, language="json")
                else:
                    st.json(r)
        else:
            st.error("Unexpected data format for Step 3 results")
            st.json(step3)

def display_step4_results(step4: Dict[str, Any], expander: st.expander) -> None:
    """Display the results of step 4 in the provided expander."""
    with expander:
        if isinstance(step4, dict) and "scoring_logs" in step4:
            st.markdown("**Scoring Logs**")
            for trial_id, data in step4["scoring_logs"].items():
                st.markdown(f"**Trial {trial_id}**")
                st.json(data)
        else:
            st.error("Unexpected data format for Step 4 results")
            st.json(step4)

def display_ranked_trials(logs: Dict[str, Any], container: st.empty) -> None:
    """Display the final ranked trials in the provided container."""
    with container.container():
        st.header("Ranked Clinical Trials")
        st.markdown("---")
        if isinstance(logs, dict) and "step4" in logs and isinstance(logs["step4"], dict) and "ranked" in logs["step4"]:
            for i, (tid, d) in enumerate(logs["step4"]["ranked"], start=1):
                eligibility = d.get('eligibility_score_E', 'N/A')
                relevance = d.get('relevance_score_R', 'N/A')

                # Create a card-like display for each trial
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.subheader(f"#{i}: Trial {tid}")
                        if 'relevance_explanation' in d:
                            st.markdown(f"**Relevance:** {d.get('relevance_explanation')}")
                        if 'eligibility_explanation' in d:
                            st.markdown(f"**Eligibility:** {d.get('eligibility_explanation')}")
                    with col2:
                        st.metric("Eligibility", f"{eligibility}")
                        st.metric("Relevance", f"{relevance}")
                    st.markdown("---")
        else:
            st.error("No ranked trials available to display")
            st.json(logs)

if __name__ == "__main__":
    main()
