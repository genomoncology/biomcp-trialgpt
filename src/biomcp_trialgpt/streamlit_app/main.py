import time
import traceback
from typing import Dict, Any

import streamlit as st

from biomcp_trialgpt.streamlit_app.components import input_form
from biomcp_trialgpt.streamlit_app.components.framework_selector import framework_selector

# Set page configuration
st.set_page_config(page_title="BioMCP Trial Matching", layout="wide")

# Import available agent frameworks
try:
    from biomcp_trialgpt.streamlit_app.services.agent_pydantic import run_pydantic_agent

    pydantic_available = True
except ValueError as e:
    if "OPENAI_API_KEY" in str(e) or "ANTHROPIC_API_KEY" in str(e) or "GEMINI_API_KEY" in str(e):
        pydantic_available = False
    else:
        raise

try:
    from biomcp_trialgpt.streamlit_app.services.agent_langgraph import run_langgraph_agent

    langgraph_available = True
except ValueError as e:
    if "OPENAI_API_KEY" in str(e) or "ANTHROPIC_API_KEY" in str(e) or "GEMINI_API_KEY" in str(e):
        langgraph_available = False
    else:
        raise


def init_session_state():
    """Initialize all session state variables."""
    if 'workflow_complete' not in st.session_state:
        st.session_state.workflow_complete = False
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'run_id' not in st.session_state:
        st.session_state.run_id = None
    if 'is_running' not in st.session_state:
        st.session_state.is_running = False
    if 'submitted' not in st.session_state:
        st.session_state.submitted = False
    if 'runtime_id' not in st.session_state:
        st.session_state.runtime_id = time.time()

    # Explicitly set expander states
    st.session_state.step1_expanded = False
    st.session_state.step2_expanded = False
    st.session_state.step3_expanded = False
    st.session_state.step4_expanded = False


def reset_workflow():
    """Reset the workflow state."""
    print("Resetting workflow state")
    st.session_state.workflow_complete = False
    st.session_state.results = None
    st.session_state.raw_results = None
    st.session_state.is_running = False
    st.session_state.submitted = False
    st.session_state.runtime_id = time.time()

    # Reset all expander states
    st.session_state.step1_expanded = False
    st.session_state.step2_expanded = False
    st.session_state.step3_expanded = False
    st.session_state.step4_expanded = False


def print_output(title, data):
    """Print formatted output to the console for debugging."""
    print(f"\n===== {title} =====")
    if isinstance(data, dict):
        print(f"Keys: {list(data.keys())}")
        for key, value in data.items():
            print(f"  {key}: {type(value)}")
            if isinstance(value, dict):
                print(f"    Sub-keys: {list(value.keys())}")
    else:
        print(f"Type: {type(data)}")
        print(f"Value: {data}")
    print("=" * (len(title) + 14))


def run_workflow(agent_function, presentation, llm_model, recruiting_status, min_date, max_date, phase):
    """Run the selected agent workflow and handle results."""
    try:
        print(f"Starting workflow... (Runtime ID: {st.session_state.runtime_id})")
        st.session_state.is_running = True
        st.info("Starting workflow - this may take a few minutes...")

        with st.spinner("Running full workflow pipeline..."):
            # Run the agent with all steps
            results = agent_function(
                presentation, llm_model, recruiting_status, min_date, max_date, phase)

            print("Workflow completed, results received:")
            print_output("Results", results)

            # Debug JSON serialization
            try:
                import json
                json_str = json.dumps(results, default=str)
                print(f"JSON serialization successful: {len(json_str)} characters")
            except Exception as e:
                print(f"JSON serialization failed: {str(e)}")
                # Try to identify problematic fields
                for key, value in results.items():
                    try:
                        json.dumps({key: value}, default=str)
                    except Exception as e:
                        print(f"Problem with key {key}: {str(e)}")

                # Try to create a clean version of the results
                clean_results = {}
                for key, value in results.items():
                    if value is None:
                        clean_results[key] = None
                    elif isinstance(value, dict):
                        try:
                            # Try to convert to JSON and back
                            clean_results[key] = json.loads(json.dumps(value, default=str))
                        except:
                            # Fall back to string representation
                            clean_results[key] = str(value)
                    else:
                        clean_results[key] = str(value)

                # Replace original results with clean version
                results = clean_results
                print("Created clean results for JSON serialization:")
                print_output("Clean Results", results)

            # Store results and update workflow state
            st.session_state.results = results
            st.session_state.workflow_complete = True

            # Expanders should initially be closed
            st.session_state.step1_expanded = False
            st.session_state.step2_expanded = False
            st.session_state.step3_expanded = False
            st.session_state.step4_expanded = False

            st.success("Workflow completed successfully!")
            st.session_state.is_running = False
            st.session_state.submitted = False  # Reset submitted to prevent re-running

            print(f"Results stored in session state. Workflow complete: {st.session_state.workflow_complete}")

            return results

    except Exception as e:
        print(f"Error in run_workflow: {str(e)}")
        traceback.print_exc()
        st.error(f"Error running workflow: {str(e)}")
        st.session_state.is_running = False
        st.session_state.submitted = False
        return None


def display_step1_results(step1: Dict[str, Any], expander: st.expander) -> None:
    """Display the results of step 1 in the provided expander."""
    with expander:
        if step1 is None:
            st.warning("No data available for Step 1")
            return

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
                try:
                    st.json(step1.get("data", {}))
                except:
                    st.write(step1.get("data", {}))
        else:
            st.error("Unexpected data format for Step 1 results")
            st.write(step1)


def display_step2_results(step2: Dict[str, Any], expander: st.expander) -> None:
    """Display the results of step 2 in the provided expander."""
    with expander:
        if step2 is None:
            st.warning("No data available for Step 2")
            return

        if isinstance(step2, dict):
            if "params" in step2:
                st.markdown("**Parameters sent to BioMCP**")
                try:
                    st.json(step2.get("params", {}))
                except:
                    st.write(step2.get("params", {}))
            if "response" in step2:
                st.markdown("**Raw Response**")
                # Show only the first few trials if there are many
                response_data = step2.get("response", [])
                if isinstance(response_data, list) and len(response_data) > 3:
                    try:
                        st.json(response_data[:3])
                    except:
                        st.write(response_data[:3])
                    st.info(f"Showing first 3 of {len(response_data)} trials.")
                else:
                    try:
                        st.json(response_data)
                    except:
                        st.write(response_data)
        else:
            st.error("Unexpected data format for Step 2 results")
            st.write(step2)


def display_step3_results(step3: Dict[str, Any], expander: st.expander) -> None:
    """Display the results of step 3 in the provided expander."""
    with expander:
        if step3 is None:
            st.warning("No data available for Step 3")
            return

        # Try to handle different result formats
        if isinstance(step3, dict):
            # Check if it's a dictionary of trial IDs to eligibility data
            if not any(key in ["results", "error"] for key in step3.keys()):
                # Loop through each trial's eligibility results
                for trial_id, r in step3.items():
                    st.markdown(f"**Trial {trial_id}**")
                    if isinstance(r, dict) and "inclusion" in r and "exclusion" in r:
                        # Extract inclusion and exclusion data
                        inc = r["inclusion"]
                        exc = r["exclusion"]

                        # Handle both tuple and non-tuple formats
                        if isinstance(inc, tuple) and len(inc) == 2:
                            inc_p, inc_r = inc
                        else:
                            inc_p, inc_r = "", str(inc)

                        if isinstance(exc, tuple) and len(exc) == 2:
                            exc_p, exc_r = exc
                        else:
                            exc_p, exc_r = "", str(exc)

                        # Display the information
                        st.markdown("**Inclusion Prompt**")
                        st.code(inc_p, language="text")
                        st.markdown("**Inclusion Response**")
                        st.code(inc_r, language="json")
                        st.markdown("**Exclusion Prompt**")
                        st.code(exc_p, language="text")
                        st.markdown("**Exclusion Response**")
                        st.code(exc_r, language="json")
                    else:
                        try:
                            st.json(r)
                        except:
                            st.write(r)
            elif "results" in step3:
                # Handle the case where there's a 'results' key
                results = step3["results"]
                if isinstance(results, list):
                    for item in results:
                        if isinstance(item, dict) and "trial_id" in item:
                            trial_id = item["trial_id"]
                            eligibility = item.get("eligibility", {})

                            st.markdown(f"**Trial {trial_id}**")
                            if isinstance(eligibility, dict):
                                inc = eligibility.get("inclusion", ("", ""))
                                exc = eligibility.get("exclusion", ("", ""))

                                # Handle both tuple and non-tuple formats
                                if isinstance(inc, tuple) and len(inc) == 2:
                                    inc_p, inc_r = inc
                                else:
                                    inc_p, inc_r = "", str(inc)

                                if isinstance(exc, tuple) and len(exc) == 2:
                                    exc_p, exc_r = exc
                                else:
                                    exc_p, exc_r = "", str(exc)

                                # Display the information
                                st.markdown("**Inclusion Prompt**")
                                st.code(inc_p, language="text")
                                st.markdown("**Inclusion Response**")
                                st.code(inc_r, language="json")
                                st.markdown("**Exclusion Prompt**")
                                st.code(exc_p, language="text")
                                st.markdown("**Exclusion Response**")
                                st.code(exc_r, language="json")
                            else:
                                try:
                                    st.json(eligibility)
                                except:
                                    st.write(eligibility)
                else:
                    st.write("Unexpected format for eligibility results:", type(results))
                    st.write(results)
            elif "error" in step3:
                st.error(f"Error in Step 3: {step3['error']}")
        else:
            st.error("Unexpected data format for Step 3 results")
            st.write(step3)


def display_step4_results(step4: Dict[str, Any], expander: st.expander) -> None:
    """Display the results of step 4 in the provided expander."""
    with expander:
        if step4 is None:
            st.warning("No data available for Step 4")
            return

        if isinstance(step4, dict):
            if "scoring_logs" in step4:
                st.markdown("**Scoring Logs**")
                scoring_logs = step4["scoring_logs"]

                if isinstance(scoring_logs, dict):
                    # Dictionary mapping trial IDs to scoring data
                    for trial_id, data in scoring_logs.items():
                        st.markdown(f"**Trial {trial_id}**")
                        try:
                            st.json(data)
                        except:
                            st.write(data)
                elif isinstance(scoring_logs, list):
                    # List of scoring objects
                    for item in scoring_logs:
                        if isinstance(item, dict) and "trial_id" in item:
                            trial_id = item["trial_id"]
                            scoring = item.get("scoring", {})

                            st.markdown(f"**Trial {trial_id}**")
                            try:
                                st.json(scoring)
                            except:
                                st.write(scoring)
                else:
                    st.write("Unexpected format for scoring logs:", type(scoring_logs))
                    st.write(scoring_logs)
            else:
                st.json(step4)
        else:
            st.error("Unexpected data format for Step 4 results")
            st.write(step4)


def display_ranked_trials(results) -> None:
    """Display the final ranked trials."""
    st.header("Ranked Clinical Trials")
    st.markdown("---")

    if not results:
        st.error("No results available")
        return

    # Get step4 data safely
    step4 = results.get("step4", {}) if isinstance(results, dict) else {}
    if not step4:
        st.warning("No ranking information available")
        return

    # Check for ranked data in different formats
    ranked_trials = []

    # Try to get ranked data
    if isinstance(step4, dict):
        if "ranked" in step4 and step4["ranked"]:
            ranked_trials = step4["ranked"]
        elif "scoring_logs" in step4 and step4["scoring_logs"]:
            # Convert scoring_logs to ranked format
            scoring_logs = step4["scoring_logs"]
            if isinstance(scoring_logs, dict):
                # Sort by eligibility score
                ranked_items = sorted(
                    scoring_logs.items(),
                    key=lambda x: x[1].get('eligibility_score_E', 0) if isinstance(x[1], dict) else 0,
                    reverse=True
                )
                ranked_trials = ranked_items

    if not ranked_trials:
        st.warning("No ranked trials available")
        return

    # Display ranked trials
    for i, item in enumerate(ranked_trials, start=1):
        # Handle different formats of the ranked trial data
        if isinstance(item, tuple) and len(item) == 2:
            trial_id, data = item
        elif isinstance(item, dict) and "trial_id" in item:
            trial_id = item["trial_id"]
            data = {k: v for k, v in item.items() if k != "trial_id"}
        else:
            st.warning(f"Trial #{i} has unknown format: {type(item)}")
            st.write(item)
            continue

        # Display trial information
        with st.container():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.subheader(f"#{i}: Trial {trial_id}")
                if isinstance(data, dict):
                    if 'relevance_explanation' in data:
                        st.markdown(f"**Relevance:** {data.get('relevance_explanation')}")
                    if 'eligibility_explanation' in data:
                        st.markdown(f"**Eligibility:** {data.get('eligibility_explanation')}")
            with col2:
                eligibility = data.get('eligibility_score_E', 'N/A') if isinstance(data, dict) else 'N/A'
                relevance = data.get('relevance_score_R', 'N/A') if isinstance(data, dict) else 'N/A'
                st.metric("Eligibility", f"{eligibility}")
                st.metric("Relevance", f"{relevance}")
            st.markdown("---")


def main():
    st.title("BioMCP TrialGPT Comparative Evaluation")
    st.sidebar.header("Patient-to-Trial Matching")

    # Debug info
    st.sidebar.write(f"Runtime ID: {st.session_state.runtime_id if 'runtime_id' in st.session_state else 'Not set'}")

    # Initialize session state variables
    init_session_state()

    # Determine available frameworks
    available_frameworks = []
    if pydantic_available:
        available_frameworks.append("pydantic")
    if langgraph_available:
        available_frameworks.append("langgraph")

    if not available_frameworks:
        st.error("""
        No agent frameworks are available. Please set the required environment variables:
        - OPENAI_API_KEY 
        - ANTHROPIC_API_KEY (for Pydantic and LangGraph)
        - GEMINI_API_KEY (for Pydantic and LangGraph)
        """)
        return

    # Get input parameters and the submitted button state
    presentation, llm_model, recruiting_status, min_date, max_date, phase, submitted = input_form.patient_input_form()

    # Track submission in session state to prevent rerunning
    if submitted:
        print(f"Form submitted (Runtime ID: {st.session_state.runtime_id})")
        st.session_state.submitted = True

    # Display framework selector
    framework = framework_selector(available_options=available_frameworks)

    # Reset workflow if inputs change
    current_run_id = f"{presentation[:50]}_{llm_model}_{recruiting_status}_{min_date}_{max_date}_{phase}_{framework}"
    if st.session_state.run_id != current_run_id:
        reset_workflow()
        st.session_state.run_id = current_run_id

    # Create expanders with the session state expanded values
    step1_expander = st.expander("Step 1: Clinical Note Extraction", expanded=st.session_state.step1_expanded)
    step2_expander = st.expander("Step 2: Trial Retrieval", expanded=st.session_state.step2_expanded)
    step3_expander = st.expander("Step 3: Eligibility Matching", expanded=st.session_state.step3_expanded)
    step4_expander = st.expander("Step 4: Trial Scoring", expanded=st.session_state.step4_expanded)

    # Progress indicator
    progress = st.progress(0 if not st.session_state.workflow_complete else 100)

    # Set up the agent function
    agent_map = {
        "pydantic": run_pydantic_agent if pydantic_available else None,
        "langgraph": run_langgraph_agent if langgraph_available else None
    }
    agent_function = agent_map.get(framework)

    if agent_function is None:
        st.error(f"The {framework} agent is not available. Please check your environment variables.")
        return

    # Run the workflow when the submit button is clicked
    if st.session_state.submitted and not st.session_state.workflow_complete and not st.session_state.is_running:
        print(f"Starting workflow execution (Runtime ID: {st.session_state.runtime_id})")
        results = run_workflow(agent_function, presentation, llm_model, recruiting_status, min_date, max_date, phase)
        progress.progress(100)
        print(f"Workflow execution complete, results obtained: {results is not None}")

    # Display results if workflow is complete
    if st.session_state.workflow_complete and st.session_state.results is not None:
        print(f"Displaying results (Runtime ID: {st.session_state.runtime_id})")
        results = st.session_state.results
        progress.progress(100)

        # Display each step's results
        if "step1" in results and results["step1"] is not None:
            display_step1_results(results["step1"], step1_expander)

        if "step2" in results and results["step2"] is not None:
            display_step2_results(results["step2"], step2_expander)

        if "step3" in results and results["step3"] is not None:
            display_step3_results(results["step3"], step3_expander)

        if "step4" in results and results["step4"] is not None:
            display_step4_results(results["step4"], step4_expander)

        # Display the ranked trials outside the expanders
        display_ranked_trials(results)

    # Reset button in the sidebar
    if st.sidebar.button("Reset Workflow", key="reset_workflow"):
        reset_workflow()


if __name__ == "__main__":
    print(f"Main script started. Time: {time.time()}")
    main()
