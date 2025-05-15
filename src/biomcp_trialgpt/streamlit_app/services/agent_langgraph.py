import os
import networkx as nx
from typing import List, Dict, Any, Optional, Sequence, Annotated, TypedDict, Literal, Union
from datetime import date, datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import asyncio

# Import LangGraph components
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent, ToolNode, tools_condition, InjectedState
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

# Import original components
from .biomcp_client import BioMCPClient
from .note_extractor import parse_clinical_note
from .eligibility import run_eligibility
from .scoring import run_scoring


# Define the agent state schema
class AgentState(TypedDict):
    """The state of the agent."""
    # Messages that will be passed between nodes
    messages: Annotated[List[BaseMessage], add_messages]
    # Clinical data extracted from presentation
    clinical_data: Dict[str, Any]
    # Parameters for trial retrieval
    retrieval_params: Dict[str, Any]
    # List of retrieved trials
    trials: List[Dict[str, Any]]
    # Eligibility results
    eligibility_logs: Dict[str, Any]
    # Scoring results
    scoring_logs: Dict[str, Any]
    # Final ranked trials
    ranked_trials: List[tuple]


# Define tools for the LLM to use
# Direct function without @tool decorator for internal use
def extract_clinical_data_internal(presentation: str, model_name: str) -> Dict[str, Any]:
    """
    Extract structured clinical data from a clinical presentation text.
    Internal version that doesn't use the tool decorator.

    Args:
        presentation: The clinical text to parse
        model_name: The LLM model to use for extraction

    Returns:
        Structured clinical data including conditions, terms, and interventions
    """
    data, prompt, response = parse_clinical_note(presentation, model_name)
    return {
        "data": data,
        "prompt": prompt,
        "response": response
    }


# Tool version for use with agents
@tool
def extract_clinical_data(presentation: str, model_name: str) -> Dict[str, Any]:
    """
    Extract structured clinical data from a clinical presentation text.

    Args:
        presentation: The clinical text to parse
        model_name: The LLM model to use for extraction

    Returns:
        Structured clinical data including conditions, terms, and interventions
    """
    return extract_clinical_data_internal(presentation, model_name)


@tool
def retrieve_trials(
        conditions: List[str],
        terms: List[str],
        interventions: List[str],
        recruiting_status: str,
        min_date: str,
        max_date: str,
        phase: str
) -> List[Dict[str, Any]]:
    """
    Retrieve clinical trials based on provided parameters.

    Args:
        conditions: List of medical conditions
        terms: List of search terms
        interventions: List of medical interventions
        recruiting_status: Status of recruitment (e.g., "Recruiting")
        min_date: Start date in YYYY-MM-DD format
        max_date: End date in YYYY-MM-DD format
        phase: Clinical trial phase

    Returns:
        List of clinical trials matching the criteria
    """
    client = BioMCPClient()
    params = {
        "recruiting_status": recruiting_status,
        "min_date": min_date,
        "max_date": max_date,
        "phase": phase,
        "conditions": conditions,
        "terms": terms,
        "interventions": interventions,
    }
    # Create a new event loop and run the async function
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        trials = loop.run_until_complete(client.retrieve_trials(**params))
        return trials
    finally:
        loop.close()


@tool
def check_eligibility(
        presentation: str,
        trial: Dict[str, Any],
        model_name: str,
        nct_id: Annotated[str, InjectedState("nct_id")]
) -> Dict[str, Any]:
    """
    Check if a patient is eligible for a specific clinical trial.

    Args:
        presentation: The clinical presentation text
        trial: The trial to check eligibility for
        model_name: The LLM model to use
        nct_id: The NCT ID of the trial (injected from state)

    Returns:
        Eligibility assessment results
    """
    result = run_eligibility(presentation, trial, model_name)
    return {nct_id: result}


@tool
def score_trial(
        presentation: str,
        trial: Dict[str, Any],
        eligibility_results: str,
        model_name: str,
        nct_id: Annotated[str, InjectedState("nct_id")]
) -> Dict[str, Any]:
    """
    Score a clinical trial for relevance to patient.

    Args:
        presentation: The clinical presentation text
        trial: The trial to score
        eligibility_results: Eligibility predictions to consider
        model_name: The LLM model to use
        nct_id: The NCT ID of the trial (injected from state)

    Returns:
        Scoring assessment results
    """
    _, score_resp = run_scoring(presentation, trial, eligibility_results, model_name)
    try:
        import re
        json_match = re.search(r'(\{.*\})', score_resp, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            data = json.loads(json_str)
        else:
            data = json.loads(score_resp)
    except:
        data = {
            "relevance_explanation": "Error parsing response",
            "relevance_score_R": 0,
            "eligibility_explanation": "Error parsing response",
            "eligibility_score_E": 0,
            "raw_response": score_resp[:100] + "..." if len(score_resp) > 100 else score_resp
        }
    return {nct_id: data}


# Define nodes for the state graph
def initialize_state(state: AgentState) -> Dict[str, Any]:
    """Initialize the agent state with default values."""
    # Preserve existing message values from the state if present
    messages = state.get("messages", [])

    # Return a new state with default values for all required keys
    return {
        "clinical_data": {},
        "retrieval_params": {},
        "trials": [],
        "eligibility_logs": {},
        "scoring_logs": {},
        "ranked_trials": [],
        "messages": messages  # Preserve messages
    }


def process_clinical_extraction(state: AgentState) -> Dict[str, Any]:
    """Process clinical data extraction from the presentation."""
    # Extract the presentation and model from the first message
    message_content = state["messages"][0].content
    presentation = message_content
    model_name = "gpt-4"  # Default model, could be extracted from message or state

    # Extract clinical data using the internal non-tool version
    extraction_result = extract_clinical_data_internal(presentation, model_name)

    return {
        "clinical_data": extraction_result["data"],
        "messages": [
            AIMessage(content=f"Extracted clinical data: {json.dumps(extraction_result['data'], indent=2)}")
        ]
    }


def setup_trial_retrieval(state: AgentState) -> Dict[str, Any]:
    """Set up parameters for trial retrieval."""
    # Handle date parameters
    current_year = datetime.now().year
    min_date_str = "2018-01-01"  # Default
    max_date_str = f"{current_year + 1}-12-31"  # Default to next year

    # Extract data from clinical data
    clinical_data = state["clinical_data"]

    params = {
        "recruiting_status": "Recruiting",  # Default value
        "min_date": min_date_str,
        "max_date": max_date_str,
        "phase": "Phase 1|Phase 2|Phase 3",  # Default value
        "conditions": clinical_data.get("conditions", []),
        "terms": clinical_data.get("terms", []),
        "interventions": clinical_data.get("interventions", []),
    }

    return {
        "retrieval_params": params,
        "messages": [
            AIMessage(content=f"Set up trial retrieval with parameters: {json.dumps(params, indent=2)}")
        ]
    }


def retrieve_clinical_trials(state: AgentState) -> Dict[str, Any]:
    """Retrieve clinical trials based on parameters."""
    params = state["retrieval_params"]

    # Retrieve trials
    client = BioMCPClient()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        trials = loop.run_until_complete(client.retrieve_trials(
            conditions=params.get("conditions", []),
            terms=params.get("terms", []),
            interventions=params.get("interventions", []),
            recruiting_status=params.get("recruiting_status", ""),
            min_date=params.get("min_date", ""),
            max_date=params.get("max_date", ""),
            phase=params.get("phase", "")
        ))
    finally:
        loop.close()

    return {
        "trials": trials,
        "messages": [
            AIMessage(content=f"Retrieved {len(trials)} clinical trials.")
        ]
    }


def process_eligibility(state: AgentState) -> Dict[str, Any]:
    """Process eligibility for all retrieved trials."""
    presentation = state["messages"][0].content
    trials = state["trials"]
    model_name = "gpt-4"  # Default model

    eligibility_logs = {}
    # Set up parallel processing
    max_workers = max(1, min(len(trials), 5))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        # Create a local state with nct_id injected for each trial
        for trial in trials:
            nct_id = trial.get("NCT Number") or trial.get("nct_id", "")
            # Create future with the proper trial and nct_id injection
            future = executor.submit(
                check_eligibility,
                presentation,
                trial,
                model_name,
                nct_id
            )
            futures[future] = nct_id

        # Collect results as they complete
        for future in as_completed(futures):
            try:
                result = future.result()
                eligibility_logs.update(result)
            except Exception as e:
                nct_id = futures[future]
                eligibility_logs[nct_id] = {
                    "inclusion": ("", f"Error: {e}"),
                    "exclusion": ("", f"Error: {e}"),
                }

    return {
        "eligibility_logs": eligibility_logs,
        "messages": [
            AIMessage(content=f"Completed eligibility assessment for {len(eligibility_logs)} trials.")
        ]
    }


def process_scoring(state: AgentState) -> Dict[str, Any]:
    """Score and rank all trials."""
    presentation = state["messages"][0].content
    trials = state["trials"]
    eligibility_logs = state["eligibility_logs"]
    model_name = "gpt-4"  # Default model

    scoring_logs = {}
    # Set up parallel processing
    max_workers = max(1, min(len(trials), 5))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        # Process each trial
        for trial in trials:
            nct_id = trial.get("NCT Number") or trial.get("nct_id", "")
            # Skip if no eligibility data
            if nct_id not in eligibility_logs:
                continue

            # Prepare eligibility predictions string
            inc_p, inc_r = eligibility_logs[nct_id]["inclusion"]
            exc_p, exc_r = eligibility_logs[nct_id]["exclusion"]
            pred_str = f"Inclusion predictions:\n{inc_r}\nExclusion predictions:\n{exc_r}"

            # Create future with the proper trial and nct_id injection
            future = executor.submit(
                score_trial,
                presentation,
                trial,
                pred_str,
                model_name,
                nct_id
            )
            futures[future] = nct_id

        # Collect results as they complete
        for future in as_completed(futures):
            try:
                result = future.result()
                scoring_logs.update(result)
            except Exception as e:
                nct_id = futures[future]
                scoring_logs[nct_id] = {
                    "relevance_explanation": f"Error: {e}",
                    "relevance_score_R": 0,
                    "eligibility_explanation": f"Error: {e}",
                    "eligibility_score_E": 0
                }

    # Rank trials by eligibility score
    ranked = sorted(
        scoring_logs.items(),
        key=lambda kv: kv[1].get("eligibility_score_E", 0),
        reverse=True
    )

    return {
        "scoring_logs": scoring_logs,
        "ranked_trials": ranked,
        "messages": [
            AIMessage(content=f"Completed scoring and ranking for {len(scoring_logs)} trials.")
        ]
    }


def format_final_result(state: AgentState) -> Dict[str, Any]:
    """Format the final result with top ranked trials."""
    ranked_trials = state["ranked_trials"]

    # Get top 5 trials or fewer if less are available
    top_trials = ranked_trials[:min(5, len(ranked_trials))]

    result_str = "Top ranked clinical trials:\n\n"

    for i, (nct_id, data) in enumerate(top_trials, 1):
        # Find the trial data
        trial_info = next(
            (t for t in state["trials"] if t.get("NCT Number") == nct_id or t.get("nct_id") == nct_id),
            {"Title": "Unknown", "Brief Summary": "No summary available"}
        )

        result_str += f"{i}. NCT ID: {nct_id}\n"
        result_str += f"   Title: {trial_info.get('Title', 'Unknown')}\n"
        result_str += f"   Eligibility Score: {data.get('eligibility_score_E', 0)}\n"
        result_str += f"   Relevance Score: {data.get('relevance_score_R', 0)}\n"
        result_str += f"   Summary: {trial_info.get('Brief Summary', 'No summary available')[:200]}...\n\n"

    return {
        "messages": [
            AIMessage(content=result_str)
        ]
    }


def run_langgraph_agent(
        presentation: str,
        llm_model: str,
        recruiting_status: str,
        min_date: date,
        max_date: date,
        phase: str,
        step: str = "all",
        step1_data=None,
        step2_data=None,
        step3_data=None,
) -> Dict[str, Any]:
    """Run the LangGraph agent pipeline."""

    # Create the initial state with all required keys to avoid KeyError
    initial_state = {
        "messages": [HumanMessage(content=presentation)],
        "clinical_data": {},
        "retrieval_params": {},
        "trials": [],
        "eligibility_logs": {},
        "scoring_logs": {},
        "ranked_trials": []
    }

    # Define the graph
    graph_builder = StateGraph(AgentState)

    # Add nodes to the graph
    graph_builder.add_node("initialize", initialize_state)
    graph_builder.add_node("extract_clinical_data", process_clinical_extraction)
    graph_builder.add_node("setup_retrieval", setup_trial_retrieval)
    graph_builder.add_node("retrieve_trials", retrieve_clinical_trials)
    graph_builder.add_node("check_eligibility", process_eligibility)
    graph_builder.add_node("score_trials", process_scoring)
    graph_builder.add_node("format_result", format_final_result)

    # Define the edges of the graph
    graph_builder.set_entry_point("initialize")
    graph_builder.add_edge("initialize", "extract_clinical_data")
    graph_builder.add_edge("extract_clinical_data", "setup_retrieval")
    graph_builder.add_edge("setup_retrieval", "retrieve_trials")
    graph_builder.add_edge("retrieve_trials", "check_eligibility")
    graph_builder.add_edge("check_eligibility", "score_trials")
    graph_builder.add_edge("score_trials", "format_result")
    graph_builder.add_edge("format_result", END)

    # Compile the graph
    graph = graph_builder.compile()

    # Handle step-by-step execution if requested
    if step != "all":
        if step == "step1":
            # Just do clinical note extraction directly without using the graph
            # This avoids potential issues with LangGraph's machinery
            data, prompt, response = parse_clinical_note(presentation, llm_model)

            return {
                "model": llm_model,
                "prompt": prompt,
                "response": response,
                "data": data
            }

        elif step == "step2":
            # Trial retrieval step
            if not step1_data:
                raise ValueError("Step 1 data is required for step 2")

            # Handle date formats
            if min_date is None:
                min_date_str = "2018-01-01"  # Default to 2018
            elif isinstance(min_date, int):
                min_date_str = f"{min_date}-01-01"
            else:
                min_date_str = min_date.strftime("%Y-%m-%d")

            if max_date is None:
                current_year = datetime.now().year
                max_date_str = f"{current_year + 1}-12-31"  # Default to next year
            elif isinstance(max_date, int):
                max_date_str = f"{max_date}-12-31"
            else:
                max_date_str = max_date.strftime("%Y-%m-%d")

            # Direct implementation without using the graph
            clinical_data = step1_data.get("data", {})

            # Set up retrieval parameters
            params = {
                "recruiting_status": recruiting_status,
                "min_date": min_date_str,
                "max_date": max_date_str,
                "phase": phase,
                "conditions": clinical_data.get("conditions", []),
                "terms": clinical_data.get("terms", []),
                "interventions": clinical_data.get("interventions", []),
            }

            # Retrieve trials
            client = BioMCPClient()
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                trials = loop.run_until_complete(client.retrieve_trials(
                    conditions=params.get("conditions", []),
                    terms=params.get("terms", []),
                    interventions=params.get("interventions", []),
                    recruiting_status=params.get("recruiting_status", ""),
                    min_date=params.get("min_date", ""),
                    max_date=params.get("max_date", ""),
                    phase=params.get("phase", "")
                ))
            finally:
                loop.close()

            # Format as in original code
            return {
                "params": params,
                "response": trials
            }

        elif step == "step3":
            # Eligibility matching step
            if not step1_data or not step2_data:
                raise ValueError("Step 1 and Step 2 data are required for step 3")

            # Direct implementation without using the graph
            trials = step2_data.get("response", [])

            # Run eligibility matching
            eligibility_logs = {}
            max_workers = max(1, min(len(trials), 5))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_key = {
                    executor.submit(run_eligibility, presentation, t, llm_model):
                        (t.get("NCT Number") or t.get("nct_id", ""))
                    for t in trials
                }
                for future in as_completed(future_to_key):
                    key = future_to_key[future]
                    try:
                        result = future.result()
                    except Exception as e:
                        eligibility_logs[key] = {
                            "inclusion": ("", f"Error: {e}"),
                            "exclusion": ("", f"Error: {e}"),
                        }
                        continue
                    eligibility_logs[key] = result

            return eligibility_logs

        elif step == "step4":
            # Trial Scoring & Ranking step
            if not step1_data or not step2_data or not step3_data:
                raise ValueError("Step 1, Step 2, and Step 3 data are required for step 4")

            # Direct implementation without using the graph
            trials = step2_data.get("response", [])
            eligibility_logs = step3_data

            # Score trials
            scoring_logs = {}
            max_workers = max(1, min(len(trials), 5))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_key = {}
                for trial in trials:
                    key = trial.get("NCT Number") or trial.get("nct_id", "")
                    if key not in eligibility_logs:
                        continue

                    inc_p, inc_r = eligibility_logs[key]["inclusion"]
                    exc_p, exc_r = eligibility_logs[key]["exclusion"]
                    pred_str = f"Inclusion predictions:\n{inc_r}\nExclusion predictions:\n{exc_r}"
                    future = executor.submit(run_scoring, presentation, trial, pred_str, llm_model)
                    future_to_key[future] = key

                for future in as_completed(future_to_key):
                    key = future_to_key[future]
                    try:
                        _, score_resp = future.result()
                    except Exception:
                        scoring_logs[key] = {}
                        continue

                    try:
                        # Extract JSON from response
                        import re
                        json_match = re.search(r'(\{.*\})', score_resp, re.DOTALL)
                        if json_match:
                            json_str = json_match.group(1)
                            data = json.loads(json_str)
                        else:
                            data = json.loads(score_resp)
                    except:
                        data = {
                            "relevance_explanation": "Error parsing response",
                            "relevance_score_R": 0,
                            "eligibility_explanation": "Error parsing response",
                            "eligibility_score_E": 0,
                            "raw_response": score_resp[:100] + "..." if len(score_resp) > 100 else score_resp
                        }
                    scoring_logs[key] = data

            # Rank trials
            ranked = sorted(scoring_logs.items(), key=lambda kv: kv[1].get("eligibility_score_E", 0), reverse=True)

            return {
                "scoring_logs": scoring_logs,
                "ranked": ranked
            }

    # Execute full workflow by calling each step sequentially
    # This avoids potential issues with LangGraph's machinery

    # Step 1: Clinical note extraction
    step1_result = run_langgraph_agent(
        presentation, llm_model, recruiting_status, min_date, max_date, phase,
        step="step1"
    )

    # Step 2: Trial retrieval
    step2_result = run_langgraph_agent(
        presentation, llm_model, recruiting_status, min_date, max_date, phase,
        step="step2", step1_data=step1_result
    )

    # Step 3: Eligibility matching
    step3_result = run_langgraph_agent(
        presentation, llm_model, recruiting_status, min_date, max_date, phase,
        step="step3", step1_data=step1_result, step2_data=step2_result
    )

    # Step 4: Trial scoring & ranking
    step4_result = run_langgraph_agent(
        presentation, llm_model, recruiting_status, min_date, max_date, phase,
        step="step4", step1_data=step1_result, step2_data=step2_result, step3_data=step3_result
    )

    # Return the complete result
    return {
        "step1": step1_result,
        "step2": step2_result,
        "step3": step3_result,
        "step4": step4_result
    }