import asyncio
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from typing import Annotated, Any, Callable, Optional, TypedDict, Union, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

# Import LangGraph components
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

# Import original components
from .biomcp_client import BioMCPClient
from .eligibility import run_eligibility
from .note_extractor import get_api_keys, parse_clinical_note
from .response_formatter import format_response_for_ui
from .scoring import run_scoring


# Define the agent state schema
class AgentState(TypedDict):
    """The state of the agent."""

    messages: Annotated[list[BaseMessage], add_messages]
    clinical_data: dict[str, Any]
    retrieval_params: dict[str, Any]
    trials: list[dict[str, Any]]
    eligibility_logs: dict[str, Any]
    scoring_logs: dict[str, Any]
    ranked_trials: list[tuple]
    model_name: str  # Explicitly tracking model name in state


# Define error messages as constants
OPENAI_KEY_ERROR = "OpenAI API key not set. Please provide an API key in the sidebar."
ANTHROPIC_KEY_ERROR = "Anthropic API key not set. Please provide an API key in the sidebar."
GOOGLE_KEY_ERROR = "Google Gemini API key not set. Please provide an API key in the sidebar."


def initialize_state(state: AgentState) -> AgentState:
    """Initialize the agent state with default values."""
    return {
        "messages": state.get("messages", []),
        "clinical_data": {},
        "retrieval_params": {},
        "trials": [],
        "eligibility_logs": {},
        "scoring_logs": {},
        "ranked_trials": [],
        "model_name": state.get("model_name", "gpt-4"),
    }


def process_clinical_extraction(state: AgentState) -> AgentState:
    """Extract clinical data from the presentation."""
    presentation = cast(str, state["messages"][0].content)
    model_name = state["model_name"]

    print(f"Extracting clinical data with model: {model_name}")

    try:
        data, _, _ = parse_clinical_note(presentation, model_name)
    except Exception as e:
        data = {"error": str(e), "chief_complaint": "Error in extraction"}

    return {
        **state,
        "clinical_data": data,
        "messages": state["messages"] + [AIMessage(content=f"Extracted clinical data: {json.dumps(data, indent=2)}")],
    }


def setup_trial_retrieval(state: AgentState) -> AgentState:
    """Set up parameters for trial retrieval."""
    clinical_data = state["clinical_data"]

    if not state.get("retrieval_params"):
        current_year = datetime.now().year
        params = {
            "recruiting_status": "Recruiting",
            "min_date": "2018-01-01",
            "max_date": f"{current_year + 1}-12-31",
            "phase": "Phase 1|Phase 2|Phase 3",
            "conditions": clinical_data.get("conditions", []),
            "terms": clinical_data.get("terms", []),
            "interventions": clinical_data.get("interventions", []),
        }
    else:
        params = state["retrieval_params"]

    return {
        **state,
        "retrieval_params": params,
        "messages": state["messages"]
        + [AIMessage(content=f"Set up trial retrieval with parameters: {json.dumps(params, indent=2)}")],
    }


def retrieve_clinical_trials(state: AgentState) -> AgentState:
    """Retrieve clinical trials based on parameters."""
    params = state["retrieval_params"]

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        trials = loop.run_until_complete(BioMCPClient().retrieve_trials(**params))
    finally:
        loop.close()

    return {
        **state,
        "trials": trials,
        "messages": state["messages"] + [AIMessage(content=f"Retrieved {len(trials)} clinical trials.")],
    }


def _batch_process(
    items: list[dict[str, Any]],
    process_fn: Callable[[dict[str, Any]], Any],
    batch_size: int = 10,
    max_workers: int = 4,
    desc: str = "Processing",
) -> dict[str, Any]:
    """Generic batch processing utility."""
    results: dict[str, Any] = {}

    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        print(f"{desc} batch {i // batch_size + 1}/{(len(items) + batch_size - 1) // batch_size}")

        with ThreadPoolExecutor(max_workers=min(max_workers, len(batch))) as executor:
            futures = {}
            for item in batch:
                key = item.get("NCT Number") or item.get("nct_id", "")
                futures[executor.submit(process_fn, item)] = key

            for future in as_completed(futures):
                key = futures[future]
                try:
                    results[key] = future.result()
                except Exception as e:
                    results[key] = {"error": str(e)}

    return results


def process_eligibility(state: AgentState) -> AgentState:
    """Process eligibility for all retrieved trials."""
    presentation = cast(str, state["messages"][0].content)
    trials = state["trials"]
    model_name = state["model_name"]

    print(f"Checking eligibility with model: {model_name} for {len(trials)} trials")

    # Create a function that captures the presentation and model_name
    def process_trial(trial: dict[str, Any]) -> Any:
        return run_eligibility(presentation, trial, model_name)

    eligibility_logs = _batch_process(trials, process_trial, desc="Processing eligibility")

    # Handle any errors and ensure consistent format
    for nct_id, result in eligibility_logs.items():
        if "error" in result:
            eligibility_logs[nct_id] = {
                "inclusion": ("", f"Error: {result['error']}"),
                "exclusion": ("", f"Error: {result['error']}"),
            }

    return {
        **state,
        "eligibility_logs": eligibility_logs,
        "messages": state["messages"]
        + [AIMessage(content=f"Completed eligibility assessment for {len(eligibility_logs)} trials.")],
    }


def process_scoring(state: AgentState) -> AgentState:
    """Score and rank all trials."""
    presentation = cast(str, state["messages"][0].content)
    trials = state["trials"]
    eligibility_logs = state["eligibility_logs"]
    model_name = state["model_name"]

    print(f"Scoring trials with model: {model_name}")

    # Get only trials with eligibility data
    eligible_trials = [t for t in trials if (t.get("NCT Number") or t.get("nct_id", "")) in eligibility_logs]

    # Define scoring function
    def score_single_trial(trial: dict[str, Any]) -> dict[str, Any]:
        nct_id = trial.get("NCT Number") or trial.get("nct_id", "")
        inc_p, inc_r = eligibility_logs[nct_id]["inclusion"]
        exc_p, exc_r = eligibility_logs[nct_id]["exclusion"]
        pred_str = f"Inclusion predictions:\n{inc_r}\nExclusion predictions:\n{exc_r}"

        try:
            _, score_resp = run_scoring(presentation, trial, pred_str, model_name)
            try:
                json_match = re.search(r"(\{.*\})", score_resp, re.DOTALL)
                if json_match:
                    result_dict_match: dict[str, Any] = json.loads(json_match.group(1))
                    return result_dict_match
                else:
                    result_dict_full: dict[str, Any] = json.loads(score_resp)
                    return result_dict_full
            except Exception:
                return {
                    "relevance_explanation": "Error parsing response",
                    "relevance_score_R": 0,
                    "eligibility_explanation": "Error parsing response",
                    "eligibility_score_E": 0,
                }
        except Exception as e:
            return {
                "relevance_explanation": f"Error: {e!s}",
                "relevance_score_R": 0,
                "eligibility_explanation": f"Error: {e!s}",
                "eligibility_score_E": 0,
            }

    # Process in batches
    scoring_logs = _batch_process(eligible_trials, score_single_trial, desc="Scoring trials")

    # Rank trials
    ranked_trials = sorted(scoring_logs.items(), key=lambda kv: kv[1].get("eligibility_score_E", 0), reverse=True)

    return {
        **state,
        "scoring_logs": scoring_logs,
        "ranked_trials": ranked_trials,
        "messages": state["messages"]
        + [AIMessage(content=f"Completed scoring and ranking for {len(scoring_logs)} trials.")],
    }


def format_final_result(state: AgentState) -> AgentState:
    """Format the final result with top ranked trials."""
    ranked_trials = state["ranked_trials"]
    trials = state["trials"]

    # Map trials by ID for quick lookup
    trial_map = {t.get("NCT Number") or t.get("nct_id", ""): t for t in trials}

    # Format top 5 trials
    result_parts = ["Top ranked clinical trials:"]
    for i, (nct_id, data) in enumerate(ranked_trials[:5], 1):
        trial_info = trial_map.get(nct_id, {})
        summary = trial_info.get("Brief Summary", "No summary available")
        if len(summary) > 200:
            summary = summary[:200] + "..."

        result_parts.append(
            f"{i}. NCT ID: {nct_id}\n"
            f"   Title: {trial_info.get('Title', 'Unknown')}\n"
            f"   Eligibility Score: {data.get('eligibility_score_E', 0)}\n"
            f"   Relevance Score: {data.get('relevance_score_R', 0)}\n"
            f"   Summary: {summary}\n\n"
        )

    return {**state, "messages": state["messages"] + [AIMessage(content="".join(result_parts))]}


def run_langgraph_agent(
    presentation: str,
    llm_model: str = "gpt-4",
    recruiting_status: str = "Recruiting",
    min_date: Optional[Union[date, int]] = None,
    max_date: Optional[Union[date, int]] = None,
    phase: str = "Phase 1|Phase 2|Phase 3",
    debug_mode: bool = False,
) -> dict[str, Any]:
    """Run the LangGraph agent pipeline for clinical trial matching."""
    print(f"Starting workflow with model: {llm_model}")

    # Get API keys from session state
    openai_key, anthropic_key, google_key = get_api_keys()

    # Ensure we have the appropriate API key for the selected model
    if "gpt-" in llm_model and not openai_key:
        raise ValueError(OPENAI_KEY_ERROR)
    if "anthropic" in llm_model and not anthropic_key:
        raise ValueError(ANTHROPIC_KEY_ERROR)
    if "google-" in llm_model and not google_key:
        raise ValueError(GOOGLE_KEY_ERROR)

    # Format dates
    min_date_str = _format_date(min_date, "2018-01-01")
    max_date_str = _format_date(max_date, f"{datetime.now().year + 1}-12-31")

    # Create initial state
    initial_state = {
        "messages": [HumanMessage(content=presentation)],
        "clinical_data": {},
        "retrieval_params": {
            "recruiting_status": recruiting_status,
            "min_date": min_date_str,
            "max_date": max_date_str,
            "phase": phase,
        },
        "trials": [],
        "eligibility_logs": {},
        "scoring_logs": {},
        "ranked_trials": [],
        "model_name": llm_model,
    }

    # Execute workflow
    workflow = _build_workflow_graph()
    if debug_mode:
        # Import trace only when needed to avoid import error if not available
        try:
            from langgraph.trace import trace  # type: ignore[import-not-found]

            with trace(workflow) as tracer:
                result = workflow.invoke(initial_state)
                trace_events = tracer.get_events()
                return {"result": result, "trace": trace_events}
        except ImportError:
            # Fallback if trace is not available
            result = workflow.invoke(initial_state)
            return {"result": result, "trace": []}

    # Standard execution with formatted response
    result = workflow.invoke(initial_state)
    return format_response_for_ui(
        clinical_data=result.get("clinical_data", {}),
        retrieval_params=result.get("retrieval_params", {}),
        trials=result.get("trials", []),
        eligibility_logs=result.get("eligibility_logs", {}),
        scoring_logs=result.get("scoring_logs", {}),
        ranked_trials=result.get("ranked_trials", []),
        model_name=result.get("model_name", llm_model),
    )


def _format_date(date_input: Optional[Union[date, int]], default: str) -> str:
    """Format date input for API."""
    if date_input is None:
        return default
    elif isinstance(date_input, int):
        return f"{date_input}-01-01" if "min" in default else f"{date_input}-12-31"
    else:
        return date_input.strftime("%Y-%m-%d")


def _build_workflow_graph() -> Any:
    """Build and return the workflow graph."""
    graph = StateGraph(AgentState)

    # Add nodes and edges in one concise section
    nodes = {
        "initialize": initialize_state,
        "extract_clinical_data": process_clinical_extraction,
        "setup_retrieval": setup_trial_retrieval,
        "retrieve_trials": retrieve_clinical_trials,
        "check_eligibility": process_eligibility,
        "score_trials": process_scoring,
        "format_result": format_final_result,
    }

    # Add all nodes
    for name, func in nodes.items():
        graph.add_node(name, func)

    # Define edges (flow)
    graph.set_entry_point("initialize")
    for i, (name, _) in enumerate(list(nodes.items())[:-1]):
        next_node = list(nodes.items())[i + 1][0]
        graph.add_edge(name, next_node)
    graph.add_edge("format_result", END)

    return graph.compile()
