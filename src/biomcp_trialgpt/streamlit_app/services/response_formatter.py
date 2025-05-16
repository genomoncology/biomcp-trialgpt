from typing import Any, Optional


def _format_clinical_data(
    clinical_data: Optional[dict[str, Any]] = None, patient_data: Optional[Any] = None, model_name: str = ""
) -> dict[str, Any]:
    """Format clinical data extraction results for the UI."""
    step1 = {"model": model_name, "data": {}}

    # Handle data from either agent type
    if clinical_data:
        # LangGraph data
        step1["data"] = clinical_data
    elif patient_data:
        # Pydantic data
        try:
            # Try model_dump first (newer Pydantic)
            if hasattr(patient_data, "model_dump"):
                step1["data"] = patient_data.model_dump(exclude_none=True)
            # Fall back to dict for older Pydantic
            else:
                step1["data"] = patient_data.dict(exclude_none=True)
        except Exception:
            # If all else fails, try a basic dict conversion
            step1["data"] = dict(patient_data) if patient_data else {}

    return step1


def _format_trial_retrieval(
    retrieval_params: Optional[dict[str, Any]] = None,
    trials: Optional[list[dict[str, Any]]] = None,
    config: Optional[Any] = None,
    patient_data: Optional[Any] = None,
    retrieved_trials: Optional[list[Any]] = None,
) -> dict[str, Any]:
    """Format trial retrieval results for the UI."""
    step2 = {"params": {}, "response": []}

    # Handle trial retrieval params
    if retrieval_params:
        # LangGraph parameters
        step2["params"] = retrieval_params
    elif config:
        # Pydantic parameters
        trial_params = {
            "recruiting_status": getattr(config, "recruiting_status", ""),
            "min_date": getattr(config, "min_date", ""),
            "max_date": getattr(config, "max_date", ""),
            "phase": getattr(config, "phase", ""),
            "conditions": (patient_data.conditions if patient_data else []),
            "terms": (patient_data.terms if patient_data else []),
            "interventions": (patient_data.interventions if patient_data else []),
        }
        step2["params"] = trial_params

    # Handle retrieved trials
    if trials:
        # LangGraph trials (already a list of dicts)
        step2["response"] = trials
    elif retrieved_trials:
        # Pydantic trials (need to convert to dicts)
        try:
            trial_list = []
            for trial in retrieved_trials:
                if hasattr(trial, "model_dump"):
                    trial_list.append(trial.model_dump(exclude_none=True))
                elif hasattr(trial, "dict"):
                    trial_list.append(trial.dict(exclude_none=True))
                else:
                    trial_list.append(dict(trial))
            step2["response"] = trial_list
        except Exception:
            # Fallback
            step2["response"] = []

    return step2


def _format_eligibility_assessment(
    eligibility_logs: Optional[dict[str, Any]] = None, eligibility_results: Optional[list[Any]] = None
) -> dict[str, Any]:
    """Format eligibility assessment results for the UI."""
    step3 = {"results": []}

    # For consistent formatting to match what the UI expects
    if eligibility_logs:
        # LangGraph eligibility logs - need to reformat to match Pydantic format
        results = []
        for nct_id, data in eligibility_logs.items():
            results.append({
                "trial_id": nct_id,
                "eligibility": {
                    "inclusion": data.get("inclusion", ("", "")),
                    "exclusion": data.get("exclusion", ("", "")),
                },
            })
        step3 = {"results": results}
    elif eligibility_results:
        # Pydantic eligibility results - already in correct format
        results = []
        for result in eligibility_results:
            # Convert Pydantic model to dict
            if hasattr(result, "trial_id"):
                results.append({
                    "trial_id": result.trial_id,
                    "eligibility": {
                        "inclusion": (
                            getattr(result, "inclusion_decision", ""),
                            getattr(result, "inclusion_explanation", ""),
                        ),
                        "exclusion": (
                            getattr(result, "exclusion_decision", ""),
                            getattr(result, "exclusion_explanation", ""),
                        ),
                    },
                })
        step3 = {"results": results}

    return step3


def _format_scoring_and_ranking(
    scoring_logs: Optional[dict[str, Any]] = None,
    ranked_trials: Optional[list[tuple[str, dict[str, Any]]]] = None,
    scoring_results: Optional[list[Any]] = None,
) -> dict[str, Any]:
    """Format scoring and ranking results for the UI."""
    step4 = {"scoring_logs": {}, "ranked": []}

    # Handle scoring logs
    if scoring_logs:
        # LangGraph scoring logs (already a dict)
        step4["scoring_logs"] = scoring_logs
    elif scoring_results:
        # Pydantic scoring results
        logs = {}
        for result in scoring_results:
            if hasattr(result, "trial_id"):
                logs[result.trial_id] = {
                    "relevance_explanation": getattr(result, "relevance_explanation", ""),
                    "relevance_score_R": getattr(result, "relevance_score", 0),
                    "eligibility_explanation": getattr(result, "eligibility_explanation", ""),
                    "eligibility_score_E": getattr(result, "eligibility_score", 0),
                }
        step4["scoring_logs"] = logs

    # Handle ranked trials
    if ranked_trials:
        # LangGraph ranked trials (already in correct tuple format)
        step4["ranked"] = ranked_trials
    elif scoring_results:
        # Pydantic ranked trials - create a list of tuples like LangGraph
        ranked = []
        for result in sorted(scoring_results, key=lambda r: getattr(r, "eligibility_score", 0), reverse=True):
            if hasattr(result, "trial_id"):
                trial_id = result.trial_id
                data = {
                    "relevance_explanation": getattr(result, "relevance_explanation", ""),
                    "relevance_score_R": getattr(result, "relevance_score", 0),
                    "eligibility_explanation": getattr(result, "eligibility_explanation", ""),
                    "eligibility_score_E": getattr(result, "eligibility_score", 0),
                }
                ranked.append((trial_id, data))
        step4["ranked"] = ranked

    return step4


def format_response_for_ui(
    # Core data for formatting
    clinical_data: Optional[dict[str, Any]] = None,
    retrieval_params: Optional[dict[str, Any]] = None,
    trials: Optional[list[dict[str, Any]]] = None,
    eligibility_logs: Optional[dict[str, Any]] = None,
    scoring_logs: Optional[dict[str, Any]] = None,
    ranked_trials: Optional[list[tuple[str, dict[str, Any]]]] = None,
    model_name: str = "",
    # Pydantic-specific data (optional)
    patient_data: Optional[Any] = None,
    retrieved_trials: Optional[list[Any]] = None,
    eligibility_results: Optional[list[Any]] = None,
    scoring_results: Optional[list[Any]] = None,
    config: Optional[Any] = None,
) -> dict[str, Any]:
    """
    Create a standardized response format for the UI that works with both agent types.

    This function ensures consistent output format regardless of which agent implementation
    is used (Pydantic or LangGraph).

    Args:
        clinical_data: Extracted patient clinical data (LangGraph)
        retrieval_params: Trial retrieval parameters (LangGraph)
        trials: Retrieved clinical trials (LangGraph)
        eligibility_logs: Eligibility assessment logs (LangGraph)
        scoring_logs: Trial scoring logs (LangGraph)
        ranked_trials: Ranked trials list (LangGraph)
        model_name: LLM model name used

        patient_data: Patient data model (Pydantic)
        retrieved_trials: Trial models (Pydantic)
        eligibility_results: Eligibility result models (Pydantic)
        scoring_results: Scoring result models (Pydantic)
        config: Pipeline configuration (Pydantic)

    Returns:
        A dictionary with the standard response format expected by the UI
    """
    # Initialize the response structure with each step's formatted data
    response = {
        "step1": _format_clinical_data(clinical_data, patient_data, model_name),
        "step2": _format_trial_retrieval(retrieval_params, trials, config, patient_data, retrieved_trials),
        "step3": _format_eligibility_assessment(eligibility_logs, eligibility_results),
        "step4": _format_scoring_and_ranking(scoring_logs, ranked_trials, scoring_results),
    }

    return response
