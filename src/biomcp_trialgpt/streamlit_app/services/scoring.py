import json
from typing import Dict, Any, Tuple
from biomcp_trialgpt.streamlit_app.services.eligibility import _call_llm


def build_scoring_prompt(presentation: str, trial_info: Dict[str, Any], pred_str: str) -> str:
    prompt = (
        "You are a helpful assistant for clinical trial recruitment. You will be given a "
        "patient note, a clinical trial, and the patient eligibility predictions for each criterion.\n"
    )
    prompt += (
        "Your task is to output two scores, a relevance score (R) and an eligibility score (E), "
        "between the patient and the clinical trial.\n"
    )
    prompt += (
        "First explain the consideration for determining patient-trial relevance. Predict the "
        "relevance score R (0~100), which represents the overall relevance between the patient "
        "and the clinical trial. R=0 denotes the patient is totally irrelevant, and R=100 denotes "
        "exact relevance.\n"
    )
    prompt += (
        "Then explain the consideration for determining patient-trial eligibility. Predict the "
        "eligibility score E (-R~R), where -R <= E <= R. E=-R denotes ineligible, E=R denotes "
        "fully eligible, and E=0 denotes neutral.\n"
    )
    prompt += (
        "Please output a JSON dict formatted as: {\"relevance_explanation\": string, "
        "\"relevance_score_R\": number, \"eligibility_explanation\": string, "
        "\"eligibility_score_E\": number}.\n"
    )
    prompt += "Here is the patient note:\n" + presentation + "\n\n"
    prompt += "Here is the clinical trial description:\n" + json.dumps(trial_info) + "\n\n"
    prompt += "Here are the criterion-level eligibility predictions:\n" + pred_str + "\n\n"
    prompt += "Plain JSON output:"
    return prompt


def run_scoring(presentation: str, trial_info: Dict[str, Any], pred_str: str, model: str) -> Tuple[str, str]:
    prompt = build_scoring_prompt(presentation, trial_info, pred_str)
    resp = _call_llm(prompt, model)
    return prompt, resp
