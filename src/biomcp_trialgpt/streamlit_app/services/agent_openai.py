from typing import Dict, List, Any
from .biomcp_client import BioMCPClient
import os
import json
import openai
from openai.error import RateLimitError
from datetime import date

# Configure API key
_api_key = os.getenv("OPENAI_API_KEY")
if not _api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")
openai.api_key = _api_key

# Support both legacy and new client APIs
OpenAIClient = getattr(openai, "OpenAI", None)

def _create_chat_completion(**kwargs):
    # Try legacy ChatCompletion
    if hasattr(openai, "ChatCompletion"):
        return openai.ChatCompletion.create(**kwargs)
    # Fallback to new client
    if OpenAIClient:
        client = OpenAIClient(api_key=_api_key)
        return client.chat.completions.create(**kwargs)
    raise RuntimeError("No OpenAI chat completion method available")

def extract_concepts(diagnosis: str) -> List[str]:
    """Extract key medical terms from patient diagnosis using OpenAI."""
    prompt = (
        f"Extract the key medical terms from the following patient diagnosis:\n\n{diagnosis}\n\n"
        "Return the terms as a JSON array of strings."
    )
    try:
        resp = _create_chat_completion(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You extract medical terms."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        content = resp.choices[0].message.content.strip()
        try:
            terms = json.loads(content)
            if isinstance(terms, list):
                return terms
        except json.JSONDecodeError:
            pass
        # Fallback: split on commas/newlines
        return [t.strip() for t in content.replace("\n", ",").split(",") if t.strip()]
    except RateLimitError:
        # Fallback: simple split
        return [t.strip() for t in diagnosis.replace("\n", ",").split(",") if t.strip()]

def match_eligibility(patient_data: Dict[str, Any], trial: Dict[str, Any]) -> str:
    """Use OpenAI to explain patient eligibility for a trial."""
    criteria = trial.get("eligibility", "")
    prompt = (
        f"Patient: {patient_data}\n\nTrial eligibility criteria:\n{criteria}\n\n"
        "Based on the above, explain if the patient is eligible."
    )
    try:
        resp = _create_chat_completion(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You assess clinical trial eligibility."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()
    except RateLimitError:
        return "Eligibility check skipped: rate limit exceeded"

def score_trial(patient_data: Dict[str, Any], trial: Dict[str, Any]) -> float:
    """Score how well a trial matches the patient on a 0-1 scale."""
    prompt = (
        f"Patient: {patient_data}\nTrial: {trial.get('title', '')}\n\n"
        "On a scale from 0 (poor match) to 1 (perfect match), respond with only the numeric score."
    )
    try:
        resp = _create_chat_completion(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You score patient-trial matches."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        text = resp.choices[0].message.content.strip()
        return float(text) if text else 0.0
    except RateLimitError:
        return 0.0
    except ValueError:
        return 0.0

def run_openai_agent(
    client: BioMCPClient,
    patient_data: Dict[str, Any],
    recruiting_status: str = None,
    min_date: date = None,
    max_date: date = None,
    phase: str = None
) -> List[Dict[str, Any]]:
    """Run the OpenAI-like agent pipeline."""
    diagnosis = patient_data.get("diagnosis", "")
    concepts = extract_concepts(diagnosis)
    trials = client.retrieve_trials(
        conditions=patient_data.get("conditions", []),
        terms=patient_data.get("terms", []),
        interventions=patient_data.get("interventions", []),
        recruiting_status=recruiting_status,
        min_date=min_date,
        max_date=max_date,
        phase=phase,
    )
    results: List[Dict[str, Any]] = []
    for trial in trials:
        explanation = match_eligibility(patient_data, trial)
        score = score_trial(patient_data, trial)
        results.append({**trial, "explanation": explanation, "score": score})
    return results
