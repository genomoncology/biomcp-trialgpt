from typing import Dict, Any, List, Optional
from datetime import date, datetime
from pydantic import BaseModel, validate_call
from .biomcp_client import BioMCPClient
from .note_extractor import parse_clinical_note
from .eligibility import run_eligibility
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from .scoring import run_scoring

class Patient(BaseModel):
    # defaults allow missing name/diagnosis from extractor
    name: str = ""
    age: int
    gender: str
    diagnosis: str = ""

class Trial(BaseModel):
    title: str
    nct_id: str
    eligibility: str = ""

class TrialResult(BaseModel):
    title: str
    nct_id: str
    score: float
    explanation: str

@validate_call
def extract_concepts(patient: Patient) -> List[str]:
    """Extract key terms from diagnosis."""
    return [term.strip() for term in patient.diagnosis.split(",") if term.strip()]

def retrieve_trials_model(client: BioMCPClient, concepts: List[str], patient: Patient) -> List[Trial]:
    raw = client.retrieve_trials(
        condition=" ".join(concepts), age=patient.age, gender=patient.gender
    )
    return [Trial(**t) for t in raw]

def run_pydantic_agent(
    presentation: str,
    llm_model: str,
    recruiting_status: str,
    min_date: date,
    max_date: date,
    phase: str,
    step: str = "all",
    step1_data = None,
    step2_data = None,
    step3_data = None,
) -> Dict[str, Any]:
    """Run the Pydantic agent pipeline."""
    # Execute only the requested step or all steps
    if step == "step1" or step == "all":
        # Step 1: Clinical note extraction
        data, prompt, response = parse_clinical_note(presentation, llm_model)
        step1_result = {
            "model": llm_model,
            "prompt": prompt,
            "response": response,
            "data": data
        }
        if step == "step1":
            return step1_result
    else:
        step1_result = step1_data
    
    if step == "step2" or step == "all":
        # Step 2: Trial retrieval
        client = BioMCPClient()
        # Handle different date formats and None values
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
        
        params = {
            "recruiting_status": recruiting_status,
            "min_date": min_date_str,
            "max_date": max_date_str,
            "phase": phase,
            "conditions": step1_result["data"].get("conditions", []),
            "terms": step1_result["data"].get("terms", []),
            "interventions": step1_result["data"].get("interventions", []),
        }
        trials_data = client.get_trials(**params)
        step2_result = {"params": params, "response": trials_data}
        if step == "step2":
            return step2_result
    else:
        step2_result = step2_data
        trials_data = step2_result["response"]
    
    if step == "step3" or step == "all":
        # Step 3: Eligibility matching (parallel)
        eligibility_logs = {}
        max_workers = max(1, min(len(trials_data), 5))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_key = {
                executor.submit(run_eligibility, presentation, t, llm_model):
                (t.get("NCT Number") or t.get("nct_id", ""))
                for t in trials_data
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
        if step == "step3":
            return eligibility_logs
    else:
        eligibility_logs = step3_data
    
    if step == "step4" or step == "all":
        # Step 4: Trial Scoring & Ranking (parallel)
        scoring_logs = {}
        max_workers = max(1, min(len(trials_data), 5))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_key = {}
            for trial in trials_data:
                key = trial.get("NCT Number") or trial.get("nct_id", "")
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
                    # Extract JSON from response - LLMs sometimes add text before/after JSON
                    import re
                    json_match = re.search(r'(\{.*\})', score_resp, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1)
                        data = json.loads(json_str)
                    else:
                        data = json.loads(score_resp)  # Try direct parsing as fallback
                except:
                    # If parsing fails, create a basic structure with error info
                    data = {
                        "relevance_explanation": "Error parsing response",
                        "relevance_score_R": 0,
                        "eligibility_explanation": "Error parsing response",
                        "eligibility_score_E": 0,
                        "raw_response": score_resp[:100] + "..." if len(score_resp) > 100 else score_resp
                    }
                scoring_logs[key] = data
        ranked = sorted(scoring_logs.items(), key=lambda kv: kv[1].get("eligibility_score_E", 0), reverse=True)
        step4_result = {"scoring_logs": scoring_logs, "ranked": ranked}
        if step == "step4":
            return step4_result
    
    # If we're running all steps, return the complete logs
    if step == "all":
        return {
            "step1": step1_result, 
            "step2": step2_result, 
            "step3": eligibility_logs, 
            "step4": step4_result
        }
