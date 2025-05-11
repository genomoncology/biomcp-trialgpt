from typing import Dict, List, Any
from .biomcp_client import BioMCPClient
from .note_extractor import parse_clinical_note
from .eligibility import run_eligibility
from .scoring import run_scoring
import os
import json
import openai
from openai.error import RateLimitError
from datetime import date
from concurrent.futures import ThreadPoolExecutor, as_completed
from .llm_utils import create_chat_completion as _create_chat_completion

# Configure API key
_api_key = os.getenv("OPENAI_API_KEY")
if not _api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")
openai.api_key = _api_key

# Support both legacy and new client APIs
OpenAIClient = getattr(openai, "OpenAI", None)

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

def run_openai_agent(
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
    """Run the OpenAI agent pipeline."""
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
            from datetime import datetime
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
        trials_list = client.retrieve_trials(**params)
        step2_result = {"params": params, "response": trials_list}
        if step == "step2":
            return step2_result
    else:
        step2_result = step2_data
        trials_list = step2_result["response"]
    
    if step == "step3" or step == "all":
        # Step 3: Eligibility matching (parallel)
        eligibility_logs = {}
        max_workers = max(1, min(len(trials_list), 5))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_key = {
                executor.submit(run_eligibility, presentation, t, llm_model):
                (t.get("NCT Number") or t.get("nct_id", ""))
                for t in trials_list
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
        max_workers = max(1, min(len(trials_list), 5))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_key = {}
            for trial in trials_list:
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
