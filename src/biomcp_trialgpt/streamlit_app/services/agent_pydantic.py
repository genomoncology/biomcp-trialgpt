from typing import Dict, Any, List, Optional
from datetime import date
from pydantic import BaseModel, validate_call
from .biomcp_client import BioMCPClient

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

@validate_call
def match_eligibility_model(patient: Patient, trial: Trial) -> str:
    """Placeholder: eligibility logic."""
    return "Pydantic AI eligibility not yet implemented"

@validate_call
def score_trial_model(patient: Patient, trial: Trial) -> float:
    """Placeholder: scoring logic."""
    return 0.0

def run_pydantic_agent(
    client: BioMCPClient,
    patient_data: Dict[str, Any],
    recruiting_status: Optional[str] = None,
    min_date: Optional[date] = None,
    max_date: Optional[date] = None,
    phase: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Run the Pydantic AI pipeline end-to-end."""
    patient = Patient(**patient_data)
    # Extract search parameters from model output
    conditions = patient_data.get("conditions", [])
    terms = patient_data.get("terms", [])
    interventions = patient_data.get("interventions", [])
    # Retrieve trials via BioMCP with filters
    trials_data = client.retrieve_trials(
        conditions=conditions,
        terms=terms,
        interventions=interventions,
        recruiting_status=recruiting_status,
        min_date=min_date,
        max_date=max_date,
        phase=phase,
    )
    results: List[Dict[str, Any]] = []
    for t in trials_data:
        # Map BioMCP keys to our Trial model
        title = t.get("Study Title") or t.get("title", "")
        nct_id = t.get("NCT Number") or t.get("nct_id", "")
        trial = Trial(title=title, nct_id=nct_id)
        explanation = match_eligibility_model(patient, trial)
        score = score_trial_model(patient, trial)
        result_obj = TrialResult(
            title=trial.title,
            nct_id=trial.nct_id,
            score=score,
            explanation=explanation,
        )
        results.append(result_obj.dict())
    return results
