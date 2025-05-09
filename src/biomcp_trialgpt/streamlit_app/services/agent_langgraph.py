import networkx as nx
from typing import List, Dict, Any, Optional
from datetime import date
from .biomcp_client import BioMCPClient
from .agent_openai import extract_concepts, match_eligibility, score_trial

def run_langgraph_agent(
    client: BioMCPClient,
    patient_data: Dict[str, Any],
    recruiting_status: Optional[str] = None,
    min_date: Optional[date] = None,
    max_date: Optional[date] = None,
    phase: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Agent pipeline orchestrated as a directed graph using NetworkX."""
    # Define graph structure
    graph = nx.DiGraph()
    graph.add_edges_from([
        ('concepts', 'retrieve'),
        ('retrieve', 'eligibility'),
        ('eligibility', 'scoring'),
    ])
    # 1) Extract concepts
    diagnosis = patient_data.get('diagnosis', '')
    concepts_list = extract_concepts(diagnosis)
    # 2) Retrieve trials via BioMCP with filters
    conditions = patient_data.get('conditions', [])
    terms = patient_data.get('terms', [])
    interventions = patient_data.get('interventions', [])
    trials = client.retrieve_trials(
        conditions=conditions,
        terms=terms,
        interventions=interventions,
        recruiting_status=recruiting_status,
        min_date=min_date,
        max_date=max_date,
        phase=phase,
    )
    # 3) Eligibility and scoring
    results: List[Dict[str, Any]] = []
    for trial in trials:
        explanation = match_eligibility(patient_data, trial)
        score = score_trial(patient_data, trial)
        results.append({**trial, 'explanation': explanation, 'score': score})
    return results
