import json
import logging
from datetime import date
from typing import Optional

from biomcp.trials.search import RecruitingStatus, TrialPhase, TrialQuery, search_trials


class BioMCPClient:
    """Wrapper for BioMCP trial retrieval."""

    def __init__(self):
        # Initialize BioMCP client, authentication, etc.
        pass

    async def retrieve_trials(
        self,
        conditions: list[str],
        terms: list[str],
        interventions: list[str],
        recruiting_status: Optional[str] = None,
        min_date: Optional[date] = None,
        max_date: Optional[date] = None,
        phase: Optional[str] = None,
    ) -> list[dict]:
        """
        Retrieve trials matching the condition, age, and gender.

        Returns:
            List of trial dicts with keys like 'title', 'nct_id'.
        """
        # Build trial search parameters
        query_args = {}
        if conditions:
            query_args["conditions"] = conditions
        if terms:
            query_args["terms"] = terms
        if interventions:
            query_args["interventions"] = interventions
        if recruiting_status and recruiting_status != "ANY":
            query_args["recruiting_status"] = getattr(RecruitingStatus, recruiting_status, None)
        if phase and phase != "N/A":
            phase_key = phase.upper().replace(" ", "")
            query_args["phase"] = getattr(TrialPhase, phase_key, None)
        if min_date:
            # Accept both date objects and already formatted strings
            if isinstance(min_date, date):
                query_args["min_date"] = min_date.isoformat()
            else:
                query_args["min_date"] = min_date
        if max_date:
            if isinstance(max_date, date):
                query_args["max_date"] = max_date.isoformat()
            else:
                query_args["max_date"] = max_date

        # Execute async search and parse JSON
        json_str = await search_trials(TrialQuery(**query_args), output_json=True)
        logging.info(f"{json.loads(json_str)}")
        return json.loads(json_str)
