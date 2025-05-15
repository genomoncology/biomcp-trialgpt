import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from typing import Dict, Any, List, Optional, Union

from pydantic import BaseModel, Field, model_validator, field_validator
from pydantic_ai import Agent, RunContext

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import necessary components from the original code
from .biomcp_client import BioMCPClient
from .note_extractor import parse_clinical_note
from .eligibility import run_eligibility
from .scoring import run_scoring
from .response_formatter import format_response_for_ui


#######################
# SIMPLIFIED MODELS
#######################

class PatientData(BaseModel):
    """Patient information extracted from clinical notes."""
    name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    diagnosis: Optional[str] = None
    conditions: List[str] = Field(default_factory=list)
    terms: List[str] = Field(default_factory=list)
    interventions: List[str] = Field(default_factory=list)
    medical_history: List[str] = Field(default_factory=list)
    medications: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)

    class Config:
        extra = "ignore"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PatientData":
        """Create a PatientData from a dictionary."""
        if "values" in data and isinstance(data["values"], dict):
            return cls(**data["values"])
        return cls(**data)


class TrialFilter(BaseModel):
    """Parameters for filtering clinical trials."""
    recruiting_status: str
    min_date: str
    max_date: str
    phase: str
    conditions: List[str] = Field(default_factory=list)
    terms: List[str] = Field(default_factory=list)
    interventions: List[str] = Field(default_factory=list)


class Trial(BaseModel):
    """Basic clinical trial data."""
    title: Optional[str] = None
    nct_id: Optional[str] = None
    brief_summary: Optional[str] = None
    detailed_description: Optional[str] = None
    status: Optional[str] = None
    phase: Optional[str] = None
    eligibility_criteria: Optional[str] = None
    eligibility_gender: Optional[str] = None
    minimum_age: Optional[str] = None
    maximum_age: Optional[str] = None
    study_start_date: Optional[str] = None
    study_completion_date: Optional[str] = None
    locations: List[str] = Field(default_factory=list)
    conditions: List[str] = Field(default_factory=list)
    interventions: List[str] = Field(default_factory=list)
    sponsors: List[str] = Field(default_factory=list)

    class Config:
        extra = "ignore"

    @model_validator(mode='before')
    @classmethod
    def ensure_trial_id(cls, data):
        """Ensure we have a trial ID from one of the possible fields."""
        if isinstance(data, dict):
            # Try to find an ID from various possible fields
            nct_id = data.get('nct_id') or data.get('NCT_Number') or data.get('NCT Number')

            # If we still don't have an ID, generate a fallback
            if not nct_id:
                # Use title or a UUID as fallback
                if data.get('title'):
                    import hashlib
                    title_hash = hashlib.md5(data['title'].encode()).hexdigest()[:8]
                    nct_id = f"UNKNOWN-{title_hash}"
                else:
                    import uuid
                    nct_id = f"UNKNOWN-{str(uuid.uuid4())[:8]}"

            # Update the data with the ID
            data['nct_id'] = nct_id

        return data

    @field_validator('nct_id', mode='before')
    @classmethod
    def normalize_nct_id(cls, v, info):
        """Handle different NCT ID formats."""
        if not v:
            values = info.data
            return values.get("NCT_Number") or values.get("NCT Number")
        return v

    @field_validator('conditions', 'interventions', mode='before')
    @classmethod
    def parse_delimited_lists(cls, v):
        """Parse pipe-delimited strings into lists."""
        if isinstance(v, str):
            return [item.strip() for item in v.split("|") if item.strip()]
        return v


class EligibilityResult(BaseModel):
    """Assessment of a patient's eligibility for a trial."""
    trial_id: str
    inclusion_decision: str = ""
    inclusion_explanation: str = ""
    exclusion_decision: str = ""
    exclusion_explanation: str = ""

    @field_validator('inclusion_decision', 'inclusion_explanation',
                     'exclusion_decision', 'exclusion_explanation', mode='before')
    @classmethod
    def parse_tuple_results(cls, v, info):
        """Handle tuple format from legacy code."""
        field_name = info.field_name
        if isinstance(v, tuple) and len(v) == 2:
            return v[1] if field_name.endswith('explanation') else v[0]
        return v


class ScoringResult(BaseModel):
    """Scoring of a trial's relevance and eligibility."""
    trial_id: str
    relevance_explanation: str = ""
    relevance_score: float = 0.0
    eligibility_explanation: str = ""
    eligibility_score: float = 0.0

    @field_validator('relevance_score', 'eligibility_score', mode='before')
    @classmethod
    def ensure_float(cls, v):
        """Ensure score values are floats."""
        try:
            return float(v)
        except (ValueError, TypeError):
            return 0.0


class RankedTrial(BaseModel):
    """Trial with ranking information."""
    trial_id: str
    title: Optional[str] = None
    relevance_explanation: str = ""
    relevance_score: float = 0.0
    eligibility_explanation: str = ""
    eligibility_score: float = 0.0


class PipelineResult(BaseModel):
    """Complete pipeline results with all data."""
    patient_data: Optional[PatientData] = None
    retrieved_trials: List[Trial] = Field(default_factory=list)
    eligibility_results: List[EligibilityResult] = Field(default_factory=list)
    scoring_results: List[ScoringResult] = Field(default_factory=list)
    ranked_trials: List[RankedTrial] = Field(default_factory=list)


#######################
# PIPELINE DEPENDENCIES
#######################

class PipelineConfig(BaseModel):
    """Configuration for the clinical trial matching pipeline."""
    clinical_note: str  # Clinical note text
    llm_model: str  # LLM model to use
    recruiting_status: str  # Trial recruiting status filter
    min_date: Union[date, str, int] = Field(default_factory=lambda: date(2018, 1, 1))
    max_date: Union[date, str, int] = None
    phase: str  # Trial phase filter

    @model_validator(mode='before')
    @classmethod
    def process_dates(cls, data):
        """Handle different date formats."""
        # Process min_date
        min_date = data.get('min_date')
        if min_date is None:
            data['min_date'] = "2018-01-01"
        elif isinstance(min_date, int):
            data['min_date'] = f"{min_date}-01-01"
        elif isinstance(min_date, date):
            data['min_date'] = min_date.strftime("%Y-%m-%d")

        # Process max_date
        max_date = data.get('max_date')
        if max_date is None:
            current_year = datetime.now().year
            data['max_date'] = f"{current_year + 1}-12-31"
        elif isinstance(max_date, int):
            data['max_date'] = f"{max_date}-12-31"
        elif isinstance(max_date, date):
            data['max_date'] = max_date.strftime("%Y-%m-%d")

        return data

    @field_validator('llm_model')
    @classmethod
    def normalize_model_name(cls, v):
        """Normalize LLM model names."""
        model = v.replace('google-', '').replace('anthropic-', '')

        # Add provider prefix if missing
        if not model.startswith(('openai:', 'google-gla:', 'anthropic:')):
            if "gpt" in model.lower():
                model = f"openai:{model}"
            elif "gemini" in model.lower():
                model = f"google-gla:{model}"
            elif "claude" in model.lower():
                model = f"anthropic:{model}"

        logger.info(f"Using model: {model}")
        return model


#######################
# ATOMIC TOOLS
#######################

async def extract_patient_data(ctx: RunContext[PipelineConfig]) -> PatientData:
    """
    Extract structured patient data from clinical notes.
    """
    logger.info(f"Extracting patient information using model {ctx.deps.llm_model}")

    # Call the original extract function
    data_dict, _, _ = parse_clinical_note(ctx.deps.clinical_note, ctx.deps.llm_model)

    # Convert to our simplified model
    patient_data = PatientData.from_dict(data_dict)
    logger.debug(f"Extracted patient data: {patient_data.model_dump(exclude_none=True)}")

    return patient_data


async def search_clinical_trials(
        ctx: RunContext[PipelineConfig],
        patient: PatientData
) -> List[Trial]:
    """
    Search for clinical trials based on patient data and filters.
    """
    logger.info("Searching for clinical trials")
    client = BioMCPClient()

    # Create filter parameters
    filter_params = {
        "recruiting_status": ctx.deps.recruiting_status,
        "min_date": ctx.deps.min_date,
        "max_date": ctx.deps.max_date,
        "phase": ctx.deps.phase,
        "conditions": patient.conditions,
        "terms": patient.terms,
        "interventions": patient.interventions
    }

    # Retrieve trials
    logger.info(f"Retrieving trials with parameters: {filter_params}")
    trials_data = await client.retrieve_trials(**filter_params)
    logger.info(f"Retrieved {len(trials_data)} trials")

    # Debug the trial data if we're getting empty results
    if trials_data and all(not trial.get('nct_id') for trial in trials_data):
        logger.debug(f"Trial data format: {json.dumps(trials_data[:1], indent=2, default=str)}")

    # Convert to Trial objects
    trials = []
    for i, trial_dict in enumerate(trials_data):
        try:
            # Make sure we have a minimal valid dict
            if isinstance(trial_dict, dict) and not trial_dict.get('nct_id'):
                # Try to extract NCT ID from various fields
                nct_id = (trial_dict.get('NCT_Number') or
                          trial_dict.get('NCT Number') or
                          f"TRIAL-{i + 1}")
                trial_dict['nct_id'] = nct_id

            trial = Trial.model_validate(trial_dict)
            trials.append(trial)
        except Exception as e:
            logger.warning(f"Failed to parse trial {i + 1}: {e}")

    return trials


async def assess_trial_eligibility(
        ctx: RunContext[PipelineConfig],
        patient: PatientData,
        trial: Trial
) -> EligibilityResult:
    """
    Assess if a patient is eligible for a specific trial.
    """
    logger.info(f"Assessing eligibility for trial {trial.nct_id}")

    try:
        # Ensure trial has an ID
        if not trial.nct_id:
            logger.warning("Trial missing ID, using fallback")
            trial_id = "UNKNOWN-TRIAL"
        else:
            trial_id = trial.nct_id

        # Call eligibility function - make sure to properly await it
        try:
            result = run_eligibility(
                ctx.deps.clinical_note,
                trial.model_dump(exclude_none=True),
                ctx.deps.llm_model
            )

            # Handle non-async result from run_eligibility
            if not asyncio.iscoroutine(result):
                inclusion = result.get("inclusion", ("", ""))
                exclusion = result.get("exclusion", ("", ""))
            else:
                # If it's actually an awaitable, await it
                result_data = await result
                inclusion = result_data.get("inclusion", ("", ""))
                exclusion = result_data.get("exclusion", ("", ""))

        except Exception as e:
            logger.error(f"Error calling run_eligibility: {e}")
            return EligibilityResult(
                trial_id=trial_id,
                inclusion_explanation=f"Error in eligibility assessment: {e}",
                exclusion_explanation=f"Error in eligibility assessment: {e}"
            )

        # Create structured result
        return EligibilityResult(
            trial_id=trial_id,
            inclusion_decision=inclusion[0] if isinstance(inclusion, tuple) else "",
            inclusion_explanation=inclusion[1] if isinstance(inclusion, tuple) else str(inclusion),
            exclusion_decision=exclusion[0] if isinstance(exclusion, tuple) else "",
            exclusion_explanation=exclusion[1] if isinstance(exclusion, tuple) else str(exclusion)
        )
    except Exception as e:
        logger.error(f"Error in eligibility assessment for trial {getattr(trial, 'nct_id', 'unknown')}: {e}")
        return EligibilityResult(
            trial_id=getattr(trial, 'nct_id', 'UNKNOWN-ERROR'),
            inclusion_explanation=f"Error: {e}",
            exclusion_explanation=f"Error: {e}"
        )


async def score_trial(
        ctx: RunContext[PipelineConfig],
        patient: PatientData,
        trial: Trial,
        eligibility: EligibilityResult
) -> ScoringResult:
    """
    Score a trial based on its relevance and eligibility for the patient.
    """
    logger.info(f"Scoring trial {trial.nct_id}")

    try:
        # Ensure trial has an ID
        if not trial.nct_id:
            logger.warning("Trial missing ID in scoring, using fallback")
            trial_id = eligibility.trial_id or "UNKNOWN-TRIAL"
        else:
            trial_id = trial.nct_id

        # Format eligibility for scoring
        pred_str = (
            f"Inclusion predictions:\n{eligibility.inclusion_explanation}\n"
            f"Exclusion predictions:\n{eligibility.exclusion_explanation}"
        )

        # Run scoring - check if it's async or not
        try:
            result = run_scoring(
                ctx.deps.clinical_note,
                trial.model_dump(exclude_none=True),
                pred_str,
                ctx.deps.llm_model
            )

            # Handle non-async result
            if not asyncio.iscoroutine(result):
                _, score_resp = result
            else:
                # If it's actually an awaitable, await it
                _, score_resp = await result

        except Exception as e:
            logger.error(f"Error calling run_scoring: {e}")
            return ScoringResult(
                trial_id=trial_id,
                relevance_explanation=f"Error in scoring: {e}",
                eligibility_explanation=f"Error in scoring: {e}"
            )

        # Extract JSON from response
        import re
        json_match = re.search(r'(\{.*\})', score_resp, re.DOTALL)

        try:
            if json_match:
                json_str = json_match.group(1)
                data = json.loads(json_str)
            else:
                data = json.loads(score_resp)

            # Map fields to our model
            return ScoringResult(
                trial_id=trial_id,
                relevance_explanation=data.get("relevance_explanation", ""),
                relevance_score=data.get("relevance_score_R", 0.0),
                eligibility_explanation=data.get("eligibility_explanation", ""),
                eligibility_score=data.get("eligibility_score_E", 0.0)
            )
        except json.JSONDecodeError:
            logger.error(f"Error parsing JSON response for {trial_id}")
            return ScoringResult(
                trial_id=trial_id,
                relevance_explanation="Error parsing response",
                eligibility_explanation="Error parsing response"
            )

    except Exception as e:
        logger.error(f"Error scoring trial {getattr(trial, 'nct_id', 'unknown')}: {e}")
        return ScoringResult(
            trial_id=getattr(trial, 'nct_id', 'UNKNOWN-ERROR') or 'UNKNOWN-ERROR'
        )


async def rank_trials(
        ctx: RunContext[PipelineConfig],
        trials: List[Trial],
        scoring_results: List[ScoringResult]
) -> List[RankedTrial]:
    """
    Rank trials based on their scores.
    """
    logger.info("Ranking trials")

    # Create a mapping of trial IDs to trial data
    trial_map = {trial.nct_id: trial for trial in trials}

    # Create ranked trials
    ranked = []

    # Sort scoring results by eligibility score (descending)
    sorted_scores = sorted(
        scoring_results,
        key=lambda x: x.eligibility_score,
        reverse=True
    )

    # Create ranked trial objects
    for score in sorted_scores:
        trial = trial_map.get(score.trial_id)
        if not trial:
            continue

        ranked.append(RankedTrial(
            trial_id=score.trial_id,
            title=trial.title,
            relevance_explanation=score.relevance_explanation,
            relevance_score=score.relevance_score,
            eligibility_explanation=score.eligibility_explanation,
            eligibility_score=score.eligibility_score
        ))

    return ranked


#######################
# ORCHESTRATOR TOOLS
#######################

async def process_batch_eligibility(
        ctx: RunContext[PipelineConfig],
        patient: PatientData,
        trials: List[Trial]
) -> List[EligibilityResult]:
    """
    Process eligibility for a batch of trials in parallel.
    """
    logger.info(f"Processing eligibility for {len(trials)} trials")

    # Filter out trials without enough information
    valid_trials = []
    for trial in trials:
        if not trial.nct_id:
            logger.warning(f"Skipping trial with missing ID: {trial.title}")
            continue
        valid_trials.append(trial)

    if not valid_trials:
        logger.warning("No valid trials found for eligibility assessment")
        return []

    eligibility_results = []
    max_workers = min(max(1, len(valid_trials)), 10)  # Limit concurrency

    # Process one by one if there are only a few trials
    if len(valid_trials) <= 3:
        for trial in valid_trials:
            try:
                result = await assess_trial_eligibility(ctx, patient, trial)
                eligibility_results.append(result)
            except Exception as e:
                logger.error(f"Failed to process eligibility for {trial.nct_id}: {e}")
                # Add a placeholder result
                eligibility_results.append(EligibilityResult(
                    trial_id=trial.nct_id,
                    inclusion_explanation=f"Error: {e}",
                    exclusion_explanation=f"Error: {e}"
                ))
        return eligibility_results

    # For more trials, use concurrent processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a list to store the futures
        futures = []

        # Submit tasks
        for trial in valid_trials:
            # Create a task that runs the eligibility assessment
            future = executor.submit(
                lambda t: asyncio.run(assess_trial_eligibility(ctx, patient, t)),
                trial
            )
            futures.append((future, trial))

        # Collect results
        for future, trial in futures:
            try:
                result = future.result()
                eligibility_results.append(result)
            except Exception as e:
                logger.error(f"Failed to process eligibility for {trial.nct_id}: {e}")
                # Add a placeholder result
                eligibility_results.append(EligibilityResult(
                    trial_id=trial.nct_id,
                    inclusion_explanation=f"Error processing trial: {e}",
                    exclusion_explanation=f"Error processing trial: {e}"
                ))

    return eligibility_results


async def process_batch_scoring(
        ctx: RunContext[PipelineConfig],
        patient: PatientData,
        trials: List[Trial],
        eligibility_results: List[EligibilityResult]
) -> List[ScoringResult]:
    """
    Process scoring for a batch of trials in parallel.
    """
    logger.info(f"Processing scoring for {len(trials)} trials")

    # Create a mapping of trial IDs to eligibility results
    eligibility_map = {result.trial_id: result for result in eligibility_results if result.trial_id}

    # Filter out trials without valid IDs
    valid_trials = [t for t in trials if t.nct_id and t.nct_id in eligibility_map]

    if not valid_trials:
        logger.warning("No valid trials found for scoring")
        return []

    scoring_results = []
    max_workers = min(max(1, len(valid_trials)), 10)  # Limit concurrency

    # Process one by one if there are only a few trials
    if len(valid_trials) <= 3:
        for trial in valid_trials:
            eligibility = eligibility_map[trial.nct_id]
            try:
                result = await score_trial(ctx, patient, trial, eligibility)
                scoring_results.append(result)
            except Exception as e:
                logger.error(f"Failed to process scoring for {trial.nct_id}: {e}")
                scoring_results.append(ScoringResult(trial_id=trial.nct_id))
        return scoring_results

    # For more trials, use concurrent processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a list to store the futures
        futures = []

        # Submit tasks
        for trial in valid_trials:
            eligibility = eligibility_map[trial.nct_id]

            # Create a task that runs the scoring
            future = executor.submit(
                lambda t, e: asyncio.run(score_trial(ctx, patient, t, e)),
                trial, eligibility
            )
            futures.append((future, trial))

        # Collect results
        for future, trial in futures:
            try:
                result = future.result()
                scoring_results.append(result)
            except Exception as e:
                logger.error(f"Failed to process scoring for {trial.nct_id}: {e}")
                scoring_results.append(ScoringResult(trial_id=trial.nct_id))

    return scoring_results


#######################
# MAIN PIPELINE
#######################

async def run_clinical_trial_pipeline(
        ctx: RunContext[PipelineConfig]
) -> PipelineResult:
    """
    Run the complete clinical trial matching pipeline.

    This orchestrates the entire process from extraction to ranking.
    """
    logger.info("Starting clinical trial matching pipeline")

    # Extract patient data
    patient_data = await extract_patient_data(ctx)
    logger.info("✅ Completed patient data extraction")

    # Search for clinical trials
    trials = await search_clinical_trials(ctx, patient_data)
    logger.info(f"✅ Retrieved {len(trials)} clinical trials")

    if not trials:
        logger.warning("No trials found - returning early")
        return PipelineResult(patient_data=patient_data)

    # Process eligibility in batches
    eligibility_results = await process_batch_eligibility(ctx, patient_data, trials)
    logger.info(f"✅ Completed eligibility assessment for {len(eligibility_results)} trials")

    # Process scoring in batches
    scoring_results = await process_batch_scoring(ctx, patient_data, trials, eligibility_results)
    logger.info(f"✅ Completed scoring for {len(scoring_results)} trials")

    # Rank trials
    ranked_trials = await rank_trials(ctx, trials, scoring_results)
    logger.info(f"✅ Ranked {len(ranked_trials)} trials")

    # Return the complete pipeline result
    return PipelineResult(
        patient_data=patient_data,
        retrieved_trials=trials,
        eligibility_results=eligibility_results,
        scoring_results=scoring_results,
        ranked_trials=ranked_trials
    )


#######################
# AGENT SETUP
#######################

def get_pipeline_agent(llm_model: str) -> Agent:
    """
    Create a Pydantic AI agent for clinical trial matching.
    """
    logger.info(f"Creating agent with model: {llm_model}")

    # Create the agent with all relevant tools
    return Agent(
        llm_model,
        deps_type=PipelineConfig,
        output_type=PipelineResult,
        tools=[
            # Atomic tools
            extract_patient_data,
            search_clinical_trials,
            assess_trial_eligibility,
            score_trial,
            rank_trials,

            # Batch processing tools
            process_batch_eligibility,
            process_batch_scoring,

            # Main pipeline tool
            run_clinical_trial_pipeline
        ],
        system_prompt=(
            "You are a clinical trial matching assistant. "
            "Your role is to extract patient information from clinical notes, "
            "retrieve relevant trials, assess eligibility, and rank matches."
        )
    )


#######################
# MAIN ENTRY POINT
#######################

def run_pydantic_agent(
        presentation: str,
        llm_model: str,
        recruiting_status: str,
        min_date: date,
        max_date: date,
        phase: str,
) -> Dict[str, Any]:
    """
    Run the clinical trial matching pipeline using a Pydantic AI agent.

    Args:
        presentation: Clinical note text
        llm_model: LLM model to use
        recruiting_status: Trial recruiting status filter
        min_date: Minimum trial date
        max_date: Maximum trial date
        phase: Trial phase filter

    Returns:
        Dictionary with the pipeline results in a backward-compatible format
    """
    # Create pipeline configuration
    config = PipelineConfig(
        clinical_note=presentation,
        llm_model=llm_model,
        recruiting_status=recruiting_status,
        min_date=min_date,
        max_date=max_date,
        phase=phase
    )

    # Create and run the agent
    agent = get_pipeline_agent(config.llm_model)

    try:
        # Run the agent asynchronously
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # No running event loop, create one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    try:
        # Run the pipeline
        logger.info("Running pipeline agent")
        result = loop.run_until_complete(
            agent.run(
                "Run the clinical trial matching pipeline for the given patient.",
                deps=config
            )
        )

        # Convert to output format expected by the legacy code
        if hasattr(result, 'output') and isinstance(result.output, PipelineResult):
            pipeline_result = result.output
        else:
            pipeline_result = result

        # Use the shared response formatter
        return format_response_for_ui(
            model_name=llm_model,
            patient_data=pipeline_result.patient_data,
            retrieved_trials=pipeline_result.retrieved_trials,
            eligibility_results=pipeline_result.eligibility_results,
            scoring_results=pipeline_result.scoring_results,
            config=config
        )

    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        return {
            "error": str(e),
            "model": llm_model
        }
