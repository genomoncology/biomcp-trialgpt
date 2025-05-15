import asyncio
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from functools import wraps
from typing import Dict, Any, List, Optional, Union, Tuple, TypeVar, Callable, Type

from pydantic import BaseModel, Field, model_validator, field_validator
from pydantic_ai import Agent, RunContext

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import necessary components
from .biomcp_client import BioMCPClient
from .note_extractor import parse_clinical_note
from .eligibility import run_eligibility
from .scoring import run_scoring
from .response_formatter import format_response_for_ui

# Type variables for generic functions
T = TypeVar('T')
R = TypeVar('R')


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
        return cls(**(data.get("values", {}) if "values" in data else data))


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
            # Get ID or generate fallback
            nct_id = data.get('nct_id') or data.get('NCT_Number') or data.get('NCT Number')
            if not nct_id:
                if data.get('title'):
                    import hashlib
                    nct_id = f"UNKNOWN-{hashlib.md5(data['title'].encode()).hexdigest()[:8]}"
                else:
                    import uuid
                    nct_id = f"UNKNOWN-{str(uuid.uuid4())[:8]}"
            data['nct_id'] = nct_id
        return data

    @field_validator('nct_id', mode='before')
    @classmethod
    def normalize_nct_id(cls, v, info):
        """Handle different NCT ID formats."""
        return v or info.data.get("NCT_Number") or info.data.get("NCT Number")

    @field_validator('conditions', 'interventions', mode='before')
    @classmethod
    def parse_delimited_lists(cls, v):
        """Parse pipe-delimited strings into lists."""
        return [item.strip() for item in v.split("|") if item.strip()] if isinstance(v, str) else v


class EligibilityResult(BaseModel):
    """Assessment of a patient's eligibility for a trial."""
    trial_id: str
    inclusion_decision: str = ""
    inclusion_explanation: str = ""
    exclusion_decision: str = ""
    exclusion_explanation: str = ""

    @field_validator('inclusion_decision', 'inclusion_explanation', 'exclusion_decision', 'exclusion_explanation',
                     mode='before')
    @classmethod
    def parse_tuple_results(cls, v, info):
        """Handle tuple format from legacy code."""
        if isinstance(v, tuple) and len(v) == 2:
            return v[1] if info.field_name.endswith('explanation') else v[0]
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


class PipelineConfig(BaseModel):
    """Configuration for the clinical trial matching pipeline."""
    clinical_note: str
    llm_model: str
    recruiting_status: str
    min_date: Union[date, str, int] = Field(default_factory=lambda: date(2018, 1, 1))
    max_date: Union[date, str, int] = None
    phase: str

    @model_validator(mode='before')
    @classmethod
    def process_dates(cls, data):
        """Handle different date formats."""
        if not isinstance(data, dict):
            return data

        # Format min_date
        min_date = data.get('min_date')
        if min_date is None:
            data['min_date'] = "2018-01-01"
        elif isinstance(min_date, int):
            data['min_date'] = f"{min_date}-01-01"
        elif isinstance(min_date, date):
            data['min_date'] = min_date.strftime("%Y-%m-%d")

        # Format max_date
        max_date = data.get('max_date')
        if max_date is None:
            data['max_date'] = f"{datetime.now().year + 1}-12-31"
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
        if not any(model.startswith(p) for p in ('openai:', 'google-gla:', 'anthropic:')):
            if "gpt" in model.lower():
                model = f"openai:{model}"
            elif "gemini" in model.lower():
                model = f"google-gla:{model}"
            elif "claude" in model.lower():
                model = f"anthropic:{model}"

        logger.info(f"Using model: {model}")
        return model


#######################
# HELPER UTILITIES
#######################

def error_handler(default_value, error_msg_prefix="Error"):
    """Decorator for handling errors in async functions with consistent logging."""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.error(f"{error_msg_prefix}: {e}")
                return default_value

        return wrapper

    return decorator


async def process_batch(
        items: List[T],
        process_fn: Callable[[T], R],
        max_workers: int = 10,
        process_singles: bool = True,
        desc: str = "Processing batch"
) -> List[R]:
    """Generic batch processor for concurrent operations."""
    logger.info(f"{desc}: {len(items)} items")

    if not items:
        logger.warning(f"No items to process in {desc}")
        return []

    # Process sequentially if small batch or requested
    if len(items) <= 3 and process_singles:
        results = []
        for item in items:
            try:
                result = await process_fn(item) if asyncio.iscoroutinefunction(process_fn) else process_fn(item)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing item: {e}")
        return results

    # Process concurrently for larger batches
    actual_max_workers = min(max_workers, len(items))
    results = []

    with ThreadPoolExecutor(max_workers=actual_max_workers) as executor:
        # Setup futures
        if asyncio.iscoroutinefunction(process_fn):
            # For async functions
            futures = [(executor.submit(
                lambda i: asyncio.run(process_fn(i)), item), i)
                for i, item in enumerate(items)]
        else:
            # For sync functions
            futures = [(executor.submit(process_fn, item), i)
                       for i, item in enumerate(items)]

        # Collect results while preserving order
        results_map = {}
        for future, idx in futures:
            try:
                results_map[idx] = future.result()
            except Exception as e:
                logger.error(f"Error in batch processing item {idx}: {e}")
                results_map[idx] = None

        # Convert back to ordered list
        results = [results_map.get(i) for i in range(len(items))]
        # Filter out Nones
        results = [r for r in results if r is not None]

    return results


#######################
# ATOMIC TOOLS
#######################

@error_handler(PatientData())
async def extract_patient_data(ctx: RunContext[PipelineConfig]) -> PatientData:
    """Extract structured patient data from clinical notes."""
    logger.info(f"Extracting patient information using model {ctx.deps.llm_model}")
    data_dict, _, _ = parse_clinical_note(ctx.deps.clinical_note, ctx.deps.llm_model)
    return PatientData.from_dict(data_dict)


@error_handler([])
async def search_clinical_trials(ctx: RunContext[PipelineConfig], patient: PatientData) -> List[Trial]:
    """Search for clinical trials based on patient data and filters."""
    logger.info("Searching for clinical trials")

    filter_params = {
        "recruiting_status": ctx.deps.recruiting_status,
        "min_date": ctx.deps.min_date,
        "max_date": ctx.deps.max_date,
        "phase": ctx.deps.phase,
        "conditions": patient.conditions,
        "terms": patient.terms,
        "interventions": patient.interventions
    }

    logger.info(f"Set up trial retrieval with parameters: {json.dumps(filter_params, indent=2)}")
    trials_data = await BioMCPClient().retrieve_trials(**filter_params)
    logger.info(f"Retrieved {len(trials_data)} trials")

    def transform_trial_data(data: dict) -> dict:
        """Transform BioMCP API trial data to match Trial model fields."""
        transformed = {"nct_id": data.get("NCT Number"), "title": data.get("Study Title"),
                       "brief_summary": data.get("Brief Summary"), "detailed_description": data.get("Study Design"),
                       "status": data.get("Study Status"), "phase": data.get("Phases"),
                       "study_start_date": data.get("Start Date"), "study_completion_date": data.get("Completion Date"),
                       "conditions": data.get("Conditions", ""), "interventions": data.get("Interventions", ""),
                       "enrollment": data.get("Enrollment"), "study_type": data.get("Study Type"),
                       "study_url": data.get("Study URL"), "study_results": data.get("Study Results"), "locations": [],
                       "sponsors": [], "eligibility_criteria": None, "eligibility_gender": None, "minimum_age": None,
                       "maximum_age": None}
        return transformed

    # Process all trials into Trial objects
    return [Trial.model_validate(transform_trial_data(t)) for t in trials_data if isinstance(t, dict)]


@error_handler(None)
async def assess_trial_eligibility(ctx: RunContext[PipelineConfig], patient: PatientData,
                                   trial: Trial) -> EligibilityResult:
    """Assess if a patient is eligible for a specific trial."""
    trial_id = trial.nct_id or "UNKNOWN-TRIAL"
    logger.info(f"Assessing eligibility for trial {trial_id}")

    result = run_eligibility(ctx.deps.clinical_note, trial.model_dump(exclude_none=True), ctx.deps.llm_model)

    # Handle non-async vs async result
    if not asyncio.iscoroutine(result):
        inclusion = result.get("inclusion", ("", ""))
        exclusion = result.get("exclusion", ("", ""))
    else:
        result_data = await result
        inclusion = result_data.get("inclusion", ("", ""))
        exclusion = result_data.get("exclusion", ("", ""))

    return EligibilityResult(
        trial_id=trial_id,
        inclusion_decision=inclusion[0] if isinstance(inclusion, tuple) else "",
        inclusion_explanation=inclusion[1] if isinstance(inclusion, tuple) else str(inclusion),
        exclusion_decision=exclusion[0] if isinstance(exclusion, tuple) else "",
        exclusion_explanation=exclusion[1] if isinstance(exclusion, tuple) else str(exclusion)
    )


@error_handler(None)
async def score_trial(ctx: RunContext[PipelineConfig], patient: PatientData, trial: Trial,
                      eligibility: EligibilityResult) -> ScoringResult:
    """Score a trial based on its relevance and eligibility for the patient."""
    trial_id = trial.nct_id or eligibility.trial_id or "UNKNOWN-TRIAL"
    logger.info(f"Scoring trial {trial_id}")

    # Format eligibility predictions
    pred_str = f"Inclusion predictions:\n{eligibility.inclusion_explanation}\nExclusion predictions:\n{eligibility.exclusion_explanation}"

    # Run scoring
    result = run_scoring(ctx.deps.clinical_note, trial.model_dump(exclude_none=True), pred_str, ctx.deps.llm_model)

    # Handle async vs non-async result
    if not asyncio.iscoroutine(result):
        _, score_resp = result
    else:
        _, score_resp = await result

    # Extract JSON from response
    try:
        json_match = re.search(r'(\{.*\})', score_resp, re.DOTALL)
        data = json.loads(json_match.group(1) if json_match else score_resp)
    except:
        data = {
            "relevance_explanation": "Error parsing response",
            "relevance_score_R": 0,
            "eligibility_explanation": "Error parsing response",
            "eligibility_score_E": 0
        }

    return ScoringResult(
        trial_id=trial_id,
        relevance_explanation=data.get("relevance_explanation", ""),
        relevance_score=data.get("relevance_score_R", 0.0),
        eligibility_explanation=data.get("eligibility_explanation", ""),
        eligibility_score=data.get("eligibility_score_E", 0.0)
    )


async def rank_trials(ctx: RunContext[PipelineConfig], trials: List[Trial], scoring_results: List[ScoringResult]) -> \
List[RankedTrial]:
    """Rank trials based on their scores."""
    logger.info("Ranking trials")

    # Map trials by ID for quick lookup
    trial_map = {t.nct_id: t for t in trials if t.nct_id}

    # Sort and create ranked trials
    return [
        RankedTrial(
            trial_id=score.trial_id,
            title=trial_map.get(score.trial_id, Trial()).title,
            relevance_explanation=score.relevance_explanation,
            relevance_score=score.relevance_score,
            eligibility_explanation=score.eligibility_explanation,
            eligibility_score=score.eligibility_score
        )
        for score in sorted(
            scoring_results,
            key=lambda x: x.eligibility_score,
            reverse=True
        )
        if score.trial_id in trial_map
    ]


#######################
# BATCH PROCESSING
#######################

async def process_batch_eligibility(ctx: RunContext[PipelineConfig], patient: PatientData, trials: List[Trial]) -> List[
    EligibilityResult]:
    """Process eligibility for a batch of trials in parallel."""
    valid_trials = [t for t in trials if t.nct_id]

    async def process_single_trial(trial):
        return await assess_trial_eligibility(ctx, patient, trial)

    return await process_batch(
        valid_trials,
        process_single_trial,
        max_workers=10,
        desc="Processing eligibility assessment"
    )


async def process_batch_scoring(ctx: RunContext[PipelineConfig], patient: PatientData, trials: List[Trial],
                                eligibility_results: List[EligibilityResult]) -> List[ScoringResult]:
    """Process scoring for a batch of trials in parallel."""
    # Map eligibility results by trial ID
    eligibility_map = {result.trial_id: result for result in eligibility_results if result.trial_id}

    # Filter valid trials that have eligibility results
    valid_trials = [t for t in trials if t.nct_id and t.nct_id in eligibility_map]

    async def process_single_trial(trial):
        eligibility = eligibility_map[trial.nct_id]
        return await score_trial(ctx, patient, trial, eligibility)

    return await process_batch(
        valid_trials,
        process_single_trial,
        max_workers=10,
        desc="Processing trial scoring"
    )


#######################
# MAIN PIPELINE
#######################

async def run_clinical_trial_pipeline(ctx: RunContext[PipelineConfig]) -> PipelineResult:
    """Run the complete clinical trial matching pipeline."""
    logger.info("Starting clinical trial matching pipeline")

    # Extract patient data
    patient_data = await extract_patient_data(ctx)
    logger.info("✅ Completed patient data extraction")

    # Search for clinical trials
    trials = await search_clinical_trials(ctx, patient_data)
    logger.info(f"✅ Retrieved {len(trials)} clinical trials")

    if not trials:
        return PipelineResult(patient_data=patient_data)

    # Process eligibility and scoring
    eligibility_results = await process_batch_eligibility(ctx, patient_data, trials)
    logger.info(f"✅ Completed eligibility assessment for {len(eligibility_results)} trials")

    scoring_results = await process_batch_scoring(ctx, patient_data, trials, eligibility_results)
    logger.info(f"✅ Completed scoring for {len(scoring_results)} trials")

    # Rank trials
    ranked_trials = await rank_trials(ctx, trials, scoring_results)
    logger.info(f"✅ Ranked {len(ranked_trials)} trials")

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
    """Create a Pydantic AI agent for clinical trial matching."""
    logger.info(f"Creating agent with model: {llm_model}")

    return Agent(
        llm_model.replace('gpt-', ''),
        deps_type=PipelineConfig,
        output_type=PipelineResult,
        tools=[
            # Core tools
            extract_patient_data,
            search_clinical_trials,
            assess_trial_eligibility,
            score_trial,
            rank_trials,

            # Batch processing
            process_batch_eligibility,
            process_batch_scoring,

            # Main pipeline
            run_clinical_trial_pipeline
        ],
        system_prompt="You are a clinical trial matching assistant. Your role is to extract patient information from clinical notes, retrieve relevant trials, assess eligibility, and rank matches."
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
    """Run the clinical trial matching pipeline using a Pydantic AI agent."""
    # Create configuration
    config = PipelineConfig(
        clinical_note=presentation,
        llm_model=llm_model,
        recruiting_status=recruiting_status,
        min_date=min_date,
        max_date=max_date,
        phase=phase
    )

    # Setup and run agent
    agent = get_pipeline_agent(config.llm_model)

    try:
        # Get or create event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Run pipeline
        logger.info("Running pipeline agent")
        result = loop.run_until_complete(
            agent.run("Run the clinical trial matching pipeline for the given patient.", deps=config)
        )

        # Extract result
        pipeline_result = result.output if hasattr(result, 'output') else result

        # Format for UI
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
        return {"error": str(e), "model": llm_model}