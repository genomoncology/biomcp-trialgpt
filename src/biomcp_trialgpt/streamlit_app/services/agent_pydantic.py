from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import date, datetime
from pydantic import BaseModel, Field, validator  # Using validator instead of field_validator for v1 compatibility
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
import json
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

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


# Define explicit patient data fields instead of using a generic dictionary
class PatientDataFields(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    diagnosis: Optional[str] = None
    conditions: List[str] = Field(default_factory=list)
    terms: List[str] = Field(default_factory=list)
    interventions: List[str] = Field(default_factory=list)
    medical_history: Optional[List[str]] = Field(default_factory=list)
    medications: Optional[List[str]] = Field(default_factory=list)
    allergies: Optional[List[str]] = Field(default_factory=list)

    class Config:
        extra = "ignore"


class DataModel(BaseModel):
    """A structured data model for patient information."""
    values: PatientDataFields = Field(
        default_factory=PatientDataFields,
        description="Patient data fields"
    )

    class Config:
        title = "Data Dictionary"
        json_schema_extra = {
            "description": "A structured wrapper for patient data"
        }

    @classmethod
    def create_from_dict(cls, data: Dict[str, Any]) -> "DataModel":
        """Create a DataModel from a dictionary."""
        # Handle the case where data might already have a 'values' key
        if "values" in data and isinstance(data["values"], dict):
            return cls(values=PatientDataFields(**data["values"]))
        # Otherwise, treat the entire dict as values
        return cls(values=PatientDataFields(**data))

    def to_dict(self) -> Dict[str, Any]:
        """Convert back to a plain dictionary."""
        return self.values.dict(exclude_none=True)

    def model_dump(self) -> Dict[str, Any]:
        """Alternative method for model dumping (for compatibility)."""
        try:
            return {"values": self.values.dict(exclude_none=True)}
        except Exception as e:
            logger.error(f"Error in DataModel.model_dump: {e}")
            return {"values": {}}


# Define explicit trial properties instead of using a generic dictionary
class TrialProperties(BaseModel):
    title: Optional[str] = None
    brief_summary: Optional[str] = None
    detailed_description: Optional[str] = None
    nct_id: Optional[str] = None
    NCT_Number: Optional[str] = None  # Some code uses this format
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


class TrialData(BaseModel):
    """Structured model for trial data."""
    properties: TrialProperties = Field(
        default_factory=TrialProperties,
        description="Properties of the trial"
    )

    class Config:
        title = "Trial Data"
        json_schema_extra = {
            "description": "Data for a clinical trial"
        }

    @classmethod
    def create_from_dict(cls, data: Dict[str, Any]) -> "TrialData":
        """Create a TrialData from a dictionary."""
        # Debug the input data structure
        logger.debug(f"Creating TrialData from: {json.dumps(data, indent=2, default=str)[:200]}...")

        # Handle the case where data might already have a 'properties' key
        if "properties" in data and isinstance(data["properties"], dict):
            properties_data = data["properties"]
        else:
            # Process the flat structure
            properties_data = {}
            for key, value in data.items():
                # Convert keys to match our model (e.g., NCT_Number -> nct_id)
                prop_key = key.lower()
                if key == "NCT Number":
                    prop_key = "nct_id"
                elif key == "Study Title":
                    prop_key = "title"
                elif key == "Brief Summary":
                    prop_key = "brief_summary"
                elif key == "Study Status":
                    prop_key = "status"
                elif key == "Phases":
                    prop_key = "phase"
                elif key == "Start Date":
                    prop_key = "study_start_date"
                elif key == "Completion Date":
                    prop_key = "study_completion_date"
                elif key == "Conditions":
                    prop_key = "conditions"
                    # Convert pipe-delimited string to list
                    if isinstance(value, str):
                        value = [item.strip() for item in value.split("|") if item.strip()]
                elif key == "Interventions":
                    prop_key = "interventions"
                    # Convert pipe-delimited string to list
                    if isinstance(value, str):
                        value = [item.strip() for item in value.split("|") if item.strip()]

                properties_data[prop_key] = value

        # Create the TrialProperties with the processed data
        try:
            trial_properties = TrialProperties(**properties_data)
            return cls(properties=trial_properties)
        except Exception as e:
            logger.error(f"Error creating TrialProperties: {e}")
            # Try to create a minimal valid object with just the essential fields
            minimal_data = {
                "title": data.get("Study Title", "Unknown Title"),
                "nct_id": data.get("NCT Number", "Unknown ID"),
                "conditions": [],
                "interventions": []
            }
            if "Conditions" in data and isinstance(data["Conditions"], str):
                minimal_data["conditions"] = [item.strip() for item in data["Conditions"].split("|") if item.strip()]
            if "Interventions" in data and isinstance(data["Interventions"], str):
                minimal_data["interventions"] = [item.strip() for item in data["Interventions"].split("|") if
                                                 item.strip()]

            return cls(properties=TrialProperties(**minimal_data))

    def to_dict(self) -> Dict[str, Any]:
        """Convert back to a plain dictionary."""
        return self.properties.dict(exclude_none=True)

    def model_dump(self) -> Dict[str, Any]:
        """Alternative method for model dumping (for compatibility)."""
        try:
            return {"properties": self.properties.dict(exclude_none=True)}
        except Exception as e:
            logger.error(f"Error in TrialData.model_dump: {e}")
            return {"properties": {}}


# Define structured models for the four pipeline steps
class Step1Output(BaseModel):
    """Patient information extraction output."""
    model: str
    prompt: str
    response: str
    data: DataModel

    class Config:
        title = "Patient Information Extraction Output"
        json_schema_extra = {
            "description": "Results from extracting patient information from clinical notes"
        }

    def model_dump(self) -> Dict[str, Any]:
        """Convert to a dictionary."""
        try:
            return {
                "model": self.model,
                "prompt": self.prompt,
                "response": self.response,
                "data": self.data.model_dump() if hasattr(self.data, 'model_dump') else
                (self.data.dict() if hasattr(self.data, 'dict') else {})
            }
        except Exception as e:
            logger.error(f"Error in Step1Output.model_dump: {e}")
            return {
                "model": self.model,
                "prompt": self.prompt,
                "response": self.response,
                "data": {}
            }


class Step2Params(BaseModel):
    """Parameters for retrieving clinical trials."""
    recruiting_status: str
    min_date: str
    max_date: str
    phase: str
    conditions: List[str]
    terms: List[str]
    interventions: List[str]

    class Config:
        title = "Trial Retrieval Parameters"
        json_schema_extra = {
            "description": "Parameters used for retrieving clinical trials"
        }


class Step2Output(BaseModel):
    """Trial retrieval output."""
    params: Step2Params
    response: List[TrialData]

    class Config:
        title = "Trial Retrieval Output"
        json_schema_extra = {
            "description": "Results from retrieving clinical trials"
        }

    def model_dump(self) -> Dict[str, Any]:
        """Convert to a dictionary."""
        try:
            return {
                "params": self.params.dict() if hasattr(self.params, 'dict') else {},
                "response": [
                    trial.model_dump() if hasattr(trial, 'model_dump') else
                    (trial.dict() if hasattr(trial, 'dict') else {})
                    for trial in self.response
                ]
            }
        except Exception as e:
            logger.error(f"Error in Step2Output.model_dump: {e}")
            return {
                "params": {},
                "response": []
            }


# Define structured models for eligibility and scoring results
class EligibilityResult(BaseModel):
    """Result of eligibility assessment."""
    inclusion: Tuple[str, str] = Field(default=("", ""))
    exclusion: Tuple[str, str] = Field(default=("", ""))


class TrialEligibility(BaseModel):
    """Trial eligibility with trial ID."""
    trial_id: str
    eligibility: EligibilityResult


class Step3Output(BaseModel):
    """Eligibility assessment results."""
    results: List[TrialEligibility] = Field(default_factory=list)

    @validator('results', pre=True)
    def validate_results(cls, v):
        """Convert dictionary of eligibility results to list of TrialEligibility objects."""
        if isinstance(v, dict):
            return [
                TrialEligibility(
                    trial_id=key,
                    eligibility=EligibilityResult(
                        inclusion=value["inclusion"] if isinstance(value, dict) and "inclusion" in value else ("", ""),
                        exclusion=value["exclusion"] if isinstance(value, dict) and "exclusion" in value else ("", "")
                    )
                )
                for key, value in v.items()
            ]
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        try:
            return {item.trial_id: {"inclusion": item.eligibility.inclusion, "exclusion": item.eligibility.exclusion}
                    for item in self.results}
        except Exception as e:
            logger.error(f"Error in Step3Output.to_dict: {e}")
            return {}

    def model_dump(self) -> Dict[str, Any]:
        """Alternative method for model dumping (for compatibility)."""
        try:
            return {"results": [
                {
                    "trial_id": item.trial_id,
                    "eligibility": {
                        "inclusion": item.eligibility.inclusion,
                        "exclusion": item.eligibility.exclusion
                    }
                }
                for item in self.results
            ]}
        except Exception as e:
            logger.error(f"Error in Step3Output.model_dump: {e}")
            return {"results": []}


class ScoringResult(BaseModel):
    """Result of trial scoring."""
    relevance_explanation: str = ""
    relevance_score_R: float = 0.0
    eligibility_explanation: str = ""
    eligibility_score_E: float = 0.0
    raw_response: Optional[str] = None


class TrialScoring(BaseModel):
    """Trial scoring with trial ID."""
    trial_id: str
    scoring: ScoringResult


class RankedTrialResult(BaseModel):
    """Flat structure for ranked trial results."""
    trial_id: str = ""
    relevance_explanation: str = ""
    relevance_score_R: float = 0.0
    eligibility_explanation: str = ""
    eligibility_score_E: float = 0.0


class Step4Output(BaseModel):
    """Scoring and ranking results."""
    scoring_logs: List[TrialScoring] = Field(default_factory=list)
    ranked: List[RankedTrialResult] = Field(default_factory=list)

    @validator('scoring_logs', pre=True)
    def validate_scoring_logs(cls, v):
        """Convert dictionary of scoring results to list of TrialScoring objects."""
        if isinstance(v, dict):
            return [
                TrialScoring(
                    trial_id=key,
                    scoring=ScoringResult(**value) if isinstance(value, dict) else ScoringResult()
                )
                for key, value in v.items()
            ]
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        try:
            return {
                "scoring_logs": {item.trial_id: item.scoring.model_dump()
                if hasattr(item.scoring, 'model_dump')
                else vars(item.scoring)
                                 for item in self.scoring_logs},
                "ranked": [(item.trial_id, item.model_dump() if hasattr(item, 'model_dump') else vars(item))
                           for item in self.ranked]
            }
        except Exception as e:
            logger.error(f"Error in Step4Output.to_dict: {e}")
            return {"scoring_logs": {}, "ranked": []}

    def model_dump(self) -> Dict[str, Any]:
        """Alternative method for model dumping (for compatibility)."""
        try:
            return {
                "scoring_logs": {
                    item.trial_id: item.scoring.dict() if hasattr(item.scoring, 'dict') else {}
                    for item in self.scoring_logs
                },
                "ranked": [
                    {
                        "trial_id": item.trial_id,
                        "relevance_explanation": item.relevance_explanation,
                        "relevance_score_R": item.relevance_score_R,
                        "eligibility_explanation": item.eligibility_explanation,
                        "eligibility_score_E": item.eligibility_score_E
                    }
                    for item in self.ranked
                ]
            }
        except Exception as e:
            logger.error(f"Error in Step4Output.model_dump: {e}")
            return {"scoring_logs": {}, "ranked": []}


# Define the final pipeline output
class PipelineOutput(BaseModel):
    """Complete output from the clinical trial matching pipeline."""
    step1: Optional[Step1Output] = None
    step2: Optional[Step2Output] = None
    step3: Optional[Step3Output] = None
    step4: Optional[Step4Output] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary for easy consumption by the Streamlit UI."""
        result = {}

        # Handle step1 - check for None and handle conversion
        if self.step1:
            try:
                result["step1"] = self.step1.model_dump() if hasattr(self.step1, 'model_dump') else vars(self.step1)
            except Exception as e:
                logger.error(f"Error converting step1 to dict: {e}")
                result["step1"] = {"error": str(e)}
        else:
            result["step1"] = None

        # Handle step2 - check for None and handle conversion
        if self.step2:
            try:
                result["step2"] = self.step2.model_dump() if hasattr(self.step2, 'model_dump') else vars(self.step2)
            except Exception as e:
                logger.error(f"Error converting step2 to dict: {e}")
                result["step2"] = {"error": str(e)}
        else:
            result["step2"] = None

        # Handle step3 - check for None and handle conversion
        if self.step3:
            try:
                result["step3"] = self.step3.to_dict() if hasattr(self.step3, 'to_dict') else vars(self.step3)
            except Exception as e:
                logger.error(f"Error converting step3 to dict: {e}")
                result["step3"] = {"error": str(e)}
        else:
            result["step3"] = None

        # Handle step4 - check for None and handle conversion
        if self.step4:
            try:
                result["step4"] = self.step4.to_dict() if hasattr(self.step4, 'to_dict') else vars(self.step4)
            except Exception as e:
                logger.error(f"Error converting step4 to dict: {e}")
                result["step4"] = {"error": str(e)}
        else:
            result["step4"] = None

        return result


# Define the pipeline dependencies
@dataclass
class PipelineDependencies:
    """Dependencies for the clinical trial matching pipeline."""
    presentation: str  # Clinical note text
    llm_model: str  # LLM model to use
    recruiting_status: str  # Trial recruiting status filter
    min_date: date  # Minimum trial date
    max_date: date  # Maximum trial date
    phase: str  # Trial phase filter


# Define the tool functions that will be used by the agent
async def extract_patient_information(ctx: RunContext[PipelineDependencies]) -> Step1Output:
    """
    Extract patient information from clinical notes.

    This tool parses the clinical note text to extract structured patient data.
    """
    logger.info(f"Extracting patient information from clinical note using model {ctx.deps.llm_model}")

    # Call the original parse_clinical_note function
    data, prompt, response = parse_clinical_note(ctx.deps.presentation, ctx.deps.llm_model)
    logger.debug(f"Extracted data: {data}")

    # Convert data dict to DataModel with explicit fields
    data_model = DataModel.create_from_dict(data)

    return Step1Output(
        model=ctx.deps.llm_model,
        prompt=prompt,
        response=response,
        data=data_model
    )


async def retrieve_trials(
        ctx: RunContext[PipelineDependencies],
        patient_data: Step1Output
) -> Step2Output:
    """
    Retrieve clinical trials based on patient information.

    This tool queries the BioMCP API to find relevant trials.
    """
    logger.info("Retrieving clinical trials")
    client = BioMCPClient()

    # Handle different date formats and None values
    if ctx.deps.min_date is None:
        min_date_str = "2018-01-01"  # Default to 2018
    elif isinstance(ctx.deps.min_date, int):
        min_date_str = f"{ctx.deps.min_date}-01-01"
    else:
        min_date_str = ctx.deps.min_date.strftime("%Y-%m-%d")

    if ctx.deps.max_date is None:
        current_year = datetime.now().year
        max_date_str = f"{current_year + 1}-12-31"  # Default to next year
    elif isinstance(ctx.deps.max_date, int):
        max_date_str = f"{ctx.deps.max_date}-12-31"
    else:
        max_date_str = ctx.deps.max_date.strftime("%Y-%m-%d")

    # Access data values from DataModel
    patient_fields = patient_data.data.values

    # Create parameters for trial retrieval
    params = Step2Params(
        recruiting_status=ctx.deps.recruiting_status,
        min_date=min_date_str,
        max_date=max_date_str,
        phase=ctx.deps.phase,
        conditions=patient_fields.conditions,
        terms=patient_fields.terms,
        interventions=patient_fields.interventions
    )

    # Convert to dictionary for API call
    params_dict = params.model_dump()

    # Get trials
    logger.info(f"Retrieving trials with parameters: {params_dict}")
    trials_data = await client.retrieve_trials(**params_dict)
    logger.info(f"Retrieved {len(trials_data)} trials")

    # Convert trial dictionaries to TrialData objects
    trial_objects = []
    for trial in trials_data:
        try:
            trial_obj = TrialData.create_from_dict(trial)
            trial_objects.append(trial_obj)
        except Exception as e:
            logger.error(f"Error converting trial to TrialData: {e}")
            # Create an empty trial object to maintain the count
            trial_objects.append(TrialData())

    return Step2Output(
        params=params,
        response=trial_objects
    )


async def assess_eligibility(
        ctx: RunContext[PipelineDependencies],
        patient_data: Step1Output,
        trials: Step2Output
) -> Step3Output:
    """
    Assess patient eligibility for the retrieved trials.

    This tool evaluates each trial's inclusion/exclusion criteria against the patient data.
    """
    logger.info("Assessing eligibility for trials")
    eligibility_results = {}
    max_workers = max(1, min(len(trials.response), 5))

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_key = {}
        for t in trials.response:
            t_dict = t.to_dict()
            key = t_dict.get("NCT_Number") or t_dict.get("nct_id", "")
            if not key:
                continue

            future = executor.submit(run_eligibility, ctx.deps.presentation, t_dict, ctx.deps.llm_model)
            future_to_key[future] = key

        for future in as_completed(future_to_key):
            key = future_to_key[future]
            try:
                result = future.result()
                eligibility_results[key] = result
            except Exception as e:
                logger.error(f"Error in eligibility assessment for trial {key}: {str(e)}")
                eligibility_results[key] = {
                    "inclusion": ("", f"Error: {e}"),
                    "exclusion": ("", f"Error: {e}")
                }

    logger.info(f"Completed eligibility assessment for {len(eligibility_results)} trials")
    return Step3Output(results=eligibility_results)


async def score_and_rank_trials(
        ctx: RunContext[PipelineDependencies],
        patient_data: Step1Output,
        trials: Step2Output,
        eligibility_results: Step3Output
) -> Step4Output:
    """
    Score and rank trials based on eligibility assessment.

    This tool calculates relevance and eligibility scores for each trial
    and ranks them accordingly.
    """
    logger.info("Scoring and ranking trials")

    # Check if eligibility_results contains valid data
    if not hasattr(eligibility_results, 'results') or not eligibility_results.results:
        logger.warning("Invalid eligibility_results provided")
        return Step4Output(scoring_logs=[], ranked=[])

    # Create a lookup map from trial ID to eligibility result
    eligibility_map = {}
    if isinstance(eligibility_results.results, list):
        for item in eligibility_results.results:
            if hasattr(item, 'trial_id') and hasattr(item, 'eligibility'):
                eligibility_map[item.trial_id] = item.eligibility
    elif isinstance(eligibility_results, dict):
        # Handle dictionary format for backward compatibility
        for key, value in eligibility_results.items():
            eligibility_map[key] = value

    scoring_dict = {}
    max_workers = max(1, min(len(trials.response), 5))

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_key = {}
        for trial in trials.response:
            trial_dict = trial.to_dict()
            key = trial_dict.get("nct_id") or trial_dict.get("NCT_Number", "")
            if not key or key not in eligibility_map:
                continue

            eligibility = eligibility_map[key]

            # Format the eligibility predictions for scoring
            if hasattr(eligibility, 'inclusion') and hasattr(eligibility, 'exclusion'):
                inc_result = eligibility.inclusion
                exc_result = eligibility.exclusion
            else:
                # Handle dictionary format
                inc_result = eligibility.get("inclusion", ("", ""))
                exc_result = eligibility.get("exclusion", ("", ""))

            # Handle both tuple and non-tuple formats
            if isinstance(inc_result, tuple) and len(inc_result) == 2:
                inc_p, inc_r = inc_result
            else:
                inc_p, inc_r = "", str(inc_result)

            if isinstance(exc_result, tuple) and len(exc_result) == 2:
                exc_p, exc_r = exc_result
            else:
                exc_p, exc_r = "", str(exc_result)

            pred_str = f"Inclusion predictions:\n{inc_r}\nExclusion predictions:\n{exc_r}"
            future = executor.submit(run_scoring, ctx.deps.presentation, trial_dict, pred_str, ctx.deps.llm_model)
            future_to_key[future] = key

        for future in as_completed(future_to_key):
            key = future_to_key[future]
            try:
                _, score_resp = future.result()

                # Extract JSON from response
                import re
                json_match = re.search(r'(\{.*\})', score_resp, re.DOTALL)
                try:
                    if json_match:
                        json_str = json_match.group(1)
                        data = json.loads(json_str)
                    else:
                        data = json.loads(score_resp)

                    scoring_dict[key] = ScoringResult(**data)
                except json.JSONDecodeError:
                    logger.error(f"Error parsing JSON for trial {key}")
                    scoring_dict[key] = ScoringResult(
                        relevance_explanation="Error parsing response",
                        relevance_score_R=0,
                        eligibility_explanation="Error parsing response",
                        eligibility_score_E=0,
                        raw_response=score_resp[:100] + "..." if len(score_resp) > 100 else score_resp
                    )
            except Exception as e:
                logger.error(f"Error scoring trial {key}: {str(e)}")
                scoring_dict[key] = ScoringResult()

    # Create the scoring logs list
    scoring_list = [
        TrialScoring(trial_id=key, scoring=value)
        for key, value in scoring_dict.items()
    ]

    # Create the ranked list sorted by eligibility score
    ranked_items = [
        RankedTrialResult(
            trial_id=key,
            relevance_explanation=score_result.relevance_explanation,
            relevance_score_R=score_result.relevance_score_R,
            eligibility_explanation=score_result.eligibility_explanation,
            eligibility_score_E=score_result.eligibility_score_E
        )
        for key, score_result in sorted(
            scoring_dict.items(),
            key=lambda kv: kv[1].eligibility_score_E,
            reverse=True
        )
    ]

    logger.info(f"Created scoring and ranking for {len(scoring_list)} trials")
    return Step4Output(scoring_logs=scoring_list, ranked=ranked_items)


# Main orchestrator function to run the entire pipeline
async def run_pipeline(
        ctx: RunContext[PipelineDependencies]
) -> PipelineOutput:
    """
    Run the complete clinical trial matching pipeline.

    This orchestrates the four steps: extraction, retrieval, eligibility, and scoring.
    """
    logger.info("Starting full pipeline execution")

    # Step 1: Extract patient information
    step1_output = await extract_patient_information(ctx)
    logger.info("Completed Step 1: Patient Information Extraction")

    # Step 2: Retrieve trials
    step2_output = await retrieve_trials(ctx, step1_output)
    logger.info("Completed Step 2: Trial Retrieval")

    # Step 3: Assess eligibility
    step3_output = await assess_eligibility(ctx, step1_output, step2_output)
    logger.info("Completed Step 3: Eligibility Assessment")

    # Step 4: Score and rank trials
    step4_output = await score_and_rank_trials(ctx, step1_output, step2_output, step3_output)
    logger.info("Completed Step 4: Trial Scoring and Ranking")

    # Return all outputs
    return PipelineOutput(
        step1=step1_output,
        step2=step2_output,
        step3=step3_output,
        step4=step4_output
    )


# Store a reference to a shared event loop
_GLOBAL_LOOP = None


# Create the agent with all tools registered
def get_pipeline_agent(llm_model: str) -> Agent:
    """
    Create a Pydantic AI agent for clinical trial matching.

    This creates a single agent that can execute the entire pipeline.
    """
    logger.info(f"Creating agent with model: {llm_model}")

    # Create and return the agent with all tools registered
    return Agent(
        llm_model,
        deps_type=PipelineDependencies,
        output_type=PipelineOutput,
        tools=[
            extract_patient_information,
            retrieve_trials,
            assess_eligibility,
            score_and_rank_trials,
            run_pipeline
        ],
        system_prompt=(
            "You are a clinical trial matching assistant. "
            "Your role is to extract patient information from clinical notes, "
            "retrieve relevant trials, assess eligibility, and rank matches."
        )
    )


# Main function to run the pipeline using the Pydantic AI agent
def run_pydantic_agent(
        presentation: str,
        llm_model: str,
        recruiting_status: str,
        min_date: date,
        max_date: date,
        phase: str,
        step: str = "all",  # Kept for backward compatibility
        step1_data=None,  # Unused in this refactored version
        step2_data=None,  # Unused in this refactored version
        step3_data=None,  # Unused in this refactored version
        step4_data=None  # Unused in this refactored version
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
        step: Which step to run (kept for backward compatibility)

    Returns:
        Dictionary with the results of all steps
    """
    global _GLOBAL_LOOP

    # Normalize the model name
    original_model = llm_model
    llm_model = llm_model.replace('google-', '').replace('anthropic-', '')

    # Fix model names for different providers
    if not llm_model.startswith(('openai:', 'google-gla:', 'anthropic:')):
        if "gpt" in llm_model.lower():
            llm_model = f"openai:{llm_model}"
        elif "gemini" in llm_model.lower():
            llm_model = f"google-gla:{llm_model}"
        elif "claude" in llm_model.lower():
            llm_model = f"anthropic:{llm_model}"
        logger.info(f"Converted model name to: {llm_model}")

    # Create the agent dependencies
    deps = PipelineDependencies(
        presentation=presentation,
        llm_model=llm_model,
        recruiting_status=recruiting_status,
        min_date=min_date,
        max_date=max_date,
        phase=phase
    )

    # Define the async function to run the agent
    async def run_async(model=llm_model, attempt=1):
        try:
            # Create the agent
            pipeline_agent = get_pipeline_agent(model)

            # Run the agent with run_pipeline as the tool
            logger.info("Running pipeline agent")
            prompt = "Run the clinical trial matching pipeline for the given patient."
            result = await pipeline_agent.run(prompt, deps=deps)

            # Debug the result structure
            logger.info(f"Result type: {type(result)}")
            logger.info(f"Result attributes: {dir(result)}")

            # Check if output attribute exists
            if hasattr(result, 'output'):
                logger.info(f"Output type: {type(result.output)}")

                # Try different ways to access the result data
                if isinstance(result.output, PipelineOutput):
                    return result.output.to_dict()
                else:
                    # Try accessing result directly if output is not a PipelineOutput
                    return {
                        "step1": result.step1.model_dump() if hasattr(result, 'step1') and result.step1 else None,
                        "step2": result.step2.model_dump() if hasattr(result, 'step2') and result.step2 else None,
                        "step3": result.step3.to_dict() if hasattr(result, 'step3') and result.step3 else None,
                        "step4": result.step4.to_dict() if hasattr(result, 'step4') and result.step4 else None
                    }
            else:
                # If result doesn't have an output attribute, convert the result itself
                logger.info("No output attribute, trying to convert result directly")
                if isinstance(result, PipelineOutput):
                    return result.to_dict()
                else:
                    # Fall back to a simple dict with available attributes
                    return {
                        "step1": getattr(result, 'step1', None),
                        "step2": getattr(result, 'step2', None),
                        "step3": getattr(result, 'step3', None),
                        "step4": getattr(result, 'step4', None)
                    }
        except Exception as e:
            logger.error(f"Error running pipeline with {model}: {str(e)}")

            # Fall back to GPT-4 if Gemini or Claude fails
            if attempt == 1:
                if "gemini" in model.lower():
                    logger.info("Falling back to GPT-4 after Gemini error")
                    return await run_async("openai:gpt-4", attempt=2)
                elif "claude" in model.lower():
                    logger.info("Falling back to GPT-4 after Claude error")
                    return await run_async("openai:gpt-4", attempt=2)

            # If we're already on a fallback or GPT fails, raise the error
            raise

    # Run the async function with proper event loop management
    try:
        # Check if we're already in an event loop
        try:
            loop = asyncio.get_running_loop()
            result = loop.run_until_complete(run_async())
        except RuntimeError:
            # No running event loop, create a new one
            if _GLOBAL_LOOP is None or _GLOBAL_LOOP.is_closed():
                _GLOBAL_LOOP = asyncio.new_event_loop()
                asyncio.set_event_loop(_GLOBAL_LOOP)
            result = _GLOBAL_LOOP.run_until_complete(run_async())

        # Add a final processing step to handle any non-serializable objects
        try:
            import json
            # Test if the result is JSON serializable
            json.dumps(result, default=str)
        except Exception as e:
            logger.warning(f"Result is not directly JSON serializable: {e}")
            # Create a clean copy with string conversions
            clean_result = {}
            for key, value in result.items():
                if value is None:
                    clean_result[key] = None
                elif isinstance(value, dict):
                    clean_result[key] = {
                        k: str(v) if not isinstance(v, (dict, list, str, int, float, bool, type(None))) else v
                        for k, v in value.items()}
                else:
                    clean_result[key] = str(value)
            result = clean_result

        return result
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        return {
            "error": str(e),
            "model": original_model
        }