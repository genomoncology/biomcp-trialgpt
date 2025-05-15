import json
import logging
from typing import Dict, Any, Tuple

# Google Generative AI SDK import
try:
    from google import genai
except ImportError:
    import google.generativeai as genai

# Import configurations from note_extractor
from biomcp_trialgpt.streamlit_app.services.note_extractor import anthropic_client, _google_key
from .llm_utils import create_chat_completion as _create_chat_completion
from anthropic import HUMAN_PROMPT, AI_PROMPT

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure Google SDK if available
if hasattr(genai, 'configure'):
    genai.configure(api_key=_google_key)


def build_eligibility_prompt(presentation: str, trial_info: Dict[str, Any], inc_exc: str) -> str:
    prompt = (
        f"You are a helpful assistant for clinical trial recruitment. Your task is to compare a "
        f"given patient note and the {inc_exc} criteria of a clinical trial to determine the "
        f"patient's eligibility at the criterion level.\n"
    )
    if inc_exc == "inclusion":
        prompt += (
            "The factors that allow someone to participate in a clinical study are called "
            "inclusion criteria. They are based on characteristics such as age, gender, the type "
            "and stage of a disease, previous treatment history, and other medical conditions.\n"
        )
    else:
        prompt += (
            "The factors that disqualify someone from participating are called exclusion "
            "criteria. They are based on characteristics such as age, gender, the type and stage "
            "of a disease, previous treatment history, and other medical conditions.\n"
        )
    prompt += (
        f"You should check the {inc_exc} criteria one-by-one, and output the following "
        "three elements for each criterion:\n"
    )
    prompt += (
        f"\tElement 1. For each {inc_exc} criterion, briefly generate your reasoning process: "
        "First, judge whether the criterion is not applicable (not very common), where the "
        "patient does not meet the premise of the criterion. Then, check if the patient note "
        "contains direct evidence. If so, judge whether the patient meets or does not meet the "
        "criterion. If there is no direct evidence, try to infer from existing evidence, and "
        "answer one question: If the criterion is true, is it possible that a good patient note "
        "will miss such information? If impossible, then you can assume that the criterion is not "
        "true. Otherwise, there is not enough information.\n"
    )
    prompt += (
        "\tElement 2. If there is relevant information, you must generate a list of relevant "
        "sentence IDs in the patient note. If there is no relevant information, you must annotate "
        "an empty list.\n"
    )
    prompt += (
        f"\tElement 3. Classify the patient eligibility for this specific {inc_exc} criterion: "
    )
    if inc_exc == "inclusion":
        prompt += (
            'the label must be chosen from {"not applicable", "not enough information", "included", "not included"}. '  
            '"not applicable" should only be used for criteria that are not applicable to the patient. '  
            '"not enough information" should be used where the patient note does not contain sufficient '  
            'information for making the classification. Try to use as less "not enough information" as '  
            'possible because if the note does not mention a medically important fact, you can assume '  
            'that the fact is not true for the patient. "included" denotes that the patient meets the '  
            'inclusion criterion, while "not included" means the reverse.\n'
        )
    else:
        prompt += (
            'the label must be chosen from {"not applicable", "not enough information", "excluded", "not excluded"}. '  
            '"not applicable" should only be used for criteria that are not applicable to the patient. '  
            '"not enough information" should be used where the patient note does not contain sufficient '  
            'information for making the classification. Try to use as less "not enough information" as '  
            'possible because if the note does not mention a medically important fact, you can assume '  
            'that the fact is not true for the patient. "excluded" denotes that the patient meets the '  
            'exclusion criterion and should be excluded in the trial, while "not excluded" means the reverse.\n'
        )
    prompt += (
        "You should output only a JSON dict exactly formatted as: dict{str(criterion_number): "
        "list[str(element_1_brief_reasoning), list[int(element_2_sentence_id)], "
        "str(element_3_eligibility_label)]}.\n"
    )
    prompt += "Here is the patient note:\n" + presentation + "\n\n"
    prompt += "Here is the clinical trial info:\n" + json.dumps(trial_info) + "\nPlain JSON output:"
    return prompt


def _call_llm(prompt: str, model: str) -> str:
    # OpenAI
    logger.info(f"Creating agent with model: {model}")
    if model.startswith("gpt-"):
        resp = _create_chat_completion(
            model=model,
            messages=[{"role": "system", "content": ""}, {"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()
    # Anthropic
    if model.startswith("anthropic-"):
        human_prompt = HUMAN_PROMPT + prompt + AI_PROMPT
        response = anthropic_client.completions.create(
            model=model.split("anthropic-")[1],
            prompt=human_prompt,
            max_tokens_to_sample=1000,
        )
        return response.completion.strip()
    # Google
    if model.startswith("google-"):
        model_name = model.split("google-")[1].replace('gla:', '')
        if hasattr(genai, 'chat'):
            resp = genai.chat.completions.create(
                model=model_name,
                messages=[{"author": "user", "content": prompt}],
                temperature=0.0,
            )
            return resp.candidates[0].content.strip()
        if hasattr(genai, 'Client'):
            client = genai.Client(api_key=_google_key)
            chat = client.chats.create(model=model_name)
            response = chat.send_message(prompt)
            return response.text.strip()
        raise ValueError("Google Generative AI SDK not available.")
    raise ValueError(f"Unsupported model: {model}")


def run_eligibility(presentation: str, trial_info: Dict[str, Any], model: str) -> Dict[str, Tuple[str, str]]:
    """Run inclusion and exclusion matching."""
    inc_prompt = build_eligibility_prompt(presentation, trial_info, "inclusion")
    inc_resp = _call_llm(inc_prompt, model)
    exc_prompt = build_eligibility_prompt(presentation, trial_info, "exclusion")
    exc_resp = _call_llm(exc_prompt, model)
    return {"inclusion": (inc_prompt, inc_resp), "exclusion": (exc_prompt, exc_resp)}
