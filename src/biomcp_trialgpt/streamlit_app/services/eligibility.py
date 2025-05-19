import json
import logging
from typing import Any

# Google Generative AI SDK import
try:
    from google import genai
except ImportError:
    import google.generativeai as genai

import anthropic

# Import API key getter function
from biomcp_trialgpt.streamlit_app.services.note_extractor import get_api_keys

from .llm_utils import create_chat_completion as _create_chat_completion

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_eligibility_prompt(presentation: str, trial_info: dict[str, Any], inc_exc: str) -> str:
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
    prompt += f"\tElement 3. Classify the patient eligibility for this specific {inc_exc} criterion: "
    if inc_exc == "inclusion":
        prompt += (
            'the label must be chosen from {"not applicable", "not enough information", "included", "not included"}. '
            '"not applicable" should only be used for criteria that are not applicable to the patient. '
            '"not enough information" should be used where the patient note does not contain sufficient '
            'information for making the classification. Try to use as less "not enough information" as '
            "possible because if the note does not mention a medically important fact, you can assume "
            'that the fact is not true for the patient. "included" denotes that the patient meets the '
            'inclusion criterion, while "not included" means the reverse.\n'
        )
    else:
        prompt += (
            'the label must be chosen from {"not applicable", "not enough information", "excluded", "not excluded"}. '
            '"not applicable" should only be used for criteria that are not applicable to the patient. '
            '"not enough information" should be used where the patient note does not contain sufficient '
            'information for making the classification. Try to use as less "not enough information" as '
            "possible because if the note does not mention a medically important fact, you can assume "
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
    # Get current API keys
    openai_key, anthropic_key, google_key = get_api_keys()

    # OpenAI
    logger.info(f"Creating agent with model: {model}")
    if "gpt-" in model:
        if not openai_key:
            msg = "OpenAI API key not set. Please provide an API key in the sidebar."
            raise ValueError(msg)

        resp = _create_chat_completion(
            api_key=openai_key,
            model=model.split("gpt-")[1],
            messages=[{"role": "system", "content": ""}, {"role": "user", "content": prompt}],
            temperature=1,
        )
        return resp.choices[0].message.content.strip()
    # Anthropic
    if "anthropic" in model:
        from anthropic import AI_PROMPT, HUMAN_PROMPT

        if not anthropic_key:
            msg = "Anthropic API key not set. Please provide an API key in the sidebar."
            raise ValueError(msg)

        # Create a new client instance with the current API key
        anthropic_client = anthropic.Anthropic(api_key=anthropic_key)

        human_prompt = HUMAN_PROMPT + prompt + AI_PROMPT
        response = anthropic_client.messages.create(
            model=model.split("anthropic-")[1] if model.startswith("anthropic-") else model.replace("anthropic:", ""),
            max_tokens=1000,
            temperature=0.7,
            messages=[{"role": "user", "content": human_prompt}],
        )
        return response.content[0].text.strip()
    # Google
    if model.startswith("google-"):
        if not google_key:
            msg = "Google Gemini API key not set. Please provide an API key in the sidebar."
            raise ValueError(msg)

        # Configure genai with current API key
        if hasattr(genai, "configure"):
            genai.configure(api_key=google_key)

        model_name = model.split("google-")[1].replace("gla:", "")
        if hasattr(genai, "chat"):
            resp = genai.chat.completions.create(
                model=model_name,
                messages=[{"author": "user", "content": prompt}],
                temperature=0.0,
            )
            return resp.candidates[0].content.strip()
        if hasattr(genai, "Client"):
            client = genai.Client(api_key=google_key)
            chat = client.chats.create(model=model_name)
            response = chat.send_message(prompt)
            return response.text.strip()
        msg = "Google Generative AI SDK not available."
        raise ValueError(msg)
    msg = f"Unsupported model: {model}"
    raise ValueError(msg)


def run_eligibility(presentation: str, trial_info: dict[str, Any], model: str) -> dict[str, tuple[str, str]]:
    """Run inclusion and exclusion matching."""
    inc_prompt = build_eligibility_prompt(presentation, trial_info, "inclusion")
    inc_resp = _call_llm(inc_prompt, model)
    exc_prompt = build_eligibility_prompt(presentation, trial_info, "exclusion")
    exc_resp = _call_llm(exc_prompt, model)
    return {"inclusion": (inc_prompt, inc_resp), "exclusion": (exc_prompt, exc_resp)}
