import json
import logging
import os
from typing import Any

import anthropic

try:
    from google import genai
except ImportError:
    import google.generativeai as genai

from .llm_utils import create_chat_completion as _create_chat_completion

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# API Keys
_api_key = os.getenv("OPENAI_API_KEY")
_anthropic_key = os.getenv("ANTHROPIC_API_KEY")
_google_key = os.getenv("GEMINI_API_KEY")

# Initialize clients
anthropic_client = anthropic.Anthropic(api_key=_anthropic_key) if _anthropic_key else None

# Configure for google.generativeai if available
if _google_key and hasattr(genai, "configure"):
    genai.configure(api_key=_google_key)

# Extraction system prompt
EXTRACTION_SYSTEM = """
You are a clinical data-extraction assistant.
Extract from the free-text patient note exactly these JSON fields:
- age (int)
- gender (str)
- race_ethnicity (str)
- chief_complaint (str)
- onset (str)
- duration (str)
- conditions (list[str])
- terms (list[str])
- interventions (list[str])
- medications (list[str])
- allergies (list[str])
- vitals (dict)
- labs (dict)
- imaging (dict)
- assessment (str)
- plan (str)

Output ONLY valid JSON.
"""


def _get_extraction_response(presentation: str, model: str, prompt: str) -> str:
    """
    Get extraction response from the appropriate model API.

    Args:
        presentation: The clinical note text to parse
        model: The model to use for parsing
        prompt: The prompt to send to the model

    Returns:
        The raw response content from the model as a string
    """
    # Route to the appropriate API based on selected model
    if "gpt-" in model:
        resp = _create_chat_completion(
            model=model.split("gpt-")[1],
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=1,
        )
        return resp.choices[0].message.content.strip()
    elif "anthropic" in model:
        from anthropic import AI_PROMPT, HUMAN_PROMPT

        human_prompt = HUMAN_PROMPT + prompt + AI_PROMPT
        response = anthropic_client.messages.create(
            model=model.split("anthropic-")[1] if model.startswith("anthropic-") else model.replace("anthropic:", ""),
            max_tokens=1000,
            temperature=0.7,
            messages=[{"role": "user", "content": human_prompt}],
        )
        return response.content[0].text.strip()
    elif model.startswith("google-"):
        # Google Gemini via google.generativeai or google-genai SDK
        model_name = model.split("google-")[1].replace("gla:", "")
        # If using google.generativeai
        if hasattr(genai, "chat"):
            resp = genai.chat.completions.create(
                model=model_name,
                messages=[{"author": "user", "content": prompt}],
                temperature=0.0,
            )
            return resp.candidates[0].content.strip()
        # If using google-genai (Client)
        if hasattr(genai, "Client"):
            client = genai.Client(api_key=_google_key)
            # google-genai SDK: create a chat and send message
            chat = client.chats.create(model=model_name)
            response = chat.send_message(prompt)
            return response.text.strip()
        msg = "Google Generative AI SDK not available."
        raise ValueError(msg)
    else:
        msg = f"Unsupported model: {model}"
        raise ValueError(msg)


def parse_clinical_note(presentation: str, model: str) -> tuple[dict[str, Any], str, str]:
    """
    Parse a clinical note using the specified model.

    Args:
        presentation: The clinical note text to parse
        model: The model to use for parsing (e.g., "gpt-4", "anthropic-claude", "google-gemini")

    Returns:
        A tuple containing:
        - The extracted data as a dictionary
        - The prompt used for extraction
        - The raw response content from the model
    """
    prompt = f"""{EXTRACTION_SYSTEM}\n\nNote:\n\"\"\"\n{presentation}\n\"\"\"\n"""
    try:
        logger.info(f"Using model {model} for parsing clinical note")
        resp_content = _get_extraction_response(presentation, model, prompt)
        try:
            data = json.loads(resp_content)
        except json.JSONDecodeError:
            clean = resp_content.strip()
            # strip Markdown code fences
            if clean.startswith("```") and clean.endswith("```"):
                lines = clean.splitlines()[1:-1]
                clean = "\n".join(lines)
            # extract JSON substring
            if "{" in clean and "}" in clean:
                start = clean.find("{")
                end = clean.rfind("}")
                clean = clean[start : end + 1]
            try:
                data = json.loads(clean)
            except json.JSONDecodeError:
                data = {"chief_complaint": presentation}
            return data, prompt, resp_content
        else:
            return data, prompt, resp_content
    except Exception as e:
        data = {"chief_complaint": presentation, "error": str(e)}
        logger.exception("Error parsing clinical note")
        resp_content = json.dumps(data)
        return data, prompt, resp_content
