import os
import json
import openai
import anthropic
try:
    from google import genai
except ImportError:
    import google.generativeai as genai
from typing import Dict, Any, Tuple
from openai.error import RateLimitError
from .agent_openai import _create_chat_completion

# Configure API key (in case not already set)
_api_key = os.getenv("OPENAI_API_KEY")
if not _api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")
openai.api_key = _api_key

# Configure Anthropic
_anthropic_key = os.getenv("ANTHROPIC_API_KEY")
if not _anthropic_key:
    raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
anthropic.api_key = _anthropic_key
anthropic_client = anthropic.Client()

# Load Google API key
_google_key = os.getenv("GOOGLE_API_KEY")
if not _google_key:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")
# Configure for google.generativeai if available
if hasattr(genai, 'configure'):
    genai.configure(api_key=_google_key)

EXTRACTION_SYSTEM = """
You are a clinical data-extraction assistant.
Extract from the free-text patient note exactly these JSON fields:
- age (int)
- gender (str)
- race_ethnicity (str)
- chief_complaint (str)
- onset (str)
- duration (str)
- associated_symptoms (list[str])
- past_medical_history (list[str])
- social_history (list[str])
- medications (list[str])
- physical_exam (str)
- ekg_findings (str)
- conditions (list[str])  # conditions terms for trial matching
- terms (list[str])       # general search terms from note
- interventions (list[str])  # intervention names mentioned

Output ONLY valid JSON.
"""

def _get_extraction_response(presentation: str, model: str, prompt: str) -> str:
    # Route to the appropriate API based on selected model
    if model.startswith("gpt-"):
        resp = _create_chat_completion(
            model=model,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()
    elif model.startswith("anthropic-"):
        from anthropic import HUMAN_PROMPT, AI_PROMPT
        human_prompt = HUMAN_PROMPT + prompt + AI_PROMPT
        response = anthropic_client.completions.create(
            model=model.split("anthropic-")[1],
            prompt=human_prompt,
            max_tokens_to_sample=1000,
        )
        return response.completion.strip()
    elif model.startswith("google-"):
        # Google Gemini via google.generativeai or google-genai SDK
        model_name = model.split("google-")[1]
        # If using google.generativeai
        if hasattr(genai, 'chat'):
            resp = genai.chat.completions.create(
                model=model_name,
                messages=[{"author": "user", "content": prompt}],
                temperature=0.0,
            )
            return resp.candidates[0].content.strip()
        # If using google-genai (Client)
        if hasattr(genai, 'Client'):
            client = genai.Client(api_key=_google_key)
            # google-genai SDK: create a chat and send message
            chat = client.chats.create(model=model_name)
            response = chat.send_message(prompt)
            return response.text.strip()
        raise ValueError("Google Generative AI SDK not available.")
    else:
        raise ValueError(f"Unsupported model: {model}")

def parse_clinical_note(presentation: str, model: str) -> Tuple[Dict[str, Any], str, str]:
    prompt = f"""{EXTRACTION_SYSTEM}\n\nNote:\n\"\"\"\n{presentation}\n\"\"\"\n"""
    try:
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
                clean = clean[start:end+1]
            try:
                data = json.loads(clean)
            except json.JSONDecodeError:
                data = {"chief_complaint": presentation}
        return data, prompt, resp_content
    except Exception as e:
        data = {"chief_complaint": presentation, "error": str(e)}
        resp_content = json.dumps(data)
        return data, prompt, resp_content
