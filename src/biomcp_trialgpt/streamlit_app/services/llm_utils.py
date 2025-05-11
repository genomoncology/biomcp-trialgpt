import os
import openai

# Configure API key for OpenAI
_api_key = os.getenv("OPENAI_API_KEY")
if not _api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")
openai.api_key = _api_key

# Support both legacy and new client APIs
OpenAIClient = getattr(openai, "OpenAI", None)


def create_chat_completion(**kwargs):
    """
    Wrapper for OpenAI chat completion supporting both legacy and new client.
    """
    if hasattr(openai, "ChatCompletion"):
        return openai.ChatCompletion.create(**kwargs)
    if OpenAIClient:
        client = OpenAIClient(api_key=_api_key)
        return client.chat.completions.create(**kwargs)
    raise RuntimeError("No OpenAI chat completion method available")
