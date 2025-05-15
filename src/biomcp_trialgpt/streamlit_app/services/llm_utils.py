import os

from openai import OpenAI

# Configure API key for OpenAI
_api_key = os.getenv("OPENAI_API_KEY")
if not _api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# client = OpenAI(api_key=_api_key)

# Support both legacy and new client APIs
# OpenAIClient = getattr(client, "OpenAI", None)


def create_chat_completion(**kwargs):
    """
    Wrapper for OpenAI chat completion supporting both legacy and new client.
    """
    # if hasattr(client, "ChatCompletion"):
    #     return client.chat.completions.create(**kwargs)
    client = OpenAI(api_key=_api_key)
    return client.chat.completions.create(**kwargs)
