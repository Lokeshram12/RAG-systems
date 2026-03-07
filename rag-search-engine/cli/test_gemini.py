import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model="gemma-3-27b-it",
    contents="Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum.",
)

# Print the model's response
print(response.text)

# Print token usage
prompt_tokens = response.usage_metadata.prompt_token_count
response_tokens = response.usage_metadata.candidates_token_count

print(f"\nPrompt tokens: {prompt_tokens}")
print(f"Response tokens: {response_tokens}")