import os
from dotenv import load_dotenv

# Load from .env
load_dotenv()

print("LANGCHAIN_API_KEY present:", bool(os.getenv("LANGCHAIN_API_KEY")))
print("LANGCHAIN_PROJECT:", os.getenv("LANGCHAIN_PROJECT"))
print("LANGCHAIN_ENDPOINT:", os.getenv("LANGCHAIN_ENDPOINT", "default"))
print("REDIS_URL:", os.getenv("REDIS_URL", "redis://localhost:6379"))