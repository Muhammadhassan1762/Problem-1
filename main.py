from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig, Runner
from dotenv import load_dotenv
import os
import sys
import io

# ✅ Fix Urdu display on Windows terminal
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

load_dotenv(override=True)

gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-1.5-flash-latest",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

agent = Agent(
    name="Translator",
    instructions="""You are an expert Urdu translator.
    Translate English sentences into grammatically correct, natural, and fluent Urdu.
    Only return the translated Urdu text — no explanations, no English, no extra commentary."""
)

response = Runner.run_sync(
    agent,
    input="My name is Hassan. I am an undergraduate software engineering student.",
    run_config=config
)

# ✅ Print only the final output
print(response.final_output)