import os
from agents import Agent, Runner, set_default_openai_api, set_default_openai_client, set_tracing_disabled
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents.mcp import MCPServer, MCPServerStdio

# --- Start of Modifications ---
# Load environment variables from .env file in the current directory
load_dotenv()

# Get BASE_URL and API_KEY from environment variables
# Using os.getenv allows for None if the variable is not set
BASE_URL = os.getenv("OPENAI_BASE_URL")
API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")
# Basic validation to ensure the variables are set
#if not API_KEY:
#    raise ValueError("Error: OPENAI_API_KEY environment variable not set.")
if not BASE_URL:
    raise ValueError("Error: OPENAI_BASE_URL environment variable not set.")
if not MODEL_NAME:
    raise ValueError("Error: OPENAI_BASE_URL environment variable not set.")

# Use the fetched environment variables
client = AsyncOpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
)

# Configure the agent library (assuming these functions exist)
set_default_openai_client(client=client, use_for_tracing=False)
set_default_openai_api("chat_completions")
set_tracing_disabled(disabled=True)

agent = Agent(name="Assistant", instructions="You are a helpful assistant", model=MODEL_NAME)


result = Runner.run_sync(agent, "Write me a joke about the British")
print(result.final_output)

# Code within the code,
# Functions calling themselves,
# Infinite loop's dance.
