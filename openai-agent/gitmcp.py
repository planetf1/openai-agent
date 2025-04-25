import asyncio
import base64
import os
import shutil
import nest_asyncio
import logfire


from agents import (
    Agent,
    Runner,
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
    trace,
)
from agents.mcp import MCPServer, MCPServerStdio
from dotenv import load_dotenv
from openai import AsyncOpenAI



async def run(mcp_server: MCPServer, model: str):
    agent = Agent(
        name="Assistant",
        instructions=f"You are an expert on git and github. Answer questions about git and github.",
        mcp_servers=[mcp_server],
        model=model
    )

    message = "Who's the most frequent contributor for i-am-bee/beeai-framework?"
    print("\n" + "-" * 40)
    print(f"Running: {message}")
    result = await Runner.run(starting_agent=agent, input=message)
    print(result.final_output)

    message = "Summarize the last change in the repository for odpi/egeria."
    print("\n" + "-" * 40)
    print(f"Running: {message}")
    result = await Runner.run(starting_agent=agent, input=message)
    print(result.final_output)


async def main():

    load_dotenv()

    # --- Langfuse Configuration ---
    # Get Langfuse credentials and host from environment variables
    LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
    LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
    LANGFUSE_HOST = os.getenv("LANGFUSE_HOST")

    # Validate Langfuse credentials
    if not LANGFUSE_PUBLIC_KEY:
        raise ValueError("Error: LANGFUSE_PUBLIC_KEY environment variable not set.")
    if not LANGFUSE_PUBLIC_KEY.startswith("pk-lf"):
        raise ValueError("Error: LANGFUSE_PUBLIC_KEY must start with 'pk-lf'.")

    if not LANGFUSE_SECRET_KEY:
        raise ValueError("Error: LANGFUSE_SECRET_KEY environment variable not set.")
    if not LANGFUSE_SECRET_KEY.startswith("sk-lf"):
        raise ValueError("Error: LANGFUSE_SECRET_KEY must start with 'sk-lf'.")

    if not LANGFUSE_HOST:
        raise ValueError("Error: LANGFUSE_HOST environment variable not set.")

    # Build Basic Auth header for Langfuse.
    LANGFUSE_AUTH = base64.b64encode(
        f"{LANGFUSE_PUBLIC_KEY}:{LANGFUSE_SECRET_KEY}".encode()
    ).decode()

    # Configure OpenTelemetry endpoint & headers for Langfuse
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = f"{LANGFUSE_HOST}/api/public/otel"
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic%20{LANGFUSE_AUTH}"
    # --- End Langfuse Configuration ---


    # OpenAI API Key - Ensure this is set if using OpenAI directly, not needed for Ollama via compatible endpoint
    # os.environ["OPENAI_API_KEY"] = "sk-proj-..."

    nest_asyncio.apply()
    logfire.configure(
        service_name='github_mcp',
        send_to_logfire=False, # Set to True to send traces to Langfuse Cloud/Self-hosted
    )

    logfire.instrument_openai_agents()

    # Get BASE_URL and API_KEY from environment variables
    # Using os.getenv allows for None if the variable is not set

    # URL of OpenAI compatible endpoint (e.g., Ollama)
    BASE_URL = os.getenv("OPENAI_BASE_URL")
    # OpenAI API key (set to dummy value like 'ollama' if using Ollama)
    API_KEY = os.getenv("OPENAI_API_KEY")
    # Model to use (e.g., 'granite3.2:8b' for Ollama)
    MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")
    # Personal access token, needed to access github
    GITHUB_TOKEN = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
    # Location on local filesystem of the MCP binary from https://github.com/github/github-mcp-server
    GITHUB_SERVER_BINARY = os.getenv("GITHUB_MCP_SERVER_BINARY")

    # Basic validation to ensure the required variables are set
    # API_KEY check might be optional depending on the endpoint (e.g., local Ollama)
    # if not API_KEY:
    #     raise ValueError("Error: OPENAI_API_KEY environment variable not set.")
    if not BASE_URL:
        raise ValueError("Error: OPENAI_BASE_URL environment variable not set.")
    if not MODEL_NAME:
        raise ValueError("Error: OPENAI_MODEL_NAME environment variable not set.")
    if not GITHUB_TOKEN:
        raise ValueError("Error: GITHUB_PERSONAL_ACCESS_TOKEN environment variable not set.")
    if not GITHUB_SERVER_BINARY:
        raise ValueError("Error: GITHUB_MCP_SERVER_BINARY environment variable not set.")
    if not os.path.exists(GITHUB_SERVER_BINARY):
        raise ValueError(f"Error: GITHUB_MCP_SERVER_BINARY path does not exist: {GITHUB_SERVER_BINARY}")
    if not os.access(GITHUB_SERVER_BINARY, os.X_OK):
        raise ValueError(f"Error: GITHUB_MCP_SERVER_BINARY is not executable: {GITHUB_SERVER_BINARY}")


    # Create the OpenAI client
    client = AsyncOpenAI(
        base_url=BASE_URL,
        api_key=API_KEY, # Pass the API key, even if it's a dummy value for Ollama
    )

    # Configure the agent library - if using openAI you could use the new api rather than chat completions
    set_default_openai_client(client=client, use_for_tracing=False)
    set_default_openai_api("chat_completions")
    set_tracing_disabled(disabled=False) # Enable tracing
    logfire.instrument_openai_agents() # Ensure agents are instrumented for tracing

    github_mcp_server_env = {}
    github_mcp_server_env["GITHUB_PERSONAL_ACCESS_TOKEN"] = GITHUB_TOKEN

    # Run the MCP server locally (alternative is to use SSE)
    async with MCPServerStdio(
        cache_tools_list=True,
        name="Github MCP server",
        params={
            "command": GITHUB_SERVER_BINARY,
            "args": ['stdio'],
            "env": github_mcp_server_env
        }
    )   as github_mcp_server:
        with trace(workflow_name="Git MCP Example"):
            # Now create an agent and make some calls to MCP
            await run(github_mcp_server,MODEL_NAME)


if __name__ == "__main__":

    asyncio.run(main())