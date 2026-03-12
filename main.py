from langchain.tools import tool
from langchain.agents import create_agent
from langchain_ollama import ChatOllama

from middleware import EnergyMiddleware
from reporting import present_results


# Basic multiagent setup for testing
# https://dev.to/fabiothiroki/run-langchain-locally-in-15-minutes-without-a-single-api-key-1j8m
# https://docs.langchain.com/oss/python/langchain/multi-agent/subagents

@tool("get_weather", description="Get the weather for a city")
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"The weather in {city} is rainy and 21.7 degrees Celsius"

SUBAGENT_SYSTEM_PROMPT = """You are a helpful assistant. You are an expert at researching the weather. Respond in a whimsical tone.
You have the following tools:
- get_weather: this tool takes a city as input and returns the weather in that city.
"""

tracker = EnergyMiddleware()

subagent = create_agent(
    model=ChatOllama(model="qwen3.5"),
    tools=[get_weather],
    system_prompt=SUBAGENT_SYSTEM_PROMPT,
    middleware=[tracker],
)

@tool("weather", description="Research the weather and return findings")
def call_weather_agent(query: str) -> str:
    result = subagent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content

MAIN_SYSTEM_PROMPT = """
You are a helpful assistant. Respond in a serious tone.

You have access to the following tools:
- call_weather_agent: this calls another agent that will research the weather in a city when asked. Make sure to specify the name of the city.
"""

main_agent = create_agent(
    model=ChatOllama(model="qwen3.5"),
    tools=[call_weather_agent],
    system_prompt=MAIN_SYSTEM_PROMPT,
    middleware=[tracker],
)

response = main_agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in Amsterdam?"}]}
)

present_results(tracker.get_report())

