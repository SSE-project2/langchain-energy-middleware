from langchain.tools import tool
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from middleware import log_tokens, CustomState
from reporting import get_total_energy_usage

# Basic multiagent setup for testing
# https://dev.to/fabiothiroki/run-langchain-locally-in-15-minutes-without-a-single-api-key-1j8m
# https://docs.langchain.com/oss/python/langchain/multi-agent/subagents

@tool("get_weather", description="Get the weather for a city")
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"The weather in {city} is rainy and 21.7 degrees celcius"

SUBAGENT_SYSTEM_PROMPT = """You are a helpful assistant. You are an expert at researching the weather.
You have the following tools:
- get_weather: this tool takes a city as input and returns the weather in that city.
"""

subagent = create_agent(
    model=ChatOllama(model="qwen3.5"),
    tools=[get_weather],
    system_prompt=SUBAGENT_SYSTEM_PROMPT,
    middleware=[log_tokens],
    state_schema=CustomState
)

@tool("weather", description="Research the weather and return findings")
def call_weather_agent(query: str) -> str:
    result = subagent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content

MAIN_SYSTEM_PROMPT = """
You are a helpful assistant.

You have access to the following tools:
- call_weather_agent: this calls another agent that will research the weather in a city when asked. Make sure to specify the name of the city.
"""

main_agent = create_agent(
    model=ChatOllama(model="qwen3.5"),
    tools=[call_weather_agent],
    system_prompt=MAIN_SYSTEM_PROMPT,
    middleware=[log_tokens],
    state_schema=CustomState # Maybe we can find a way to avoid having this here? Ideally we would just use it in the middleware
)

response = main_agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in Amsterdam?"}]}
)



print(response["messages"][-1].content)

print(f"Current estimated energy usage: {get_total_energy_usage(response['outputs'])}")

# TODO:
# - Get the metrics not just for the main agent but also for all subagents.
# - Find more realistic power estimates
# - A better way to manage the state? Also maybe a nicer API to get metrics from the state/model.
# - Feed metrics back into the agent. Maybe do this automatically every N steps through middleware or just give it a tool to check itself. Or maybe in the context?
# - The state object - here 'response' - actually already keeps track of all messages. We could therefore calculate metrics on demand, but I think the current pre-calculated method is also fine.