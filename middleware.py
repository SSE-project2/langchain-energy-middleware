import datetime
from langchain.agents.middleware import after_model, AgentState
from typing import Any
from langgraph.runtime import Runtime
from pydantic import BaseModel
from typing import Annotated
from operator import add

# Maybe these can actually be the same? separate for now for clarity ig
class Datapoint(BaseModel):
    input_token_count: int
    output_token_count: int
    estimated_energy_joule: float
    estimated_co2e_gram: float
    model_name: str
    timestamp: datetime.datetime
    message: str

class CustomState(AgentState):
    outputs: Annotated[list[Datapoint], add] = []


@after_model
def log_response(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    last_message = state['messages'][-1]
    
    # Only log if there is actual text content
    if last_message.content and str(last_message.content).strip():
        print(f"Model returned: \n------\n {last_message.content}\n------")
    return None

@after_model(state_schema=CustomState)
def log_tokens(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:

    last_message = state['messages'][-1]
    model_name = last_message.response_metadata.get("model_name", "unknown_model")
    input_token_count = last_message.usage_metadata.get("input_tokens", 0)
    output_token_count = last_message.usage_metadata.get("output_tokens", 0)

    if last_message.content is None or not str(last_message.content).strip():
        return None
    
    energy, co2e = estimate_energy_and_emissions(input_token_count, output_token_count, model_name)

    output_datapoint = Datapoint(
        input_token_count=input_token_count,
        output_token_count=output_token_count,
        estimated_energy_joule=energy,
        estimated_co2e_gram=co2e,
        model_name=model_name,
        timestamp=datetime.datetime.now(),
        message=str(last_message.content)[:100]
    )
    return {"outputs": [output_datapoint]}


def estimate_energy_and_emissions(input_tokens: int, output_tokens: int, model: str) -> tuple[float, float]:
    # These numbers are totally made up for now

    co2e_per_joule = 0.0000005

    # In format: (energy per input token, energy per output token)
    module_costs = {
        "qwen3.5": (0.00001, 0.00004)
    }
    
    input_energy = input_tokens * module_costs.get(model, (0, 0))[0]
    output_energy = output_tokens * module_costs.get(model, (0, 0))[1]
    total_energy = input_energy + output_energy
    co2e = total_energy * co2e_per_joule

    return total_energy, co2e

