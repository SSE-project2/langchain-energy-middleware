import datetime
import threading
import uuid
from typing import Any, Callable

from langchain.agents.middleware import after_model, AgentState, AgentMiddleware, ModelRequest, \
    ModelResponse
from langgraph.runtime import Runtime
from pydantic import BaseModel


class Datapoint(BaseModel):
    input_token_count: int
    output_token_count: int
    estimated_energy_joule: float
    estimated_co2e_gram: float
    model_name: str
    timestamp: datetime.datetime
    message: str
    prompt_id: str

class EnergyMiddleware(AgentMiddleware):
    def __init__(self):
        super().__init__()
        self.datapoints: list[Datapoint] = []
        self._lock = threading.Lock()
        self.prompt_id: str | None = None
        self.counter = 0

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        #Gets executed before any model call. If no other prompt has been executed yet, i.e. its the first (main) prompt,
        #we generate a new ID that will act as the base ID.
        if self.prompt_id is None:
            self.prompt_id = (str(uuid.uuid4()))
        return handler(request)

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:

        last_message = state['messages'][-1]
        model_name = last_message.response_metadata.get("model_name", "unknown_model")
        input_token_count = last_message.usage_metadata.get("input_tokens", 0)
        output_token_count = last_message.usage_metadata.get("output_tokens", 0)

        if last_message.content is None or not str(last_message.content).strip():
            return None
        
        energy, co2e = estimate_energy_and_emissions(input_token_count, output_token_count, model_name)

        prompt_id = self.prompt_id or 'unknown';
        self.counter += 1

        print(f'prompt_id: {prompt_id}')

        output_datapoint = Datapoint(
            input_token_count=input_token_count,
            output_token_count=output_token_count,
            estimated_energy_joule=energy,
            estimated_co2e_gram=co2e,
            model_name=model_name,
            timestamp=datetime.datetime.now(),
            message=str(last_message.content)[:100],
            prompt_id=prompt_id,
        )
        with self._lock:
            self.datapoints.append(output_datapoint)

        #We basically maintain a counter, which gets incremented each time a model is called. When the counter hits 1
        #The first (main) prompt is called, when it hits 2, the sub-prompt is called so then we know that's the last prompt
        #So we can generate a new ID for the next part
        if self.counter > 1:
            self.counter = 0
            self.prompt_id = None

        return None

    def get_report(self) -> list[Datapoint]:
        with self._lock:
            return self.datapoints.copy()



@after_model
def log_response(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    last_message = state['messages'][-1]
    
    # Only log if there is actual text content
    if last_message.content and str(last_message.content).strip():
        print(f"Model returned: \n------\n {last_message.content}\n------")
    return None




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

