import datetime
import threading
import uuid
from typing import Any

from langchain.agents.middleware import AgentState, AgentMiddleware
from langgraph.runtime import Runtime
from pydantic import BaseModel
from langgraph.config import get_config

class Datapoint(BaseModel):
    input_token_count: int
    output_token_count: int
    estimated_energy_joule: float
    estimated_co2e_gram: float
    model_name: str
    timestamp: datetime.datetime
    message: str
    prompt_id: str
    agent_name: str


class EnergyMiddleware(AgentMiddleware):
    def __init__(self):
        super().__init__()
        self.datapoints: list[Datapoint] = []
        self._lock = threading.Lock()
        self._prompt_id_stack: list[str] = [] # Could also be replaced by a field and a counter.

    @property
    def _current_prompt_id(self) -> str | None:
        with self._lock:
            return self._prompt_id_stack[-1] if self._prompt_id_stack else None

    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        existing_id = self._current_prompt_id

        if existing_id:
            with self._lock:
                self._prompt_id_stack.append(existing_id)
            return None

        new_id = str(uuid.uuid4())
        with self._lock:
            self._prompt_id_stack.append(new_id)
        return None

    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        with self._lock:
            if self._prompt_id_stack:
                self._prompt_id_stack.pop()
        return None

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        last_message = state["messages"][-1]
        model_name = last_message.response_metadata.get("model_name", "unknown_model")
        input_token_count = last_message.usage_metadata.get("input_tokens", 0)
        output_token_count = last_message.usage_metadata.get("output_tokens", 0)

        config = get_config()
        agent_name = config["metadata"].get("lc_agent_name", "unknown_agent") if config and "metadata" in config else "unknown_agent"

        if last_message.content is None or not str(last_message.content).strip():
            return None

        energy, co2e = estimate_energy_and_emissions(input_token_count, output_token_count, model_name)

        prompt_id = self._current_prompt_id

        output_datapoint = Datapoint(
            input_token_count=input_token_count,
            output_token_count=output_token_count,
            estimated_energy_joule=energy,
            estimated_co2e_gram=co2e,
            model_name=model_name,
            timestamp=datetime.datetime.now(),
            message=str(last_message.content)[:100],
            prompt_id=prompt_id,
            agent_name=agent_name,
        )
        with self._lock:
            self.datapoints.append(output_datapoint)

        return None

    def get_report(self) -> list[Datapoint]:
        with self._lock:
            return self.datapoints.copy()


def estimate_energy_and_emissions(input_tokens: int, output_tokens: int, model: str) -> tuple[float, float]:
    
    co2e_per_joule = 0.0000005
    # In format: (energy per input token, energy per output token)
    model_costs = {
        "qwen3.5": (0.00001, 0.00004),
    }
    input_energy = input_tokens * model_costs.get(model, (0, 0))[0]
    output_energy = output_tokens * model_costs.get(model, (0, 0))[1]
    total_energy = input_energy + output_energy
    co2e = total_energy * co2e_per_joule
    return total_energy, co2e