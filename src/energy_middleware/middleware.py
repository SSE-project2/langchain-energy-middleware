import datetime
import threading
import uuid
from typing import Any

from langchain.agents.middleware import AgentState, AgentMiddleware
from langgraph.runtime import Runtime
from pydantic import BaseModel
from langgraph.config import get_config

class Datapoint(BaseModel):
    """
    Measurement of estimated energy usage and environmental impact
    for a single model call.

    Attributes:
        input_token_count (int): Number of input tokens processed.
        output_token_count (int): Number of output tokens generated.
        estimated_energy_joule (float): Estimated energy usage in Joules.
        estimated_co2e_kg (float): Estimated CO2 emissions in kilograms.
        model_name (str): Name of the model used.
        timestamp (datetime.datetime): Time when the datapoint was recorded.
        message (str): Truncated model output message (first 100 chars).
        prompt_id (str): Identifier linking agent calls stemming from the same original prompt.
        agent_name (str): Name of the agent that generated the output.
    """
    input_token_count: int
    output_token_count: int
    estimated_energy_joule: float
    estimated_co2e_kg: float
    model_name: str
    timestamp: datetime.datetime
    message: str
    prompt_id: str
    agent_name: str


class EnergyMiddleware(AgentMiddleware):
    """
    Middleware that tracks token usage, estimates energy consumption,
    and calculates associated CO2 emissions for agent model calls.

    This middleware maintains a thread-safe list of datapoints and
    supports nested agent calls using a prompt ID stack.
    """

    def __init__(self):
        super().__init__()
        self.datapoints: list[Datapoint] = []
        self._lock = threading.Lock()
        self._prompt_id_stack: list[str] = [] # Could also be replaced by a field and a counter.

    @property
    def _current_prompt_id(self) -> str | None:
        """
        Get the current prompt ID from the stack. 

        Returns:
            str | None: The current prompt ID, or None if the stack is empty.
        """
        with self._lock:
            return self._prompt_id_stack[-1] if self._prompt_id_stack else None

    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """
        Hook executed before an agent runs.

        Ensures that a prompt ID is assigned and propagated through
        nested agent calls.

        Args:
            state (AgentState): Current agent state.
            runtime (Runtime): Runtime context.

        Returns:
            dict[str, Any] | None: Optional state updates (unused).
        """
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
        """
        Hook executed after an agent finishes.

        Removes the latest prompt ID from the stack.

        Args:
            state (AgentState): Current agent state.
            runtime (Runtime): Runtime context.

        Returns:
            dict[str, Any] | None: Optional state updates (unused).
        """
        with self._lock:
            if self._prompt_id_stack:
                self._prompt_id_stack.pop()
        return None

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """
        Hook executed after a model generates a response.

        Extracts token usage, estimates energy and emissions, and
        records a datapoint for a model call.

        Args:
            state (AgentState): Current agent state containing messages.
            runtime (Runtime): Runtime context.

        Returns:
            dict[str, Any] | None: Optional state updates (unused).
        """
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
            estimated_co2e_kg=co2e,
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
        """
        Retrieve a copy of all collected datapoints.

        Returns:
            list[Datapoint]: A copy of the recorded datapoints.
        """
        with self._lock:
            return self.datapoints.copy()

    def total_energy(self) -> float:
        """
        Calculate total energy consumption across all datapoints.

        Returns:
            float: Total energy in Joules.
        """
        with self._lock:
            return sum(dp.estimated_energy_joule for dp in self.datapoints)

    def total_co2(self) -> float:
        """
        Calculate total CO2 emissions across all datapoints.

        Returns:
            float: Total CO2 emissions in kilograms.
        """
        with self._lock:
            return sum(dp.estimated_co2e_kg for dp in self.datapoints)

    def breakdown_by_model(self) -> dict[str, float]:
        """
        Compute energy usage grouped by model name.

        Returns:
            dict[str, float]: Mapping of model names to total energy usage in Joules.
        """
        result = {}
        with self._lock:
            for dp in self.datapoints:
                result.setdefault(dp.model_name, 0)
                result[dp.model_name] += dp.estimated_energy_joule
        return result



def estimate_energy_and_emissions(input_tokens: int, output_tokens: int, model: str) -> tuple[float, float]:
    """
    Estimate energy consumption and CO2 emissions for a model inference.

    The estimation is based on:
    - FLOPs per token approximation for transformer models
    - Assumed hardware efficiency (FLOPs per Joule)
    - Global average carbon intensity

    Args:
        input_tokens (int): Number of input tokens.
        output_tokens (int): Number of output tokens.
        model (str): Model identifier used to determine parameter count.

    Returns:
        tuple[float, float]:
            - Total energy consumption in Joules
            - Estimated CO2 emissions in kilograms
    """
    # Carbon Intensity
    # Global average carbon intensity: 0.45 kg CO2 / kWh
    # Conversion: 1 kWh = 3,600,000 Joules
    # 0.45 / 3,600,000 ≈ 1.25e-7 kg CO2 per Joule
    co2e_per_joule = 1.25e-7  # kg CO2 per Joule

    # Hardware Efficiency Assumption (Consumer GPU Baseline) NVIDIA RTX 4070 specifications obtained online
    # FP16 (half precision) throughput: 29.15 TFLOPs
    # TDP (Thermal Design Power): 200 W
    #
    # FLOPs per Joule = FLOPs per second / Watts = (29.15e12 FLOPs/s) / 200 W
    # ≈ 1.46e11 FLOPs per Joule (theoretical peak)
    FLOPS_PER_JOULE = 1.46e11  # RTX 4070 FP16 peak efficiency

    # Transformer Inference Compute Approximation used in transformer literature:
    # FLOPs per token ≈ 2 × number_of_parameters (Forward pass only; training typically ≈ 6P)
    # Assumes dense models
    # More models could be added later, the number of parameters is usually in the name.
    MODEL_PARAMETERS = {
        "qwen3.5:4b": 4_000_000_000,
        "qwen3.5:2b": 2_000_000_000,
    }

    # I have sources for the numbers above

    params = MODEL_PARAMETERS.get(model, 0)
    total_tokens = input_tokens + output_tokens

    # Total FLOPs for inference
    total_flops = 2 * params * total_tokens

    # Convert compute to energy
    total_energy = total_flops / FLOPS_PER_JOULE  # Joules

    # Convert energy to CO2
    co2e = total_energy * co2e_per_joule  # kg CO2

    return total_energy, co2e

