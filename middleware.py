import datetime
from langchain.agents.middleware import after_model, AgentState, AgentMiddleware
from typing import Any
from langgraph.runtime import Runtime
from pydantic import BaseModel
from typing import Annotated
from operator import add
import threading

class Datapoint(BaseModel):
    input_token_count: int
    output_token_count: int
    estimated_energy_joule: float
    estimated_co2e_kg: float
    model_name: str
    timestamp: datetime.datetime
    message: str

# class CustomState(AgentState):
#     outputs: Annotated[list[Datapoint], add] = []

class EnergyMiddleware(AgentMiddleware):
    def __init__(self):
        super().__init__()
        self.datapoints: list[Datapoint] = []
        self._lock = threading.Lock()


    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:

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
            estimated_co2e_kg=co2e,
            model_name=model_name,
            timestamp=datetime.datetime.now(),
            message=str(last_message.content)[:100]
        )
        with self._lock:
            self.datapoints.append(output_datapoint)
    
    def get_report(self) -> list[Datapoint]:
        with self._lock:
            return self.datapoints.copy()

    def total_energy(self) -> float:
        """ Returns the sum of energy in the list of data points. """
        with self._lock:
            return sum(dp.estimated_energy_joule for dp in self.datapoints)

    def total_co2(self) -> float:
        """ Returns the sum of carbon dioxide emissions in the list of data points. """
        with self._lock:
            return sum(dp.estimated_co2e_kg for dp in self.datapoints)

    def breakdown_by_model(self) -> dict[str, float]:
        """ Returns a breakdown of energy consumption grouped by model. """
        result = {}
        with self._lock:
            for dp in self.datapoints:
                result.setdefault(dp.model_name, 0)
                result[dp.model_name] += dp.estimated_energy_joule
        return result



@after_model
def log_response(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    last_message = state['messages'][-1]
    
    # Only log if there is actual text content
    if last_message.content and str(last_message.content).strip():
        print(f"Model returned: \n------\n {last_message.content}\n------")
    return None


def estimate_energy_and_emissions(input_tokens: int, output_tokens: int, model: str) -> tuple[float, float]:
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

