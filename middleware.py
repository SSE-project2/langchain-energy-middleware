import datetime
import threading
import uuid
from typing import Any, Literal

from langchain.agents.middleware import AgentState, AgentMiddleware
from langgraph.runtime import Runtime
from pydantic import BaseModel
from langgraph.config import get_config

class Datapoint(BaseModel):
    input_token_count: int
    output_token_count: int
    estimated_energy_joule: float
    estimated_co2e_kg: float
    model_name: str
    timestamp: datetime.datetime
    message: str
    prompt_id: str
    agent_name: str


class GroupSummary(BaseModel):
    name: str
    total_energy_joule: float
    total_co2e_kg: float
    total_input_tokens: int
    total_output_tokens: int
    datapoint_count: int

class EnergyMiddleware(AgentMiddleware):
    def __init__(self):
        super().__init__()
        self.datapoints: list[Datapoint] = []
        self._lock = threading.Lock()
        self._prompt_id_stack: list[str] = [] # Could also be replaced by a field and a counter.
        self._prompt_order: list[str] = []

    @property
    def _current_prompt_id(self) -> str | None:
        with self._lock:
            return self._prompt_id_stack[-1] if self._prompt_id_stack else None

    def _filter_datapoints(self, last_n_prompts, last_n_hours):
        with self._lock:
            points = self.datapoints.copy()
            recent_ids = set(self._prompt_order[-last_n_prompts:]) if last_n_prompts is not None else None

        if recent_ids is not None:
            points = [dp for dp in points if dp.prompt_id in recent_ids]
        if last_n_hours is not None:
            cutoff = datetime.datetime.now() - datetime.timedelta(hours=last_n_hours)
            points = [dp for dp in points if dp.timestamp >= cutoff]
        return points

    def _group_datapoints(self, points: list[Datapoint], key: str) -> list[GroupSummary]:
        buckets: dict[str, GroupSummary] = {}
        for dp in points:
            name = getattr(dp, key)
            if name not in buckets:
                buckets[name] = GroupSummary(
                    name=name,
                    total_energy_joule=0.0,
                    total_co2e_kg=0.0,
                    total_input_tokens=0,
                    total_output_tokens=0,
                    datapoint_count=0
                )
            buckets[name].total_energy_joule += dp.estimated_energy_joule
            buckets[name].total_co2e_kg += dp.estimated_co2e_kg
            buckets[name].total_input_tokens += dp.input_token_count
            buckets[name].total_output_tokens += dp.output_token_count
            buckets[name].datapoint_count += 1

        return list(buckets.values())

    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        existing_id = self._current_prompt_id

        if existing_id:
            with self._lock:
                self._prompt_id_stack.append(existing_id)
            return None

        new_id = str(uuid.uuid4())
        with self._lock:
            self._prompt_id_stack.append(new_id)
            self._prompt_order.append(new_id)
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

    # ── Raw ────────────────────────────────────────────────────────────

    def get_report(self) -> list[Datapoint]:
        """All raw datapoints in order."""
        with self._lock:
            return self.datapoints.copy()

    def get_prompt_count(self) -> int:
        """Number of top-level prompts seen so far."""
        with self._lock:
            return len(self._prompt_order)

    # ── Totals ────────────────────────────────────────────────────

    def get_totals(self) -> dict[str, float | int]:
        """Returns all totals in a single pass over the datapoints."""
        total_energy = 0.0
        total_co2 = 0.0
        total_input_tokens = 0
        total_output_tokens = 0
        with self._lock:
            for dp in self.datapoints:
                total_energy += dp.estimated_energy_joule
                total_co2 += dp.estimated_co2e_kg
                total_input_tokens += dp.input_token_count
                total_output_tokens += dp.output_token_count
        return {
            "energy": total_energy,
            "co2": total_co2,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
        }

    def get_total_energy(self) -> float:
        return self.get_totals()["energy"]

    def get_total_co2(self) -> float:
        return self.get_totals()["co2"]

    def get_total_input_tokens(self) -> int:
        return self.get_totals()["input_tokens"]

    def get_total_output_tokens(self) -> int:
        return self.get_totals()["output_tokens"]

    # ── Grouped / filtered  Summary ───────────────────────────────────────────────────────

    def get_summary(
        self,
        group_by: Literal["model_name", "agent_name"],
        last_n_prompts: int | None = None,
        last_n_hours: float | None = None,
        ) -> list[GroupSummary]:
        """
        Returns aggregated energy/token summaries grouped by model or agent.

        Args:
            group_by: "model_name" or "agent_name"
            last_n_prompts: if set, only include data from the last N top-level prompts
            last_n_hours: if set, only include data from the last N hours
        """
        points = self._filter_datapoints(last_n_prompts, last_n_hours)
        return self._group_datapoints(points, group_by)

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

