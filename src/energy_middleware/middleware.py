import datetime
import threading
import uuid
from typing import Any, Literal

from langchain.agents.middleware import AgentState, AgentMiddleware
from langgraph.config import get_config
from langgraph.runtime import Runtime

from .energy_estimation_model import estimate_energy_and_emissions
from .models import EnergyDataPoint, EnergyGroupSummary


class EnergyMiddleware(AgentMiddleware):
    """
    Middleware that tracks token usage, estimates energy consumption,
    and calculates associated CO2 emissions for agent model calls.

    This middleware maintains a thread-safe list of datapoints and
    supports nested agent calls using a prompt ID stack.
    """

    def __init__(self):
        super().__init__()
        self.datapoints: list[EnergyDataPoint] = []
        self._lock = threading.Lock()
        self._prompt_id_stack: list[str] = []
        self._prompt_order: list[str] = []

    @property
    def _current_prompt_id(self) -> str | None:
        """
        Get the current prompt ID from the stack. 

        Returns:
            str | None: The current prompt ID, or None if the stack is empty.
        """
        with self._lock:
            return self._prompt_id_stack[-1] if self._prompt_id_stack else None

    def _filter_datapoints(self, last_n_prompts: int | None, last_n_hours: int | None) -> list[EnergyDataPoint]:
        """
        Filter datapoints based on recent prompts or time.
        
        Args:
            last_n_prompts (int | None): If set, only include datapoints from the last N top-level prompts.
            last_n_hours (float | None): If set, only include datapoints from the last N hours.

        Returns:
            list[EnergyDataPoint]: A filtered list of datapoints based on the provided criteria.
        """
        with self._lock:
            points = self.datapoints.copy()
            recent_ids = set(self._prompt_order[-last_n_prompts:]) if last_n_prompts is not None else None

        if recent_ids is not None:
            points = [dp for dp in points if dp.prompt_id in recent_ids]
        if last_n_hours is not None:
            cutoff = datetime.datetime.now() - datetime.timedelta(hours=last_n_hours)
            points = [dp for dp in points if dp.timestamp >= cutoff]
        return points

    def _group_datapoints(self, points: list[EnergyDataPoint], key: str) -> list[EnergyGroupSummary]:
        """
        Group datapoints by a specified key (model_name or agent_name) and aggregates energy/token usage.
        
        Args:
            points (list[EnergyDataPoint]): List of datapoints to group.
            key (str): Attribute to group by ("model_name" or "agent_name").

        Returns:
            list[EnergyGroupSummary]: A list of aggregated summaries for each group.
        """
        buckets: dict[str, EnergyGroupSummary] = {}
        for dp in points:
            name = getattr(dp, key)
            if name not in buckets:
                buckets[name] = EnergyGroupSummary(
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
            self._prompt_order.append(new_id)
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

        output_datapoint = EnergyDataPoint(
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

    def get_report(self) -> list[EnergyDataPoint]:
        """
        Retrieve a copy of all collected datapoints.

        Returns:
            list[EnergyDataPoint]: A copy of the recorded datapoints.
        """
        with self._lock:
            return self.datapoints.copy()
        
    def get_prompt_count(self) -> int:
        """
        Number of top-level prompts seen so far.
        
        Returns:
            int: Count of unique top-level prompts.
        """
        with self._lock:
            return len(self._prompt_order)

    # ── Totals ────────────────────────────────────────────────────

    def get_totals(self) -> dict[str, float | int]:
        """
        Returns all total accumulated over the collected datapoints.
        
        Returns:
            dict[str, float | int]: A dictionary containing total energy (J), total CO2 (kg), total input tokens, and total output tokens.
        """
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
        """
        Get the total estimated energy consumption of all collected datapoints in Joules.

        Returns:
            float: Total estimated energy in Joules.
        """
        return self.get_totals()["energy"]

    def get_total_co2(self) -> float:
        """
        Get the total estimated CO2 emissions of all collected datapoints in kilograms.

        Returns:
            float: Total estimated CO2 emissions in kilograms.
        """
        return self.get_totals()["co2"]

    def get_total_input_tokens(self) -> int:
        """
        Get the total number of input tokens across all collected datapoints.

        Returns:
            int: Total number of input tokens.
        """
        return self.get_totals()["input_tokens"]

    def get_total_output_tokens(self) -> int:
        """
        Get the total number of output tokens across all collected datapoints.

        Returns:
            int: Total number of output tokens.
        """
        return self.get_totals()["output_tokens"]

    # ── Grouped / filtered  Summary ───────────────────────────────────────────────────────

    def get_summary(
        self,
        group_by: Literal["model_name", "agent_name"],
        last_n_prompts: int | None = None,
        last_n_hours: float | None = None,
        ) -> list[EnergyGroupSummary]:
        """
        Returns aggregated energy/token summaries grouped by model or agent.

        Args:
            group_by: "model_name" or "agent_name"
            last_n_prompts: if set, only include data from the last N top-level prompts
            last_n_hours: if set, only include data from the last N hours
        """
        points = self._filter_datapoints(last_n_prompts, last_n_hours)
        return self._group_datapoints(points, group_by)

