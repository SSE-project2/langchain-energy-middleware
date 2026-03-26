import datetime

from pydantic import BaseModel


class EnergyDataPoint(BaseModel):
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


class EnergyGroupSummary(BaseModel):
    """
    Aggregated summary of energy and token usage for a group of datapoints.

    Attributes:
        name (str): Name of the group.
        total_energy_joule (float): Total estimated energy usage in Joules
        total_co2e_kg (float): Total estimated CO2 emissions in kilograms.
        total_input_tokens (int): Total number of input tokens.
        total_output_tokens (int): Total number of output tokens.
    """
    name: str
    total_energy_joule: float
    total_co2e_kg: float
    total_input_tokens: int
    total_output_tokens: int
    datapoint_count: int