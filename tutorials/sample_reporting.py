from collections import defaultdict

from energy_middleware.middleware import Datapoint

def present_results(report: list[Datapoint]) -> None:
    """
    Print a human-readable summary of energy and CO2 usage for a list of datapoints.

    Groups the datapoints by `prompt_id` so that nested agent/model calls
    are shown together.

    Attributes:
        report (list[Datapoint]): A list of `Datapoint` instances collected
            from the `EnergyMiddleware`.
    """
    grouped: dict[str, list[Datapoint]] = defaultdict(list[Datapoint])
    for dp in report:
        grouped[dp.prompt_id].append(dp)

    for prompt_id, points in grouped.items():
        print(f"\nPrompt [{prompt_id}]:")
        for dp in points:
            print(f"  [{dp.model_name}] {dp.message}")  # If we have multiple models for the sub-prompts, that will change here
            print(f'  Energy: {dp.estimated_energy_joule} J')
            print(f'  CO2: {dp.estimated_co2e_kg} gCO2e')
            print(f'  Input: {dp.input_token_count} tokens')
            print(f'  Output: {dp.output_token_count} tokens')
            print(f'  Timestamp: {dp.timestamp}\n')
            print(f'  Prompt ID: {dp.prompt_id}\n')
            print(f'  Agent Name: {dp.agent_name}\n')
