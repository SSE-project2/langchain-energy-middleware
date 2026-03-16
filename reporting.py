from collections import defaultdict

from middleware import Datapoint

def present_results(report: list[Datapoint]) -> None:
    grouped = defaultdict(list)
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
            print(f'  Timestamp: {dp.timestamp}')
            print(f'  Prompt ID: {dp.prompt_id}\n')