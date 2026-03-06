from middleware import Datapoint

def get_total_energy_usage(outputs: list[Datapoint]) -> float:
    return sum(dp.estimated_energy_joule for dp in outputs)