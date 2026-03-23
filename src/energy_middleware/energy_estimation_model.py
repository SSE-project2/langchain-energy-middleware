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

    # FLOPs per Joule = FLOPs per second / Watts = (29.15e12 FLOPs/s) / 200 W
    # ≈ 1.46e11 FLOPs per Joule (theoretical peak)
    FLOPS_PER_JOULE = 1.46e11  # RTX 4070 FP16 peak efficiency

    # Transformer Inference Compute Approximation used in transformer literature:
    # FLOPs per token ≈ 2 × number_of_parameters (Forward pass only; training typically ≈ 6P)
    # Assumes dense models
    MODEL_PARAMETERS = {
        "qwen3.5:4b": 4_000_000_000,
        "qwen3.5:2b": 2_000_000_000,
    }

    params = MODEL_PARAMETERS.get(model, 0)
    total_tokens = input_tokens + output_tokens

    # Total FLOPs for inference
    total_flops = 2 * params * total_tokens

    # Convert compute to energy
    total_energy = total_flops / FLOPS_PER_JOULE  # Joules

    # Convert energy to CO2
    co2e = total_energy * co2e_per_joule  # kg CO2

    return total_energy, co2e