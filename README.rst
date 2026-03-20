EnergyMiddleware
================

**EnergyMiddleware** is a lightweight Python middleware for tracking
energy consumption and CO₂ emissions of LLM-based agent systems.

It integrates with agent frameworks to records token usage,
estimated energy consumption, and environmental impact for every model call.


Overview
--------

Modern AI systems, especially multi-agent LLM workflows, can involve
complex chains of model calls. Understanding their **computational cost**
and **environmental impact** is increasingly important.

EnergyMiddleware provides:

- Transparent tracking of token usage
- Energy estimation based on model size and compute assumptions
- CO₂ emissions estimation using global carbon intensity


Key Features
------------

- **Plug-and-play middleware** for agent systems
- Tracks **input/output tokens**, energy (J), and CO₂ (kg)
- Supports **nested agent calls** via prompt tracking
- Works with **multi-agent architectures**
- Thread-safe and lightweight
- Provides structured outputs via :class:`middleware.Datapoint`


Installation
------------

.. code-block:: bash

    pip install energy-middleware


Quick Example
-------------

Start tracking energy usage in your agent system in just a few lines:

.. code-block:: python

    from langchain.agents import create_agent
    from langchain_ollama import ChatOllama

    from energy_middleware.middleware import EnergyMiddleware
    from energy_middleware.reporting import present_results

    tracker = EnergyMiddleware()

    # Attach to your agent
    agent = create_agent(
        model=ChatOllama(model="qwen3.5:2b"),
        middleware=[tracker],
        name="MyAgent"
    )

    # Run your system
    agent.invoke({
        "messages": [
            {
                "role": "user", 
                "content": "What is the capital of Italy?"
            }
        ]
    })

    # Print results
    present_results(tracker.get_report())

The energy middleware tracker will be called after all model calls, including nested ones, and will log token usage, energy, and CO₂ for each call. The ``present_results`` function can be used to display the collected data in a readable format.


Example Output
--------------

Each model call produces a :class:`middleware.Datapoint` containing:

- Token usage (input/output)
- Estimated energy consumption
- Estimated CO₂ emissions
- Model name and timestamp
- Associated prompt ID and agent


Use Cases
---------

EnergyMiddleware is useful for:

- 🔬 Research on **efficient AI systems**
- 🌱 Measuring **environmental impact of LLMs**
- 🤖 Adaptable **multi-agent systems** based on real-time estimated consumption
- 🧪 Profiling experimental pipelines


Contributing
------------

Contributions are welcome! Feel free to:

- Report issues
- Suggest features
- Improve documentation
