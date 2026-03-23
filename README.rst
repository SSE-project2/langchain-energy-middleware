EnergyMiddleware
================

**EnergyMiddleware** is a lightweight Python middleware for tracking
energy consumption and CO₂ emissions of LLM-based agent systems.

It integrates with agent frameworks to record token usage,
estimated energy consumption, and environmental impact for every model call.


Overview
--------

Modern AI systems, especially multi-agent LLM workflows, can involve
complex chains of model calls. Understanding their computational cost
and environmental impact is increasingly important.

EnergyMiddleware provides:

- Transparent tracking of token usage
- Energy estimation based on model size and compute assumptions
- CO₂ emissions estimation using global carbon intensity


Key Features
------------

- Plug-and-play middleware for agent systems
- Tracks input/output tokens, energy (J), and CO₂ (kg)
- Supports nested agent calls via prompt tracking
- Works with multi-agent architectures
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


The energy middleware tracker will be called after all model calls, including nested ones, and will log token usage, energy, and CO₂ for each call.

Thereafter, each model call produces a :class:`middleware.Datapoint` containing:

- Token usage (input/output)
- Estimated energy consumption
- Estimated CO₂ emissions
- Model name and timestamp
- Associated prompt ID and agent


Use Cases
---------

EnergyMiddleware may be useful for:

- 🔬 Research on efficient AI systems
- 🌱 Measuring environmental impact of LLMs
- 🤖 Adaptive multi-agent systems based on real-time energy usage 
- 🧪 Profiling experimental pipelines


Contributing
------------

Contributions are welcome! You can help by:

- Reporting bugs via GitHub Issues
- Suggesting new features
- Improving documentation


**Development Setup**:

From the root directory, install the package in editable mode:

.. code-block:: bash

    pip install -e .


**Running Tutorials**:

Before running examples:

- Install Ollama
- Download models:
  
  - ``qwen3.5:2b``
  - ``qwen3.5:4b``

- Install dependencies:

.. code-block:: bash

    pip install -r tutorials/requirements.txt

For detailed instructions regarding the installation process, please refer to the multi-agent tutorial documentation.

**Example Scripts**:

Run the multi-agent example:

.. code-block:: bash

    python tutorials/sample_queries.py


Or, launch the dashboard:

.. code-block:: bash

    streamlit run tutorials/streamlit_visualisation.py


**Building the Documentation**:

First, install documentation dependencies:

.. code-block:: bash

    pip install sphinx sphinx-rtd-theme

Then, build the docs from the root directory:

.. code-block:: bash

    make html

The built documentation will be available in the ``docs/build/html`` directory. You can open the ``index.html`` file in your browser to view it.