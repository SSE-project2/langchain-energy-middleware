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

- Report issues via GitHub issues
- Suggest features
- Improve documentation

To build the package locally for testing, you can run:

.. code-block:: bash

    pip install -e .

This will install the package in editable mode, allowing you to test changes without reinstalling. 

Before running some examples, make sure to install Ollama, its models (``qwen3.5:2b``, ``qwen3.5:4b``) and the dependencies for the tutorials first, which can be found in ``tutorials/requirements.txt``. Detailed installation instructions can be found in the tutorial pages.
You can then run the multi-agent tutorial to see the middleware in action:

.. code-block:: bash

    python tutorials/sample_queries.py

Or, if you want to see the streamlit dashboard, you can run:

.. code-block:: bash

    streamlit run tutorials/streamlit_visualisation.py

Lastly, if you want to build the documentation locally, first install the dependencies for documentation:

.. code-block:: bash

    pip install sphinx sphinx-rtd-theme 

Then, you can run:

.. code-block:: bash

    make html

The built documentation will be available in the ``docs/build/html`` directory. You can open the ``index.html`` file in your browser to view it.

*Note*: All of the commands above assume you are in the root directory of the project.