Visualising Energy Usage with Streamlit
=======================================

In this tutorial, we build a simple **interactive dashboard** using
`Streamlit <https://streamlit.io/>`_ to visualise the collected data in real time.

This allows you to:

- Monitor energy usage as you interact with agents
- Compare models and agents
- Explore usage over time
- Inspect raw datapoints


Prerequisites
-------------

Make sure you have installed:

- ``streamlit``
- ``plotly``
- ``pandas``

You should also already have:

- A working agent setup (see "Using EnergyMiddleware with Multi-Agent Setup" tutorial). In this example we import it from a ``sample_agents`` file previously created from the multi-agent tutorial, but you can use your own
- A shared :class:`energy_middleware.middleware.EnergyMiddleware` instance (``tracker``)

Overview
--------

The dashboard consists of two main parts:

1. **Chat interface** (main panel)

   - Send prompts to the agent
   - Display responses

2. **Energy dashboard** (sidebar)

   - Aggregate metrics (energy, CO₂, tokens)
   - Charts grouped by model or agent
   - Raw datapoint inspection


Basic Streamlit App Structure
-----------------------------

We start by setting up a simple chat interface:

.. code-block:: python

    import streamlit as st
    from sample_agents import main_agent, tracker

    st.title("Agent chat")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

- ``st.session_state`` stores conversation history.
- This allows the UI to persist messages across interactions.


Sending a Prompt
----------------

We use Streamlit's chat input to send messages to the agent:

.. code-block:: python

    prompt = st.chat_input("Ask something…")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        response = main_agent.invoke(
            {"messages": [{"role": "user", "content": prompt}]}
        )

        answer = response["messages"][-1].content

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

- Each prompt triggers the **full agent pipeline**.
- Because all agents use ``EnergyMiddleware``, every call is tracked automatically.


Accessing Collected Data
------------------------

At any point, we can retrieve the tracked data:

.. code-block:: python

    report = tracker.get_report()

This returns a list of :class:`energy_middleware.models.EnergyDataPoint` objects, each representing
a single model call.


Sidebar: Aggregate Metrics
--------------------------

We display high-level metrics in the sidebar:

.. code-block:: python

    with st.sidebar:
        st.header("Energy usage")

        report = tracker.get_report()

        if not report:
            st.caption("No data yet — send a message first.")
        else:
            totals = tracker.get_totals()

            col1, col2 = st.columns(2)
            col1.metric("Total energy", f"{totals['energy']:.2f} J")
            col2.metric("Total CO2e", f"{totals['co2']:.2e} kg")

- These metrics give a quick overview of total usage.
- Useful for comparing different prompts or sessions.


Filtering Data
--------------

To explore recent activity, we allow filtering:

.. code-block:: python

    filter_by = st.radio("Filter by", ["prompts", "hours"])

    if filter_by == "prompts":
        n = st.slider("Last N prompts", 1, tracker.get_prompt_count(), 5)
        filter_kwargs = {"last_n_prompts": n}
    else:
        n = st.slider("Last N hours", 1, 24, 1)
        filter_kwargs = {"last_n_hours": n}

- This enables analysis of **recent usage patterns**.
- Filtering is applied before aggregation.


Visualising Data
----------------

We can aggregate datapoints and visualise them using Plotly:

.. code-block:: python

    summaries = tracker.get_summary(group_by="model_name", **filter_kwargs)

    import pandas as pd
    import plotly.express as px

    df = pd.DataFrame([s.model_dump() for s in summaries])

    fig = px.bar(df, x="name", y="total_energy_joule")
    st.plotly_chart(fig)

- ``get_summary`` groups datapoints (e.g. by model or agent).
- Plotly provides interactive charts for exploration.


Comparing Models and Agents
---------------------------

We can create tabs to compare different groupings:

.. code-block:: python

    tab1, tab2 = st.tabs(["By model", "By agent"])

    with tab1:
        summaries = tracker.get_summary(group_by="model_name")
        ...

    with tab2:
        summaries = tracker.get_summary(group_by="agent_name")
        ...

- This helps identify which models or agents are most expensive.


Inspecting Raw Data
-------------------

For debugging and deeper inspection:

.. code-block:: python

    st.dataframe([
        {
            "time": dp.timestamp,
            "agent": dp.agent_name,
            "model": dp.model_name,
            "energy": dp.estimated_energy_joule,
            "tokens_in": dp.input_token_count,
            "tokens_out": dp.output_token_count,
        }
        for dp in report
    ])

- This gives full visibility into individual calls.


Running the App
---------------

Save the script (e.g. ``app.py``) and run:

.. code-block:: bash

    streamlit run app.py

Then open the provided local URL in your browser.
