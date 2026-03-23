Using EnergyMiddleware with Multi-Agent Setup
=======================================================

This tutorial demonstrates how to use **EnergyMiddleware** to track energy and CO₂ usage
in a multi-agent LangChain setup with Python and math subagents. 

Prerequisites
-------------

We will be using Ollama models in this tutorial, so make sure you have it installed. It can be downloaded from https://ollama.com/.
Additionally, make sure you have the following modules installed:

- ``langchain``
- ``langchain_ollama``
- ``sympy``
- ``energy_middleware``

During this tutorial, we will use ``qwen3.5:4b`` for the subagents, and ``qwen3.5:2b`` for the main agent, to simulate a situation where the main agent is slightly less powerful than the subagents. However, you can use any models you have available.
To set up the Ollama models, you can run the following commands in your terminal:

.. code-block:: bash

    ollama pull qwen3.5:4b
    ollama pull qwen3.5:2b
    
These can then be started by running:

.. code-block:: bash

    ollama run qwen3.5


Setup EnergyMiddleware
----------------------

Initialize the tracker:

.. code-block:: python

    from energy_middleware import EnergyMiddleware

    tracker = EnergyMiddleware()

We will use this ``tracker`` as a middleware in all of our agents. As such, it will be invoked after all model calls, recording token usage to estimate energy and CO₂ consumption.


Math Subagent
-------------

We now define a simple subagent, simulating a math expert that can evaluate mathematical expressions using the ``sympy`` library. This agent will be called by the main agent when it receives a math-related query.

First, define a tool for evaluating mathematical expressions:

.. code-block:: python

    from sympy import sympify
    from langchain.tools import tool

    @tool("calculate", description="Evaluate a mathematical expression")
    def calculate(expression: str) -> str:
        try:
            result = sympify(expression)
            return str(result)
        except Exception as e:
            return f"Invalid mathematical expression: {expression}\\nError: {str(e)}"

- *Note*: You can, of course, implement more complex math capabilities, but this serves as a simple example for demonstration purposes.

This tool will be used by the math agent to perform calculations when needed:

.. code-block:: python

    from langchain.agents import create_agent
    from langchain_ollama import ChatOllama

    MATH_SYSTEM_PROMPT = """
    You are a mathematics expert.

    You specialize in solving mathematical and numerical problems such as:
    - arithmetic
    - algebra
    - basic calculus
    - statistics

    You have access to the following tool:

    calculate
    - evaluates a mathematical expression and returns the result.

    Use the calculate tool whenever an exact numerical computation is needed.

    Explain your reasoning clearly. If the problem involves computation, you may first reason about the steps and then use the tool to obtain the final value.

    If the question is not related to mathematics, say that you are a math specialist and cannot answer it.
    """

    math_agent = create_agent(
        model=ChatOllama(model="qwen3.5:4b"),
        tools=[calculate],
        system_prompt=MATH_SYSTEM_PROMPT,
        middleware=[tracker],
        name="math_agent",
    )

Notice that we include ``middleware=[tracker]`` in the math agent, ensuring that **all calls** to this subagent are tracked for energy and CO₂ usage.



Coding Subagent
---------------

To complement the math agent, and to simulate a situation where there are multiple subagents, we can create a coding agent that can execute Python code. This agent will be called by the main agent when it receives programming-related queries.

So, we first create a tool to execute Python code:

.. code-block:: python

    import io
    import contextlib
    import traceback

    @tool("run_python", description="Execute Python code and return output or errors")
    def run_python(code: str) -> str:
        stdout = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout):
                exec(code, {})
            return stdout.getvalue() or "Program executed successfully."
        except Exception:
            return traceback.format_exc()

- *Note*: Once again, this is a simple implementation for demonstration purposes, as it allows us to differentiate between the two subagents.

Then, create the coding agent that uses this tool, as well as the ``tracker`` middleware:

.. code-block:: python

    CODING_SYSTEM_PROMPT = """
    You are an expert software engineer.

    You help with:
    - writing code
    - debugging programs
    - explaining program behavior
    - fixing errors

    You have access to the tool:

    run_python
    - executes Python code and returns the printed output or any errors.

    Use this tool when running the code would help understand what it does or diagnose a problem.

    Prefer Python unless another language is explicitly requested.

    If the question is unrelated to programming or software development, say that you are a coding specialist and cannot answer it.
    """

    coding_agent = create_agent(
        model=ChatOllama(model="qwen3.5:4b"),
        tools=[run_python],
        system_prompt=CODING_SYSTEM_PROMPT,
        middleware=[tracker],
        name="coding_agent",
    )



Main Router Agent
-----------------

Before we create the main agent, we define two helper functions that will allow the main agent to call the math and coding subagents when needed. These functions will also be decorated with `@tool` so that they can be used as tools by the main agent.

Helper function to call the math agent:

.. code-block:: python

    @tool("math_agent", description="Solve mathematical or numerical problems")
    def call_math_agent(query: str) -> str:
        result = math_agent.invoke({
            "messages": [{"role": "user", "content": query}]
        })
        return result["messages"][-1].content

Helper function to call the coding agent:

.. code-block:: python

    @tool("coding_agent", description="Answer programming or coding questions")
    def call_coding_agent(query: str) -> str:
        result = coding_agent.invoke({
            "messages": [{"role": "user", "content": query}]
        })
        return result["messages"][-1].content

Now, we are ready to create the main agent. In our example, the main agent is a slightly smaller model (2 billion parameters) compared to the subagents(4 billion parameters).
This agent can either choose to answer questions on its own if they are simple enough, or delegate to the math or coding subagents when it detects that the query is related to math or programming, respectively.
Of course, the main agent can also delegate to both subagents if the query requires both math and coding expertise.

.. code-block:: python

    MAIN_SYSTEM_PROMPT = """
    You are an intelligent assistant that answers the user's question. You may either answer directly or delegate the task to a specialized agent if their expertise would help.

    Available specialized agents:

    math_agent
    - expert in mathematics and numerical reasoning
    - has access to a calculator tool for evaluating numerical expressions

    coding_agent
    - expert in programming and software engineering
    - can execute Python code to inspect program behavior or debug errors

    If the question is simple or unrelated to these domains, answer it yourself. Otherwise route the question to the most appropriate agent.
    """

    main_agent = create_agent(
        model=ChatOllama(model="qwen3.5:2b"),
        tools=[call_math_agent, call_coding_agent],
        system_prompt=MAIN_SYSTEM_PROMPT,
        middleware=[tracker],
        name="main_agent",
    )

As usual, we include ``middleware=[tracker]`` to ennsure that calls are logged for energy/CO₂ tracking.


Testing Queries
---------------

Before we run some test queries, let's define a helper function to display the energy and CO₂ report in a readable format:

.. code-block:: python

    from collections import defaultdict

    from energy_middleware import EnergyDataPoint

    def present_results(report: list[EnergyDataPoint]) -> None:
        """
        Print a human-readable summary of energy and CO2 usage for a list of datapoints.

        Groups the datapoints by `prompt_id` so that nested agent/model calls
        are shown together.

        Attributes:
            report (list[EnergyDataPoint]): A list of `EnergyDataPoint` instances collected
                from the `EnergyMiddleware`.
        """
        grouped: dict[str, list[EnergyDataPoint]] = defaultdict(list[EnergyDataPoint])
        for dp in report:
            grouped[dp.prompt_id].append(dp)

        for prompt_id, points in grouped.items():
            print(f"\nPrompt [{prompt_id}]:")
            for dp in points:
                print(f"  [{dp.model_name}] {dp.message}")
                print(f'  Energy: {dp.estimated_energy_joule} J')
                print(f'  CO2: {dp.estimated_co2e_kg} gCO2e')
                print(f'  Input: {dp.input_token_count} tokens')
                print(f'  Output: {dp.output_token_count} tokens')
                print(f'  Timestamp: {dp.timestamp}\n')
                print(f'  Prompt ID: {dp.prompt_id}\n')
                print(f'  Agent Name: {dp.agent_name}\n')


Example: execute a Python program and track energy usage:

.. code-block:: python

    main_agent.invoke({
        "messages": [
            {"role": "user", "content": """
            What does this Python program output?
             ```python
             def mystery(n):
                if n <= 1:
                    return n
                return mystery(n-1) + mystery(n-2)

            print(mystery(10))```

            """}
        ]
    })

    present_results(tracker.get_report())

- Via the function above, the ``present_results`` function nicely prints **energy, CO₂, tokens, prompt IDs, and agent name** for each call.

Example: solve a math problem:

.. code-block:: python

    main_agent.invoke({
        "messages": [
            {"role": "user", "content": """
             Calculate the exact result of: (452 * 18.5) / 3.2 + 5**3
             """}
        ]
    })

    present_results(tracker.get_report())
