from langchain.tools import tool
from langchain.agents import create_agent
from langchain_ollama import ChatOllama

from middleware import EnergyMiddleware
from sympy import sympify
import io
import contextlib
import traceback

# Basic multiagent setup for testing
# https://dev.to/fabiothiroki/run-langchain-locally-in-15-minutes-without-a-single-api-key-1j8m
# https://docs.langchain.com/oss/python/langchain/multi-agent/subagents

tracker = EnergyMiddleware()

# -----------------------------
# MATHS SUBAGENT
# -----------------------------

@tool("calculate", description="Evaluate a mathematical expression")
def calculate(expression: str) -> str:
    try:
        result = sympify(expression)
        return str(result)
    except Exception as e:
        return f"Invalid mathematical expression: {expression}\nError: {str(e)}"

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
)

@tool("math_agent", description="Solve mathematical or numerical problems")
def call_math_agent(query: str) -> str:
    result = math_agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    return result["messages"][-1].content


# -----------------------------
# CODING SUBAGENT
# -----------------------------

@tool("run_python", description="Execute Python code and return output or errors")
def run_python(code: str) -> str:

    stdout = io.StringIO()

    try:
        with contextlib.redirect_stdout(stdout):
            exec(code, {})
        return stdout.getvalue() or "Program executed successfully."

    except Exception:
        return traceback.format_exc()

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
)

@tool("coding_agent", description="Answer programming or coding questions")
def call_coding_agent(query: str) -> str:
    result = coding_agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    return result["messages"][-1].content


# -----------------------------
# MAIN ROUTER AGENT
# -----------------------------

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
)
