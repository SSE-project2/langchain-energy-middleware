from agents import main_agent, tracker
from reporting import present_results

# -----------------------------
# TEST QUERIES
# -----------------------------

response = main_agent.invoke({
    "messages": [
        {"role": "user", "content": """What does this Python program output? 
         ```python
         def mystery(n):
            if n <= 1:
                return n
            return mystery(n-1) + mystery(n-2)

        print(mystery(10))"""}
    ]
})

present_results(tracker.get_report())


response = main_agent.invoke({
    "messages": [
        {"role": "user", "content": """What is the solution for the following mathematical problem? 
         Calculate the exact result of: (452 * 18.5) / 3.2 + 5**3
         """}
    ]
})

present_results(tracker.get_report())
