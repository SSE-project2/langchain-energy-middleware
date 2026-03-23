from sample_agents import main_agent, tracker
from sample_reporting import present_results

# -----------------------------------------------
# TEST QUERIES USING AGENTS FROM sample_agents.py
# -----------------------------------------------

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

main_agent.invoke({
    "messages": [
        {"role": "user", "content": """ 
         Calculate the exact result of: (452 * 18.5) / 3.2 + 5**3
         """}
    ]
})

present_results(tracker.get_report())
