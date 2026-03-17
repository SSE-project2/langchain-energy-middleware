import streamlit as st
from agents import main_agent, tracker
from reporting import present_results

st.title("Agent chat")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

prompt = st.chat_input("Ask something…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    response = main_agent.invoke(
        {"messages": [{"role": "user", "content": prompt}]}
    )
    answer = response["messages"][-1].content

    print(answer)
    present_results(tracker.get_report())

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)
