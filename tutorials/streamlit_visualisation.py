import pandas as pd
import plotly.express as px
import streamlit as st

from jamanota import EnergyGroupSummary
from sample_agents import main_agent, tracker

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

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)

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
        col3, col4 = st.columns(2)
        col3.metric("Input tokens", totals["input_tokens"])
        col4.metric("Output tokens", totals["output_tokens"])

        st.divider()

        filter_by = st.radio("Filter by", ["prompts", "hours"], horizontal=True)
        if filter_by == "prompts":
            n = st.slider("Last N prompts", 0, max(tracker.get_prompt_count(), 1),
                          min(5, tracker.get_prompt_count()))
            filter_kwargs = {"last_n_prompts": n}
            filter_label = f"last {n} prompt(s)"
        else:
            n = st.slider("Last N hours", 0, 24, 1)
            filter_kwargs = {"last_n_hours": n}
            filter_label = f"last {n}h"

        metric = st.radio("Show", ["total", "average"], horizontal=True)

        tab1, tab2, tab3 = st.tabs(["By model", "By agent", "Raw"])

        def show_chart(summaries: list[EnergyGroupSummary], title: str, key: str) -> None:
            if not summaries:
                st.caption("No data for this selection.")
                return
            df = pd.DataFrame([s.model_dump() for s in summaries])

            if metric == "average":
                df["energy_joule"] = df["total_energy_joule"] / df["datapoint_count"]
                df["co2e_kg"] = df["total_co2e_kg"] / df["datapoint_count"]
            else:
                df["energy_joule"] = df["total_energy_joule"]
                df["co2e_kg"] = df["total_co2e_kg"]

            col1, col2 = st.columns(2)
            with col1:
                fig1 = px.bar(df, x="name", y="energy_joule", title=f"Energy — {title}",
                              labels={"energy_joule": "energy (J)", "name": ""})
                fig1.update_layout(margin=dict(t=30, b=10), height=200)
                st.plotly_chart(fig1, width='stretch', key=f"energy_{key}")
            with col2:
                fig2 = px.bar(df, x="name", y="co2e_kg", title=f"CO2e — {title}",
                              labels={"co2e_kg": "CO2e (kg)", "name": ""})
                fig2.update_layout(margin=dict(t=30, b=10), height=200)
                st.plotly_chart(fig2, width='stretch', key=f"co2_{key}")

            st.dataframe(df, width='stretch', hide_index=True)

        with tab1:
            show_chart(tracker.get_summary(group_by="model_name", **filter_kwargs),
                       filter_label, key="model")
        with tab2:
            show_chart(tracker.get_summary(group_by="agent_name", **filter_kwargs),
                       filter_label, key="agent")
        with tab3:
            st.dataframe(
                pd.DataFrame([{
                    "time": dp.timestamp.strftime("%H:%M:%S"),
                    "agent": dp.agent_name,
                    "model": dp.model_name,
                    "energy (J)": dp.estimated_energy_joule,
                    "in_tokens": dp.input_token_count,
                    "out_tokens": dp.output_token_count,
                    "message": dp.message,
                } for dp in report]),
                width='stretch', hide_index=True,
            )