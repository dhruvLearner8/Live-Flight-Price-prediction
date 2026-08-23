"""
Simple UI for the LangGraph buy/wait agent - lets a user pick a route
and date, and see the agent's real recommendation (live SerpAPI price
+ model prediction + Gemini reasoning) without touching any code.

The right panel shows the actual graph structure (matching
agent_graph.png) with the currently-executing node highlighted live
as the agent runs, plus its real input/output below - using
stream_ask() rather than waiting for just the final answer.

Run with: uv run streamlit run streamlit_app.py
"""
import datetime
import os

import streamlit as st

# Streamlit Cloud secrets land in st.secrets, not os.environ - bridge them
# here since agent/*.py reads its API keys via os.environ.
for _key in ("SERPAPI_API_KEY", "GEMINI_API_KEY"):
    if _key in st.secrets:
        os.environ[_key] = st.secrets[_key]

from agent.langgraph_agent import stream_ask
from agent.tools import ROUTE_DISTANCES

st.set_page_config(page_title="Flight Buy/Wait Advisor", page_icon="✈️", layout="wide")

st.title("Flight Buy/Wait Advisor")
st.caption(
    "Combines a live Google Flights price (via SerpAPI) with an ML model trained on "
    "2022 historical fare data. Only these 10 routes are supported."
)

ACTIVE_COLOR = "#ffb84d"    # currently running
VISITED_COLOR = "#b6fcb6"   # already ran
DEFAULT_COLOR = "#f2f0ff"   # not yet reached
NODE_ICONS = {"model": "🧠", "tools": "🔧"}


def make_graph_dot(active: str | None, visited: set[str]) -> str:
    def color_for(node: str) -> str:
        if node == active:
            return ACTIVE_COLOR
        if node in visited:
            return VISITED_COLOR
        return DEFAULT_COLOR

    return f"""
    digraph {{
        rankdir=TB;
        node [fontname="Helvetica"];
        __start__ [shape=ellipse, style=filled, fillcolor="{color_for('__start__')}", label="start"];
        model [shape=box, style="filled,rounded", fillcolor="{color_for('model')}"];
        tools [shape=box, style="filled,rounded", fillcolor="{color_for('tools')}"];
        __end__ [shape=ellipse, style=filled, fillcolor="{color_for('__end__')}", label="end"];
        __start__ -> model;
        model -> tools [style=dashed];
        tools -> model [style=dashed];
        model -> __end__ [style=dashed];
    }}
    """


left_col, right_col = st.columns([1, 1])

with left_col:
    departures = sorted({d for d, a in ROUTE_DISTANCES})
    input_col1, input_col2 = st.columns(2)
    departure_id = input_col1.selectbox("From", departures)

    valid_arrivals = sorted(a for d, a in ROUTE_DISTANCES if d == departure_id)
    arrival_id = input_col2.selectbox("To", valid_arrivals)

    travel_date = st.date_input(
        "Travel date",
        value=datetime.date.today() + datetime.timedelta(days=30),
        min_value=datetime.date.today() + datetime.timedelta(days=1),
    )

    run_clicked = st.button("Get recommendation", type="primary")
    answer_placeholder = st.container()

with right_col:
    st.subheader("Agent execution graph")
    graph_placeholder = st.empty()
    st.subheader("Node trace")
    trace_placeholder = st.empty()

if run_clicked:
    question = (
        f"Should I book a flight from {departure_id} to {arrival_id} "
        f"on {travel_date.isoformat()}?"
    )

    visited = {"__start__"}
    active = "model"
    with right_col:
        graph_placeholder.graphviz_chart(make_graph_dot(active, visited))
        trace_placeholder.empty()
        trace_container = trace_placeholder.container()

    final_answer = None
    model_predictions = None  # {date_str: price} from predict_price_trend, if called
    live_prices_summary = None  # raw text from search_live_flight_prices, if called

    try:
        for event in stream_ask(question):
            node = event["node"]
            visited.add(node)

            # decide what runs next, so the diagram highlights it
            if node == "model":
                active = "tools" if "Decided to call" in event["output"] else "__end__"
            elif node == "tools":
                active = "model"

            if node == "tools" and event["input"].startswith("predict_price_trend"):
                model_predictions = {}
                for line in event["output"].splitlines():
                    date_str, price_str = line.split(": $")
                    model_predictions[date_str] = float(price_str)
            elif node == "tools" and event["input"].startswith("search_live_flight_prices"):
                live_prices_summary = event["output"]

            with right_col:
                graph_placeholder.graphviz_chart(make_graph_dot(active, visited))
                icon = NODE_ICONS.get(node, "⚙️")
                with trace_container:
                    with st.expander(f"{icon} Node: **{node}**", expanded=True):
                        st.markdown("**Input:**")
                        st.code(event["input"], language="text")
                        st.markdown("**Output:**")
                        st.code(event["output"], language="text")

            if node == "model" and "Decided to call" not in event["output"]:
                final_answer = event["output"]

        visited.add("__end__")
        with right_col:
            graph_placeholder.graphviz_chart(make_graph_dot(None, visited))
    except Exception as e:
        with right_col:
            st.error(f"Something went wrong: {e}")

    with left_col:
        if model_predictions:
            st.markdown("### Model's predicted price (next 8 days)")
            st.caption("Trained on 2022 historical data - directional pattern, not exact current price.")
            st.line_chart(model_predictions)
            target_date_str = travel_date.isoformat()
            if target_date_str in model_predictions:
                st.metric(f"Model prediction for {target_date_str}", f"${model_predictions[target_date_str]:.2f}")

        if live_prices_summary:
            cheapest_line = live_prices_summary.splitlines()[0]
            st.markdown("### Live price (Google Flights, via SerpAPI)")
            st.caption("Real, current prices - includes 1 carry-on bag.")
            st.text(cheapest_line)

        with answer_placeholder:
            if final_answer:
                st.markdown("### Recommendation")
                st.markdown(final_answer)
