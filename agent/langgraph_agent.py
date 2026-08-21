"""
The actual LangGraph buy/wait agent. Unlike reasoning.py (which always
calls both tools in a fixed sequence), this lets the LLM itself decide
which tools to call and when, based on the user's question - using
LangGraph's prebuilt ReAct-style tool-calling agent.
"""
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from agent.tools import get_current_price, get_price_prediction_next_7_days

load_dotenv()

MODEL_NAME = "gemini-3.5-flash"

SYSTEM_PROMPT = """You are a flight buy/wait advisor. A traveler will ask you about a \
specific route and date. Your job is to help them decide whether to book now or wait.

You have two tools:
- search_live_flight_prices: real, current flight prices from Google Flights (accurate 2026 dollars)
- predict_price_trend: our ML model's predicted price for the target date and the following \
7 days. This model was trained on 2022 historical data - prices have generally risen since \
then due to inflation and market changes, so treat its dollar amounts as directional/relative \
signals about which days tend to be cheaper or pricier within the week, NOT as exact current prices.

Always call search_live_flight_prices for the target date to get real current prices. Also call \
predict_price_trend to see whether the target date looks cheap or expensive relative to the \
surrounding week, according to historical patterns.

Give a clear plain-English recommendation: BOOK NOW or WAIT, with 2-4 sentences of reasoning \
citing specific real numbers. Be honest about uncertainty - don't overstate confidence the data \
doesn't support. Only 10 routes are supported: ATL, LAX, LGA, BOS, JFK, DFW, ORD, DTW, EWR, CLT \
(pairs connecting to/from LAX, plus LGA-ORD). If asked about an unsupported route, say so."""


@tool
def search_live_flight_prices(departure_id: str, arrival_id: str, outbound_date: str) -> str:
    """Fetch real, live one-way flight prices for a route and date via Google Flights.

    departure_id / arrival_id: airport codes, e.g. "LAX", "BOS"
    outbound_date: "YYYY-MM-DD"
    """
    options = get_current_price(departure_id, arrival_id, outbound_date)
    lines = [
        f"${o.price:.2f}  {o.airline} {o.flight_number}  stops={o.stops}  "
        f"depart={o.departure_time}  class={o.travel_class}"
        for o in options[:15]
    ]
    return "\n".join(lines)


@tool
def predict_price_trend(
    departure_id: str,
    arrival_id: str,
    outbound_date: str,
    airline: str = "Delta",
    stops: int = 0,
    departure_hour: int = 8,
) -> str:
    """Get our ML model's predicted price for the target date and the following 7 days.

    Useful for seeing whether the target date looks relatively cheap or expensive
    compared to nearby days, based on historical patterns (route, season, day of week).
    departure_id / arrival_id: airport codes, e.g. "LAX", "BOS"
    outbound_date: "YYYY-MM-DD"
    """
    predictions = get_price_prediction_next_7_days(
        departure_id, arrival_id, outbound_date,
        airline=airline, stops=stops, departure_hour=departure_hour,
    )
    return "\n".join(f"{d}: ${p:.2f}" for d, p in predictions.items())


def build_agent():
    model = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=os.environ["GEMINI_API_KEY"],
    )
    return create_agent(
        model=model,
        tools=[search_live_flight_prices, predict_price_trend],
        system_prompt=SYSTEM_PROMPT,
    )


def _extract_text(content) -> str:
    """Gemini's response content can be a plain string or a list of
    content blocks (text + metadata) - normalize to plain text either way."""
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "") for block in content if isinstance(block, dict)
    )


def ask(question: str) -> str:
    agent = build_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    return _extract_text(result["messages"][-1].content)


if __name__ == "__main__":
    answer = ask("Should I book a flight from LAX to BOS on 2026-09-24?")
    print(answer)
