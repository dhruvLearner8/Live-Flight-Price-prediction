"""
Agent reasoning step: takes the model's 8-day price predictions, plus
today's real live prices from SerpAPI, and asks Gemini to produce a
plain-English buy/wait recommendation.

The model is explicitly told the prediction data is 2022-trained and
may sit below current real prices due to inflation - it's asked to
weigh the *relative* day-to-day pattern from the model against the
*absolute* anchor from the live price, rather than trusting either
number blindly.
"""
import os

from dotenv import load_dotenv
from google import genai

from agent.tools import (
    FlightOption,
    get_current_price,
    get_price_prediction_next_7_days,
)

load_dotenv()

MODEL_NAME = "gemini-3.5-flash"


def build_prompt(
    departure_id: str,
    arrival_id: str,
    target_date: str,
    predictions: dict[str, float],
    live_options: list[FlightOption],
) -> str:
    predictions_text = "\n".join(
        f"  {d}: ${p:.2f}" for d, p in predictions.items()
    )
    live_text = "\n".join(
        f"  ${o.price:.2f}  {o.airline} {o.flight_number}  stops={o.stops}  "
        f"depart={o.departure_time}  class={o.travel_class}"
        for o in live_options[:10]
    )

    return f"""You are helping a traveler decide whether to book a flight now or wait.

Route: {departure_id} -> {arrival_id}
Target flight date: {target_date}

Our machine learning model's predicted price for the target date and the following 7 days
(trained on 2022 historical Expedia flight data - prices generally rose since then due to
inflation and market changes, so treat these as relative/directional signals about which
days tend to be cheaper or pricier, not as exact current dollar amounts):
{predictions_text}

Today's REAL, live flight prices for the target date ({target_date}), fetched just now from
Google Flights (these ARE current, accurate 2026 dollar amounts):
{live_text}

Based on:
1. Whether the target date looks cheap or expensive relative to the model's predicted
   pattern for the surrounding week (even though the model's absolute dollars are dated,
   its relative day-to-day pattern is still meaningful)
2. Whether the live prices for the target date look reasonable or high compared to what's
   actually available today

Give a clear, plain-English recommendation: should the traveler BOOK NOW or WAIT?
Explain your reasoning in 2-4 sentences, mentioning specific numbers. Be honest about
uncertainty where it exists - don't overstate confidence the data doesn't support."""


def get_buy_wait_recommendation(
    departure_id: str,
    arrival_id: str,
    target_date: str,
    airline: str,
    stops: int = 0,
    departure_hour: int = 8,
) -> str:
    predictions = get_price_prediction_next_7_days(
        departure_id, arrival_id, target_date,
        airline=airline, stops=stops, departure_hour=departure_hour,
    )
    live_options = get_current_price(departure_id, arrival_id, target_date)

    prompt = build_prompt(departure_id, arrival_id, target_date, predictions, live_options)

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
    return response.text


if __name__ == "__main__":
    recommendation = get_buy_wait_recommendation(
        "LAX", "BOS", "2026-09-24", airline="Delta", stops=0, departure_hour=8,
    )
    print(recommendation)
