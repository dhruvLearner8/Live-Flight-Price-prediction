"""
Tools the LangGraph agent will call. Each function here does one real,
verifiable thing (fetch a live price, call our model) - kept separate
from the agent's reasoning logic so each piece can be tested on its
own before wiring them together.
"""
import os
from dataclasses import dataclass

import requests
from dotenv import load_dotenv

load_dotenv()

SERPAPI_URL = "https://serpapi.com/search.json"


@dataclass
class FlightOption:
    price: float
    airline: str
    flight_number: str
    departure_time: str
    arrival_time: str
    stops: int
    travel_class: str


def get_current_price(
    departure_id: str,
    arrival_id: str,
    outbound_date: str,
) -> list[FlightOption]:
    """
    Fetch real, live one-way flight options for a route/date via
    SerpAPI's Google Flights engine.

    departure_id / arrival_id: airport codes, e.g. "LAX", "BOS"
    outbound_date: "YYYY-MM-DD"

    Returns a list of FlightOption, sorted cheapest first. Combines
    SerpAPI's "best_flights" (Google's curated top picks) and
    "other_flights" (everything else) into one flat, sorted list,
    since for our purposes every real option is equally usable data,
    not just the ones Google chose to highlight.
    """
    api_key = os.environ["SERPAPI_API_KEY"]
    params = {
        "engine": "google_flights",
        "departure_id": departure_id,
        "arrival_id": arrival_id,
        "outbound_date": outbound_date,
        "type": "2",  # one-way
        "currency": "USD",
        "hl": "en",
        "api_key": api_key,
    }

    response = requests.get(SERPAPI_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    if "error" in data:
        raise RuntimeError(f"SerpAPI error: {data['error']}")

    options = []
    for section in ("best_flights", "other_flights"):
        for entry in data.get(section, []):
            first_leg = entry["flights"][0]
            last_leg = entry["flights"][-1]
            options.append(FlightOption(
                price=entry["price"],
                airline=first_leg["airline"],
                flight_number=first_leg.get("flight_number", ""),
                departure_time=first_leg["departure_airport"]["time"],
                arrival_time=last_leg["arrival_airport"]["time"],
                stops=len(entry["flights"]) - 1,
                travel_class=first_leg.get("travel_class", "Economy"),
            ))

    options.sort(key=lambda o: o.price)
    return options


if __name__ == "__main__":
    results = get_current_price("LAX", "BOS", "2026-09-24")
    print(f"Found {len(results)} real flight options:\n")
    for opt in results:
        print(f"  ${opt.price:<7.2f} {opt.airline:15s} {opt.flight_number:8s} "
              f"stops={opt.stops} depart={opt.departure_time} class={opt.travel_class}")
