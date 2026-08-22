"""
Tools the LangGraph agent will call. Each function here does one real,
verifiable thing (fetch a live price, call our model) - kept separate
from the agent's reasoning logic so each piece can be tested on its
own before wiring them together.
"""
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import requests
from dotenv import load_dotenv

load_dotenv()

SERPAPI_URL = "https://serpapi.com/search.json"
PREDICTION_API_URL = "https://flight-price-api-j1sh.onrender.com/predict"

# median distance (miles) per route, from our own training data - lets
# callers just give a route/date without needing to know the distance
ROUTE_DISTANCES = {
    ("ATL", "LAX"): 1954, ("LAX", "ATL"): 1963,
    ("BOS", "LAX"): 2643, ("LAX", "BOS"): 2643,
    ("CLT", "LAX"): 2266, ("LAX", "CLT"): 2255,
    ("DFW", "LAX"): 1291, ("LAX", "DFW"): 1242,
    ("DTW", "LAX"): 2188, ("LAX", "DTW"): 2076,
    ("EWR", "LAX"): 2466, ("LAX", "EWR"): 2466,
    ("JFK", "LAX"): 2458, ("LAX", "JFK"): 2466,
    ("LAX", "LGA"): 2573, ("LGA", "LAX"): 2573,
    ("LAX", "ORD"): 1751, ("ORD", "LAX"): 1751,
    ("LGA", "ORD"): 720, ("ORD", "LGA"): 720,
}


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
    include_airlines: list[str] | None = None,
) -> list[FlightOption]:
    """
    Fetch real, live one-way flight options for a route/date via
    SerpAPI's Google Flights engine. Always includes 1 carry-on bag in
    the search (bags=1), so prices reflect the cost of a real trip a
    traveler would actually take, not a fare that looks artificially
    cheap because it excludes carry-on baggage.

    departure_id / arrival_id: airport codes, e.g. "LAX", "BOS"
    outbound_date: "YYYY-MM-DD"
    include_airlines: optional list of 2-letter IATA airline codes to
        restrict results to, e.g. ["DL", "AA", "AS"] for Delta,
        American, Alaska. None (default) returns all airlines.

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
        "bags": "1",  # at least 1 carry-on bag included
        "exclude_basic": "true",  # bags=1 alone isn't reliably enforced for every
                                   # carrier (e.g. Frontier); this guarantees free
                                   # seat selection + carry-on are actually included
        "api_key": api_key,
    }
    if include_airlines:
        params["include_airlines"] = ",".join(include_airlines)

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


def get_price_prediction(
    departure_id: str,
    arrival_id: str,
    outbound_date: str,
    airline: str,
    stops: int = 0,
    is_basic_economy: bool = False,
    departure_hour: int = 8,
) -> float:
    """
    Call our deployed model (Render) for the expected/typical price of
    this itinerary. Computes the date-derived features (days until
    departure, day of week, month) internally so the caller only needs
    to supply what a real search would naturally have: route, date,
    airline, stops, cabin, and roughly what time of day.
    """
    if (departure_id, arrival_id) not in ROUTE_DISTANCES:
        raise ValueError(
            f"No known distance for route {departure_id}-{arrival_id}. "
            f"Model was only trained on these 20 routes: {sorted(ROUTE_DISTANCES.keys())}"
        )

    travel_date = datetime.strptime(outbound_date, "%Y-%m-%d").date()
    days_until_departure = (travel_date - date.today()).days

    payload = {
        "startingAirport": departure_id,
        "destinationAirport": arrival_id,
        "primaryAirline": airline,
        "flightDayOfWeek": travel_date.isoweekday(),
        "flightMonth": travel_date.month,
        "departureHour": departure_hour,
        "numStops": stops,
        "totalTravelDistance": ROUTE_DISTANCES[(departure_id, arrival_id)],
        "daysUntilDeparture": days_until_departure,
        "isBasicEconomy": is_basic_economy,
    }

    response = requests.post(PREDICTION_API_URL, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()["predictedFare"]


def get_price_prediction_next_7_days(
    departure_id: str,
    arrival_id: str,
    outbound_date: str,
    airline: str,
    stops: int = 0,
    is_basic_economy: bool = False,
    departure_hour: int = 8,
) -> dict[str, float]:
    """
    Model's predicted price for the target flight date plus the
    following 7 days (8 dates total) - one call to get_price_prediction
    per date. Keys are "YYYY-MM-DD" strings, in date order.
    """
    start_date = datetime.strptime(outbound_date, "%Y-%m-%d").date()
    predictions = {}
    for offset in range(8):
        this_date = start_date + timedelta(days=offset)
        this_date_str = this_date.strftime("%Y-%m-%d")
        predictions[this_date_str] = get_price_prediction(
            departure_id, arrival_id, this_date_str,
            airline=airline, stops=stops,
            is_basic_economy=is_basic_economy, departure_hour=departure_hour,
        )
    return predictions


if __name__ == "__main__":
    results = get_current_price("LAX", "BOS", "2026-09-24")
    print(f"Found {len(results)} real flight options:\n")
    for opt in results:
        print(f"  ${opt.price:<7.2f} {opt.airline:15s} {opt.flight_number:8s} "
              f"stops={opt.stops} depart={opt.departure_time} class={opt.travel_class}")

    print()
    predicted = get_price_prediction("LAX", "BOS", "2026-09-24", "Delta", stops=0, departure_hour=8)
    print(f"Model's predicted price (Delta, ~8am, main cabin): ${predicted:.2f}")
