"""
One-off: compute the live-price calibration factor by comparing real
SerpAPI prices against our model's predictions for the *same* exact
itinerary (same airline, stops, departure hour), across a handful of
different routes/dates. Averaging several real comparisons (not just
one) avoids the mistake we made earlier trusting a single noisy point.

Uses only a few SerpAPI searches (each search returns many flight
options we can all use), to stay well within the free-tier quota.
"""
from statistics import mean

from agent.tools import get_current_price, get_price_prediction

# a handful of different routes/dates, ~30-45 days out, spread across
# our 20 trained routes for a more robust estimate
SEARCHES = [
    ("LAX", "BOS", "2026-09-24"),
    ("ATL", "LAX", "2026-10-05"),
    ("ORD", "LGA", "2026-09-30"),
]

MAX_OPTIONS_PER_SEARCH = 5  # cheapest N non-outlier options per search


def main():
    ratios = []

    for departure_id, arrival_id, outbound_date in SEARCHES:
        print(f"\n=== {departure_id} -> {arrival_id} on {outbound_date} ===")
        options = get_current_price(departure_id, arrival_id, outbound_date)

        # drop obvious outliers (e.g. last-seat pricing) using a simple
        # median-multiple cutoff, then take the cheapest few remaining
        prices = [o.price for o in options]
        med = sorted(prices)[len(prices) // 2]
        normal_options = [o for o in options if o.price <= med * 2.5]
        sample = normal_options[:MAX_OPTIONS_PER_SEARCH]

        for opt in sample:
            departure_hour = int(opt.departure_time.split(" ")[1].split(":")[0])
            try:
                predicted = get_price_prediction(
                    departure_id, arrival_id, outbound_date,
                    airline=opt.airline, stops=opt.stops,
                    departure_hour=departure_hour,
                )
            except Exception as e:
                print(f"  skipped {opt.airline} {opt.flight_number}: {e}")
                continue

            ratio = opt.price / predicted
            ratios.append(ratio)
            print(f"  {opt.airline:12s} {opt.flight_number:8s} real=${opt.price:<7.2f} "
                  f"predicted=${predicted:<7.2f} ratio={ratio:.3f}")

    print(f"\n=== Calibration factor: mean of {len(ratios)} ratios ===")
    print(f"Mean:   {mean(ratios):.3f}")
    print(f"Min:    {min(ratios):.3f}")
    print(f"Max:    {max(ratios):.3f}")


if __name__ == "__main__":
    main()
