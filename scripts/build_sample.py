"""
Build flight_sample.csv from the full 31GB itineraries.csv.

Approach:
1. Lazily scan the full CSV (no data loaded yet, just a query plan).
2. Filter to rows on the 10 major city-pairs (20 directed routes),
   selected for having the highest combined bidirectional row volume
   in the dataset -- this guarantees enough per-route history for
   route-level seasonality/trend features later.
3. Collect with the streaming engine, which processes the file in
   chunks so peak memory stays well under the 16GB budget even
   though the source file is 31GB.
4. Randomly downsample the filtered ~12.25M rows to ~1.5M rows with
   a fixed seed, so route-level proportions are preserved and the
   result is reproducible.
"""
import polars as pl

DATA_PATH = "itineraries.csv"
OUTPUT_PATH = "flight_sample.csv"
SAMPLE_SIZE = 1_500_000
SEED = 42

# 10 city-pairs = 20 directed routes, chosen by combined bidirectional
# row volume (see scripts/inspect_routes.py for the analysis).
ROUTE_PAIRS = [
    ("ATL", "LAX"), ("LAX", "LGA"), ("BOS", "LAX"), ("JFK", "LAX"),
    ("DFW", "LAX"), ("LAX", "ORD"), ("DTW", "LAX"), ("EWR", "LAX"),
    ("CLT", "LAX"), ("LGA", "ORD"),
]

# Build the set of all 20 directed (start, dest) combinations.
directed_routes = set()
for a, b in ROUTE_PAIRS:
    directed_routes.add((a, b))
    directed_routes.add((b, a))

route_filter = pl.lit(False)
for start, dest in directed_routes:
    route_filter = route_filter | (
        (pl.col("startingAirport") == start) & (pl.col("destinationAirport") == dest)
    )

lf = pl.scan_csv(DATA_PATH).filter(route_filter)

print("Filtering and collecting matching rows (streaming)...")
filtered = lf.collect(engine="streaming")
print(f"Filtered rows: {filtered.height:,}")

sample = filtered.sample(n=SAMPLE_SIZE, seed=SEED, shuffle=True)
print(f"Sampled rows: {sample.height:,}")

sample.write_csv(OUTPUT_PATH)
print(f"Wrote {OUTPUT_PATH}")

print("\nRoute distribution in sample:")
print(
    sample.group_by(["startingAirport", "destinationAirport"])
    .len()
    .sort("len", descending=True)
)
