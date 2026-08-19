"""
Lazily scan itineraries.csv to find route volume (row count per
startingAirport -> destinationAirport pair), without loading the
full 31GB file into memory.

Only reads the two airport columns via projection pushdown, and uses
the streaming engine so peak memory stays low regardless of file size.
"""
import polars as pl

DATA_PATH = "itineraries.csv"

lf = pl.scan_csv(DATA_PATH)

route_counts = (
    lf.select(["startingAirport", "destinationAirport"])
    .group_by(["startingAirport", "destinationAirport"])
    .len()
    .sort("len", descending=True)
)

result = route_counts.collect(engine="streaming")
print(result)

print("\nTotal distinct routes:", result.height)
print("Total rows in dataset:", result["len"].sum())

top_airports = (
    result.group_by("startingAirport")
    .agg(pl.col("len").sum())
    .sort("len", descending=True)
)
print("\nRow volume by origin airport:")
print(top_airports)
