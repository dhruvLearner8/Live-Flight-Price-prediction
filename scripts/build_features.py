"""
Build the model-ready feature table from flight_sample.csv.

Reads the raw sample (27 columns) and produces features.csv containing
only the 10 validated features + target, derived exactly the way they
were validated in notebooks/02_price_exploration.ipynb and
notebooks/03_route_layover_price.ipynb. See README.md for the full
rationale behind every kept/excluded column.

Categorical columns (startingAirport, destinationAirport, primaryAirline,
flightDayOfWeek, flightMonth) are left as plain strings/ints here, not
one-hot encoded. Encoding happens later, per model, at training time.

Also adds a "split" column (train/validation/test) based on a chronological
65/15/20 split of flightDate, so temporal order is respected: the model
never trains on a date that comes after a date it's validated/tested on.
flightDate itself is not included in the output, only the resulting label.
"""
import polars as pl

INPUT_PATH = "flight_sample.csv"
OUTPUT_PATH = "features.csv"

# carriers with fewer than this many rows get grouped into "Other" so they
# don't create noisy near-empty categories (see README: Cape Air, Boutique
# Air, Southern Airways Express, Key Lime Air all had under 130 rows)
MIN_AIRLINE_ROWS = 500

# chronological split boundaries, locked based on row-count percentiles of
# flightDate (not calendar-even months, since row counts vary a lot by month)
TRAIN_CUTOFF = "2022-09-03"       # <= this date: train (65%)
VALIDATION_CUTOFF = "2022-10-02"  # this date < x <= this date: validation (20%), after: test (15%)

df = pl.read_csv(INPUT_PATH)
print(f"Loaded {df.height:,} rows")

# --- date-derived features ---
df = df.with_columns([
    pl.col("searchDate").str.to_date(),
    pl.col("flightDate").str.to_date(),
])
df = df.with_columns(
    (pl.col("flightDate") - pl.col("searchDate")).dt.total_days().alias("daysUntilDeparture")
)
df = df.with_columns([
    pl.col("flightDate").dt.weekday().alias("flightDayOfWeek"),
    pl.col("flightDate").dt.month().alias("flightMonth"),
])

# --- chronological train/validation/test split label ---
df = df.with_columns(
    pl.when(pl.col("flightDate") <= pl.lit(TRAIN_CUTOFF).str.to_date())
    .then(pl.lit("train"))
    .when(pl.col("flightDate") <= pl.lit(VALIDATION_CUTOFF).str.to_date())
    .then(pl.lit("validation"))
    .otherwise(pl.lit("test"))
    .alias("split")
)

# --- departure hour, from the first leg only ---
df = df.with_columns(
    pl.col("segmentsDepartureTimeRaw").str.split("||").list.get(0).alias("_firstDepartureTimeRaw")
)
df = df.with_columns(
    pl.col("_firstDepartureTimeRaw").str.extract(r"T(\d{2}):", 1).cast(pl.Int64).alias("departureHour")
)

# --- number of stops, from segment count ---
df = df.with_columns(
    (pl.col("segmentsAirlineCode").str.split("||").list.len() - 1).alias("numStops")
)

# --- primary airline, from the first leg, rare carriers grouped into "Other" ---
df = df.with_columns(
    pl.col("segmentsAirlineName").str.split("||").list.get(0).alias("_primaryAirline")
)
airline_counts = df.group_by("_primaryAirline").agg(pl.len().alias("n"))
common_airlines = airline_counts.filter(pl.col("n") >= MIN_AIRLINE_ROWS)["_primaryAirline"].to_list()
df = df.with_columns(
    pl.when(pl.col("_primaryAirline").is_in(common_airlines))
    .then(pl.col("_primaryAirline"))
    .otherwise(pl.lit("Other"))
    .alias("primaryAirline")
)

# --- totalTravelDistance: impute missing values from segmentsDistance (sum of leg distances) ---
df = df.with_columns(
    pl.col("segmentsDistance")
    .str.split("||")
    .list.eval(pl.element().cast(pl.Float64, strict=False))
    .list.sum()
    .alias("_distanceFromSegments")
)
df = df.with_columns(
    pl.when(pl.col("totalTravelDistance").is_not_null())
    .then(pl.col("totalTravelDistance"))
    .otherwise(pl.col("_distanceFromSegments"))
    .alias("totalTravelDistanceFilled")
)
still_missing = df.filter(pl.col("totalTravelDistanceFilled").is_null()).height
print(f"totalTravelDistance still missing after imputation: {still_missing:,} rows")

# --- final feature selection ---
features = df.select([
    "startingAirport",
    "destinationAirport",
    "numStops",
    "primaryAirline",
    pl.col("totalTravelDistanceFilled").alias("totalTravelDistance"),
    "isBasicEconomy",
    "daysUntilDeparture",
    "flightDayOfWeek",
    "flightMonth",
    "departureHour",
    "totalFare",
    "split",
])

# drop any row still missing a value anywhere (should be a small number, from
# the rare case where segmentsDistance was also unparseable)
before = features.height
features = features.drop_nulls()
after = features.height
print(f"Dropped {before - after:,} rows with unresolvable nulls ({before:,} -> {after:,})")

features.write_csv(OUTPUT_PATH)
print(f"Wrote {OUTPUT_PATH}: {features.height:,} rows, {len(features.columns)} columns")
print(features.schema)
print()
print("Split sizes:")
print(features.group_by("split").agg(pl.len().alias("n")).sort("n", descending=True))
