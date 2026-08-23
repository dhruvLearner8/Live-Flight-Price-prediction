# Challenges, Solutions, and What I Learned

This project (a flight price prediction model + a LangGraph buy/wait agent built
on top of it) had several points where the obvious first approach was wrong, and
figuring out why taught me more than the parts that went smoothly. This document
is the honest version of that process — including the mistakes.

## 1. A pooled statistical trend that was actually backwards (Simpson's paradox)

**What happened:** early in EDA, I found that more layovers meant higher prices —
nonstop averaged $327, one stop $375, two stops $434, three stops $520, pooling
all 20 routes together. It looked like a clean, usable signal.

**The problem:** I sanity-checked it against a real personal experience (a trip
where a connecting flight was cheaper than nonstop) and it didn't match. Instead
of dismissing the anecdote, I broke the aggregate down **per route** rather than
trusting the pooled number. The story reversed for about half the routes — on
LAX-BOS specifically, a 1-stop flight was **$76 cheaper** than nonstop, the
opposite of the pooled trend.

**What was actually happening:** route and stop-count were confounded. Some
routes are simply pricier overall *and* happen to have a different mix of
nonstop/connecting inventory. Pooling blended those two separate facts into one
misleading number — a textbook case of Simpson's paradox.

**What I learned:** an aggregate trend is not evidence on its own — it's a
hypothesis that needs to survive being broken down by the most likely
confounding variable before I trust it. This changed how I did every subsequent
analysis in the project: check the pooled number, then immediately ask "pooled
across what, and would it survive being split apart?"

## 2. A feature that looked useless until I stopped trusting the wrong tool

**What happened:** departure hour showed a correlation with price of -0.008 —
essentially zero. I initially reported it as a null result.

**The problem:** I'd only looked at linear correlation and a coarse 4-bucket
grouping (quarters of the day). When I went back and printed the raw
hour-by-hour numbers, there was a real $142 spread — midnight and 10-11pm
departures were notably pricier than the rest of the day. My bucketing had
averaged a real spike together with its cheaper neighboring hours, erasing it.

**What I learned:** correlation and coarse binning only detect *monotonic*,
*smooth* relationships. Several of the most useful signals in this project
(departure hour, days-until-departure) were real but non-linear, and would have
been thrown away by trusting the first summary statistic instead of looking at
the raw shape of the data.

## 3. Overfitting I initially claimed without evidence

**What happened:** I doubled XGBoost's tree depth and count (6→8, 300→500
trees), validation performance got *worse*, and I confidently said "that's
overfitting."

**The problem:** a colleague-style check-in ("how do you know?") caught that I
hadn't actually verified this — I'd only compared validation scores, never
looked at training performance for either model. I went back and computed
train-vs-validation R² for both: the smaller model had a 0.19 gap (train 0.59,
val 0.40), the bigger one had a 0.32 gap (train 0.68, val 0.36). *That* is
the actual, checkable definition of overfitting — and only then was the
diagnosis actually earned rather than assumed.

**What I learned:** "that's probably overfitting" is a hypothesis, not a
finding, until there's a train-vs-validation comparison to back it up. I now
treat any performance-regression claim as something to verify before stating
it, not something to infer from general ML knowledge alone.

## 4. A SHAP result so implausible it had to be a bug, not a finding

**What happened:** early SHAP analysis on the final XGBoost model showed one
airline category with a mean absolute SHAP value of **$541** — larger than the
dataset's average fare ($364). That's not a real effect; no single feature
should out-weigh the entire average price.

**The cause:** the model had been trained on data where the one-hot encoder
output a sparse matrix, but I explained it using a densified array. XGBoost
treats implicit sparse zeros as "missing" during training, but literal zeros
in a dense array are treated as a real value — a subtle mismatch between how
the model was trained and how it was being explained. Fixed by keeping the
encoder dense throughout, both training and explanation, which reproduced
sane, EDA-consistent results.

**What I learned:** implausible numbers are information, not just noise to
average away. A finding that contradicts basic sanity checks (here: a single
feature can't outweigh the mean of the target) is usually a pipeline bug, and
tracing it down taught me a real, specific gotcha in how XGBoost + sparse
encoders + SHAP interact — something I wouldn't have learned by getting a
"reasonable-looking" wrong answer instead of an absurd one.

## 5. A live-calibration design that looked reasonable, and wasn't

**What happened:** the model is trained on 2022 data; it's 2026 by deployment.
The plan was to compute a "calibration ratio" (real SerpAPI price ÷ model
prediction) from a few live lookups, then apply it to correct future
predictions before comparing to a live price.

**Measuring it exposed two problems.** First, the ratio was wildly inconsistent
— 0.34 to 1.73 across just three routes, varying by route, by how far out the
booking was, and by airline. There was no single defensible number. Second, and
more fundamentally: if computing the calibration factor requires live API calls
anyway, the "corrected" model prediction adds nothing that isn't already sitting
in the live data used to compute the correction — it's circular.

**What I did instead:** dropped the calibration bridge entirely. The agent now
shows the model's prediction (a historical/structural estimate) and the live
price (the real current number) side by side, and lets the reasoning step weigh
both rather than mechanically merging them into one artificial figure.

**What I learned:** an approach can be internally consistent and still be
solving the wrong problem. The instinct to sanity-check "does this design
survive being used the way it's actually meant to be used" — not just "does the
math work" — came directly from a direct, blunt challenge to defend the design,
and it changed the whole agent architecture for the better.

## 6. A "predict the future" feature that was never possible in the first place

**What happened:** the original plan included `get_price_trend(days=7)` —
forecast whether waiting N more days would lower the price for one specific
flight.

**Testing it directly:** predicted prices across a 7-day booking window moved
by less than $15 total ($470-484), and were identical on 3 consecutive days.
`daysUntilDeparture` was simply too weak a signal to support day-by-day
resolution — asking for it was asking the model to do something the data
didn't support.

**What replaced it:** comparing the model's prediction (and a live price)
across 7 *different upcoming flight dates* — a question that's actually
answerable today, versus trying to forecast a future price that doesn't exist
yet.

**What I learned:** it's worth explicitly testing a planned feature's actual
resolution before building a UI or agent logic around it. A feature can sound
reasonable in a design doc and still fail the moment you print out what it
actually predicts.

## 7. Cost discipline: choosing not to repeat a past mistake

**What happened:** a previous unrelated project had run up an unexpected ~$1500
AWS SageMaker bill from a real-time endpoint left running and billing hourly
regardless of traffic.

**What I did differently here:** deployed the trained model on Render instead
— a platform with a genuine, hard-capped free tier — specifically to avoid
that failure mode for a portfolio project with low, sporadic traffic. Where
"real predictive infrastructure" and "safe experimentation budget" were in
tension, I picked the option that couldn't silently cost money while I wasn't
looking.

## 8. A capability I built but forgot to actually connect

**What happened:** asked to restrict live flight search results to three
specific airlines, I added an `include_airlines` parameter to the underlying
SerpAPI tool function — tested it in isolation, confirmed it worked, and moved
on. The excluded airline kept showing up in the actual running app anyway.

**The bug:** the parameter existed on the low-level function, but the
higher-level tool the agent and UI actually call never passed it through. I'd
verified the *piece* worked without verifying it was *wired into* the thing
being tested.

**What I learned:** testing a function in isolation is not the same as testing
the path a user actually exercises. Now I trace a fix all the way to the
interface being used before calling it done, not just to the function where the
capability lives.

## Why this is not a finished, perfect system

Being honest about the ceiling here matters more than overselling it:

- **The model explains about 40% of price variance (R²), at best.** The rest
  is driven by things that simply aren't in this dataset at all — real-time
  seat inventory, competitor pricing, algorithmic demand-based pricing. No
  amount of feature engineering on the columns available fixes that; it would
  need genuinely different data sources.
- **The training data covers a single ~7-month window in 2022.** I can't
  separate real recurring seasonality from a one-time trend over that specific
  period, and there's no winter holiday data at all.
- **Absolute price levels are stale by construction.** The model was never
  recalibrated against 2026 prices, on purpose — the alternative (live
  calibration) turned out to be circular and unreliable, as covered above. The
  model's dollar predictions should be read as directional/historical, not as
  literal current prices.
- **Test-set performance is honestly weaker than validation** (R² 0.40 →
  0.13 for the winning model), because the chronological split deliberately
  tests on the hardest, least-seen part of the year rather than a random,
  easier split. This was a deliberate choice for rigor, not an oversight — but
  it does mean the headline numbers look less impressive than a random split
  would have produced.
- **The agent only supports 10 airports / 20 routes by design** (the ones with
  enough training data to trust), even though it's now deployed publicly via
  the Streamlit UI.

## What this project actually taught me about being an ML practitioner

The technical skills (Polars for out-of-core data processing, XGBoost/LightGBM,
SHAP, MLflow, LangGraph) were the smaller part of it. The bigger shift was
learning to treat every "this looks like it's working" moment as a claim to be
tested, not a result to report — checking pooled trends against subgroups,
checking overfitting claims against actual train/validation gaps, checking
implausible numbers for bugs before treating them as findings, and checking a
clever design against whether it actually needs the thing it claims to fix. The
project's real value, going into interviews, isn't "I built a model that
predicts flight prices" — it's the list above of places where the first answer
was wrong and I had a concrete way of finding out.
