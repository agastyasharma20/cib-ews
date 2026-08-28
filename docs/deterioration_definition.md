# What "Deterioration" Means in This System (and Why)

This doc explains, in plain language, how the labeling framework
(`src/labeling/`) decides whether a CIB customer is "silently deteriorating"
— the target the model in later phases will learn to predict. It's written
to be the answer to the interview question "how did you define your label,
and why not just threshold the balance?"

## The problem with an obvious approach

The naive label would be: "flag a customer if their balance drops by more
than X%." Three things break that immediately:

1. **Scale.** An SME's balance moving by ₹5 lakh is a big deal; the same
   move for a Large Corporate is rounding error. A flat percentage doesn't
   fix this either — a small SME can legitimately swing 20% month to month
   from ordinary working-capital cycles, while a stable Large Corporate
   rarely moves 5%.
2. **Shared seasonality.** Quarter-end window dressing, festive-season
   turnover spikes, advance-tax payment cycles — these move a lot of
   customers' balances in the same direction at the same time. A flat
   threshold can't tell "everyone's balance dipped this quarter" apart from
   "this customer's balance dipped and nobody else's did."
3. **One signal is fragile.** Balance alone is the last thing to move when a
   relationship is quietly eroding — money usually shifts in transaction
   flow and product usage well before the account balance visibly craters
   (that's the whole premise of an *early* warning system). A label built
   from balance alone would just rediscover what an RM already sees in a
   monthly balance report — no early warning at all.

## The approach: a composite, peer-relative Deterioration Index (DI)

Instead, every customer-month gets a **Deterioration Index (DI)**: a 0–1
score built from four behavioral signals, each expressed as *how much worse
this customer's trailing 3-month trend is, compared to same-segment peers
in the same month* — not compared to an absolute number.

### Step 1 — trailing-vs-prior % change, per signal

For each signal, compare the last 3 months' average to the 3 months before
that (a rough stand-in for "trailing 90 days" given the data is monthly —
see the note on this simplification below):

- **AMB decline** — average monthly balance
- **Transaction activity decline** — transaction count and value combined
- **Digital activity decline** — digital logins and mobile-banking share
- **Payroll / trade-finance decline** — payroll credits and trade-finance
  utilization (only for customers who actually hold these products — a
  customer with no trade-finance facility isn't penalized for a metric that
  doesn't apply to them)

Both windows only use data up to and including the month being scored —
never future data. That matters: the whole point is a signal usable in real
time, not one built with hindsight.

### Step 2 — turn each % change into a peer-relative percentile

A customer's raw "-8% transaction value" means nothing on its own. So each
signal's % change is ranked against every other customer **in the same
segment, in the same month**, and converted to a percentile: 1.0 means
"worse decline than every peer this month," 0.0 means "better than every
peer." This is what cancels out both scale differences (SME vs Large
Corporate) and shared seasonality (if the whole segment dips 8% that month,
nobody's percentile moves).

### Step 3 — blend the four percentiles into one DI

The four percentile scores are combined with documented weights
(`src/labeling/config.py`):

| Signal | Weight | Why |
|---|---|---|
| AMB decline | 0.35 | The actual business outcome, but laggy and seasonal |
| Transaction activity decline | 0.30 | Moves earlier than balance — the leading tell |
| Digital activity decline | 0.15 | Real signal, but noisy on its own |
| Payroll/trade-finance decline | 0.20 | Rare to move, but a strong signal when it does |

If a component doesn't apply to a customer (e.g. no trade-finance facility),
the weights are renormalized over whatever's left — the DI is never
penalized for a metric that structurally can't exist for that customer.

A customer "breaches" the index when DI exceeds a threshold — currently
**0.85**, chosen by grid search (see below), meaning "this customer's
blended decline signal is worse than 85% of their segment peers this
month."

## Handling seasonal false positives explicitly

An index breach alone isn't enough — a customer whose balance dipped for
tax payments or a dividend payout and bounced back a month later shouldn't
be flagged. The filter (`src/labeling/seasonal_filter.py`) downgrades a
breach from "confirmed deterioration" to "harmless dip" only when **both**:

1. **It reverts** — DI falls back below 0.50 within 2 months, and
2. **It's uncorroborated** — digital activity and payroll/trade activity
   were *not* also declining at the time.

A dip that reverts but was accompanied by real digital or payroll/trade
decline is kept — two independent signals moving together isn't a
coincidence, even if the balance itself recovers quickly.

## Turning DI into forward labels

`confirmed_deterioration` is a per-month, trailing-only flag. The actual
supervised-learning targets look forward from it:

```
deteriorates_in_30d(month t) = 1 if confirmed_deterioration is True
                                  in month t+1
deteriorates_in_60d(month t) = 1 if True in t+1 or t+2
deteriorates_in_90d(month t) = 1 if True in t+1, t+2, or t+3
```

Rows where `t + horizon` falls beyond a customer's last observed month are
left as `NaN` (not enough future history yet to know the right answer) and
must be dropped before training — never treated as negatives.

**Note on 30/60/90 days vs. months:** the synthetic panel (and, importantly,
most core-banking behavioral extracts) is monthly. "30/60/90 days" is
mapped to "1/2/3 months" throughout — an honest simplification. A
production system with daily/weekly transaction feeds could compute the
same DI on a rolling daily basis for tighter windows.

## Validation against known ground truth

Phase 1's synthetic generator planted four cohorts with a hidden true
cohort label (`ground_truth_cohort`) — never used as an input, only to
check afterward whether the framework built purely from behavior recovers
what was planted.

**Customer-level (was this customer ever confirmed-deteriorating?):**

| True cohort | Flag rate |
|---|---|
| sudden_deterioration | 100.0% |
| gradual_deterioration | 97.5% |
| seasonal_false_positive | 1.7% |
| stable | 0.1% |

Precision 98.9%, recall 98.3%, false-positive rate on the two healthy
cohorts combined: **0.5%**.

**Row-level, per horizon** (a stricter test: for each customer-month, does
the label correctly predict whether the customer is in a genuine,
already-onset deterioration episode within that exact forward window?):

| Horizon | Precision | Recall |
|---|---|---|
| 30d | 99.7% | 37.2% |
| 60d | 99.5% | 44.4% |
| 90d | 99.3% | 50.6% |

Row-level recall looks much lower than the customer-level number — this is
expected, not a flaw. The DI needs a few months of trailing evidence to
accumulate before it fires (see the lead-time table below), so it
under-fires in the *earliest* months of an episode even though it reliably
catches the episode eventually. Row-level precision stays high throughout —
when the label does fire, it's almost always right.

**Lead time** (how many months after the true, hidden onset month does the
index first confirm deterioration):

| Cohort | Detected | Median lag |
|---|---|---|
| sudden_deterioration | 100% | 2 months |
| gradual_deterioration | 97% | 4 months |

This lag is a direct consequence of using a trailing 3-month window (the
signal needs time to accumulate) — it is not a flaw to hide, but the
expected cost of a percentile-based, noise-robust design over a
same-day balance-threshold alarm. The business case still holds: an RM
looking only at monthly balance reports would typically notice a gradual
decline even later than this, once the balance itself has visibly dropped —
this framework still gets there first.

## Threshold selection (the actual trade-off table)

| DI threshold | Precision | Recall | FPR (stable+seasonal) |
|---|---|---|---|
| 0.60 | 0.399 | 1.000 | 0.672 |
| 0.70 | 0.662 | 1.000 | 0.228 |
| 0.75 | 0.827 | 1.000 | 0.093 |
| 0.80 | 0.945 | 0.998 | 0.026 |
| **0.85** | **0.989** | **0.983** | **0.005** |
| 0.90 | 1.000 | 0.865 | 0.000 |

0.80 already gave near-perfect recall, but its 5.2%-specific false-positive
rate on `seasonal_false_positive` alone (not shown in the combined FPR
above) was judged too noisy — an RM who gets flagged on a customer whose
balance recovers on its own within a month stops trusting the system fast.
0.85 cuts that to 1.7% for a 1.5-point recall cost, and 0.90 kills recall
too aggressively (13.5% of true deterioration missed) to buy an already-thin
remaining false-positive margin. 0.85 is the chosen operating point.
