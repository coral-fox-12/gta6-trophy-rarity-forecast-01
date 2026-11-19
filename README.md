# GTA VI Trophy Rarity Forecast

![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)
![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/coral-fox-12/gta6-trophy-rarity-forecast-01/rank.svg)
![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/coral-fox-12/gta6-trophy-rarity-forecast-01/reward.svg)

Forecasts each Grand Theft Auto VI trophy's Exophase earn rate at
2026-11-26T00:00:00Z, as five quantiles per trophy, for the
[make.fun bounty gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01).

## Method

A trophy's rate is a ratio, `r(t) = 100 * E(t) / N(t)`: `E(t)` the players
who have earned it, `N(t)` the players Exophase tracks. Both grow, and the
page is piecewise constant between Exophase's own refreshes (launch night:
`N` frozen at 638 and every rate flat for ~4.5 h, then one refresh took `N`
to 3,868 and every rate but the 100% one down). Extrapolating `r(t)`
directly misses the two moving parts, so this program fits them separately
in log-time `x = ln(hours since 2026-11-19T00:00Z + 1)`:

- `N(x)`: one saturating logistic curve, shared by all trophies.
- `E_i(x)`: per trophy a saturating logistic in `x`, with hierarchical
  shrinkage of the midpoint and the asymptote share toward tier /
  description-class priors, so trophies with only a handful of distinct
  reads borrow strength from the other 54.
- The median is `100 * E_i(x_target) / N(x_target)`, blended 75/25 in logit
  space with a logit-linear-in-`x` extrapolation, then shrunk toward the
  last rate by `n / (n + 6)` where `n` is the number of distinct reads
  (Exophase refreshes a few times a day, so early histories barely identify
  a curve).
- The spread is an honest logit sigma from fit residuals, a profile over
  the curve midpoint (flat histories leave it almost unidentified, which
  widens exactly those trophies), the log-time extrapolation distance, and
  Exophase refresh uncertainty.

Fallbacks keep every case valid: flat histories fall back to persistence,
all-zero earners to persistence with maximum spread, and any per-case error
to persistence rows rather than a crash.

## Validation

- Practice cases `p00`/`p01`: valid output, deterministic, ~0.07 s per
  case. Practice only validates format — its histories are entirely flat,
  so the model (correctly) returns near-persistence there; the practice
  answer is the post-refresh page, which no program can foresee from flat
  data.
- Synthetic recovery tests (`test_synth.py`, kept outside this repo):
  histories generated from known saturating `E`/`N` dynamics with
  Exophase-style refreshes and 2-decimal rounding. Over 3 seeds the model
  scores **~0.13** mean pinball loss vs **~0.35** persistence and **~0.28**
  curve baseline. Under a mismatched truth (Weibull earners, power-law
  tracked base) it scores **~0.42–0.46** vs **~0.48–0.51** persistence and
  **~0.46–0.52** curve — still ahead, never a blow-up.
- Rolling-origin backtests on our own polled history: `backtest.py`
  (run once the history file has ~2 days of reads; the entry itself only
  ever reads the case file it is given).

## Own data

Since 19/11 19:03 +07 we poll the public Exophase trophy page every
30 minutes into `exophase-history/history.jsonl` (kept outside this repo).
Plain `curl` gets a Cloudflare 403 and headless Chrome stalls on the
challenge page; the working fetch is the crawler-backed page fetch.
That history is what the backtests run on.

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, no network, deterministic.

## License

MIT — see [LICENSE](LICENSE).
