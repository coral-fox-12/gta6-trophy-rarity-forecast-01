"""GTA VI trophy rarity forecast — make.fun bounty gta6-trophy-rarity-forecast-01.

Forecasts each trophy's Exophase earn rate at the case target
(2026-11-26T00:00:00Z for the final) as five quantiles, scored by mean
pinball loss on the logit scale.

The core observation: a rate is a ratio, r(t) = 100 * E(t) / N(t), where
E(t) is the number of players who have earned the trophy and N(t) the
number of players Exophase tracks. Both move, and the page only changes
when Exophase itself refreshes it (launch night: N sat at 638 and every
rate was flat for ~4.5 h, then one refresh took N to 3,868 and pushed
almost every rate down). Extrapolating r(t) directly therefore misses the
two moving parts: early on N grows faster than E, so rates FALL even while
players are earning trophies; later E catches up for achievable trophies
and rates RISE steeply, while grind trophies stay rare. A logit-linear
curve in log-time (the house curve baseline) has no saturation and
overshoots both ends.

So this program fits the two parts separately, in log-time
x = ln(hours since 2026-11-19T00:00Z + 1):

* N(x): a single saturating logistic, shared by all trophies.
* E_i(x): per trophy a saturating logistic in x (earners grow as players
  progress; the asymptote A_i is that trophy's eventual earners among the
  tracked population), with hierarchical shrinkage of the midpoint and
  the asymptote share toward tier / description-class priors, so trophies
  with only a handful of distinct reads (Exophase refreshes a few times a
  day) borrow strength from the other 54.

The median forecast is r = 100 * E_i(x_target) / N(x_target), blended
75/25 in logit space with a logit-linear-in-x extrapolation (a hedge
against the structural form being wrong), then shrunk toward the last
observed rate by n / (n + 6) where n is the number of distinct reads.
The spread is an honest logit sigma from the fit residuals, a profile
over the curve midpoint (flat histories leave it almost unidentified,
which widens exactly those trophies), the log-time extrapolation
distance, and Exophase refresh uncertainty.

Fallbacks keep every case valid: flat histories fall back to persistence,
all-zero earners to persistence with maximum spread, and any per-case
error to the starter's persistence rows rather than a crash.

Standard library only, no network, deterministic: the same input bytes
give the same output bytes. Runs in well under a second per case.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0)
in python:3.13-slim. Outside the grader's container, MAKEFUN_INPUT_DIR
and MAKEFUN_OUTPUT_DIR stand in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
from datetime import datetime
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Release midnight: log-time is measured from here. Reads start just
# before it (18/11 23:50Z), so x can be slightly negative; ln(t + 1)
# stays defined.
RELEASE = "2026-11-19T00:00:00Z"
# Blend weight of the structural (E/N) forecast vs the logit-linear hedge.
W_STRUCT = 0.75
# Spread pieces, in logit units.
SIGMA_REFRESH = 0.25          # the answer is the page state at the target,
                              # i.e. after its last refresh before then
SIGMA_EXTRAP_PER_X = 0.25     # per unit of log-time beyond the last read
SIGMA_MIN = 0.35
SIGMA_MAX = 2.2
# Shrinkage strength: a trophy's own fit gets weight n / (n + SHRINK_K).
SHRINK_K = 4.0
# Persistence shrink: the extrapolated move away from the last rate is
# shrunk by n_distinct / (n_distinct + PERSIST_K). With only a handful of
# distinct reads the structural form is weakly identified, so a model that
# fits the wrong shape must not run far from persistence; as the history
# fills in, the model takes over.
PERSIST_K = 6.0


def parse_time(s):
    """ISO 8601 UTC timestamp -> epoch seconds."""
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s).timestamp()


RELEASE_TS = parse_time(RELEASE)


def x_of(ts):
    """Log-time coordinate: ln(hours since release + 1)."""
    hours = (ts - RELEASE_TS) / 3600.0
    return math.log(max(0.0, hours) + 1.0)


def clamp_pct(v):
    return min(99.99, max(0.01, v))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def round4(x):
    """x rounded to 4 decimals, half up, as JavaScript's
    Math.round(x * 10000) / 10000."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def nelder_mead(f, x0, step, iters=400, tol=1e-10):
    """Deterministic Nelder-Mead on a list of floats. Returns (best_x, best_f)."""
    n = len(x0)
    pts = [list(x0)]
    for i in range(n):
        p = list(x0)
        p[i] += step[i]
        pts.append(p)
    vals = [f(p) for p in pts]
    for _ in range(iters):
        order = sorted(range(n + 1), key=lambda i: vals[i])
        pts = [pts[i] for i in order]
        vals = [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) < tol * (abs(vals[0]) + tol):
            break
        cen = [sum(pts[i][j] for i in range(n)) / n for j in range(n)]
        xr = [cen[j] + (cen[j] - pts[-1][j]) for j in range(n)]
        fr = f(xr)
        if fr < vals[0]:
            xe = [cen[j] + 2.0 * (cen[j] - pts[-1][j]) for j in range(n)]
            fe = f(xe)
            if fe < fr:
                pts[-1], vals[-1] = xe, fe
            else:
                pts[-1], vals[-1] = xr, fr
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = [cen[j] + 0.5 * (pts[-1][j] - cen[j]) for j in range(n)]
            fc = f(xc)
            if fc < vals[-1]:
                pts[-1], vals[-1] = xc, fc
            else:
                for i in range(1, n + 1):
                    pts[i] = [pts[0][j] + 0.5 * (pts[i][j] - pts[0][j])
                              for j in range(n)]
                    vals[i] = f(pts[i])
    best = min(range(n + 1), key=lambda i: vals[i])
    return pts[best], vals[best]


def logistic_fit(xs, ys, y_floor, y_cap, m0, s0=1.0):
    """Fit y = A * sigmoid((x - m) / s) by least squares on y.

    Returns (A, m, s, rmse) or None when the data cannot support a fit.
    A is bounded to [y_floor, y_cap]; s to [0.15, 3.0]; m to the data's x
    range widened by 3. A penalty keeps the optimum inside the bounds.
    """
    if len(xs) < 3:
        return None
    x_lo, x_hi = min(xs), max(xs)
    y_max = max(ys)

    def unpack(p):
        return math.exp(p[0]), p[1], math.exp(p[2])

    def loss(p):
        a, m, s = unpack(p)
        pen = 0.0
        if a < y_floor or a > y_cap:
            pen += 1000.0 * (min(abs(a - y_floor), abs(a - y_cap)) + 1.0) ** 2
        if s < 0.15 or s > 3.0:
            pen += 1000.0
        if m < x_lo - 3.0 or m > x_hi + 3.0:
            pen += 1000.0
        a_c = min(max(a, y_floor), y_cap)
        s_c = min(max(s, 0.15), 3.0)
        m_c = min(max(m, x_lo - 3.0), x_hi + 3.0)
        se = 0.0
        for x, y in zip(xs, ys):
            se += (a_c * sigmoid((x - m_c) / s_c) - y) ** 2
        return se / len(xs) + pen

    a0 = min(max(y_max * 1.5, y_floor * 1.1), y_cap)
    best, _ = nelder_mead(loss, [math.log(a0), m0, math.log(s0)],
                          [0.3, 0.5, 0.3])
    a, m, s = unpack(best)
    a = min(max(a, y_floor), y_cap)
    s = min(max(s, 0.15), 3.0)
    m = min(max(m, x_lo - 3.0), x_hi + 3.0)
    rmse = math.sqrt(sum((a * sigmoid((x - m) / s) - y) ** 2
                         for x, y in zip(xs, ys)) / len(xs))
    return (a, m, s, rmse)


def linfit(xs, ys):
    """Least squares y ~ a + b*x. Returns (a, b) or None."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return (my, 0.0)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return (my - b * mx, b)


def trophy_class(trophy):
    """story / grind / other, from the trophy's description (domain prior)."""
    d = (trophy.get("description") or "").lower()
    if "chapter" in d or "final mission" in d:
        return "story"
    grind_words = ("250", "100%", "100 locations", "50 challenges",
                   "collectible", "every spot", "discover 100", "win 20",
                   "all shooting", "every location", "100 vehicles",
                   "gold medal in all")
    if any(w in d for w in grind_words):
        return "grind"
    return "other"


# Prior log-time midpoints (median earner-growth time) by class, and prior
# eventual-earner shares of the tracked base by tier. Weak priors: they
# only regularise trophies whose own fit is flat or degenerate.
PRIOR_M = {"story": 3.2, "other": 4.3, "grind": 5.7, "platinum": 6.5}
PRIOR_SHARE = {"platinum": 0.02, "gold": 0.10, "silver": 0.25, "bronze": 0.50}


def distinct_reads(history):
    """Collapse consecutive identical reads (same N, same rates) to one
    observation each: Exophase's page is piecewise constant between its
    own refreshes, so duplicates carry no curve information."""
    out = []
    prev = None
    for r in history:
        key = (r.get("players_tracked"), tuple(r.get("rates") or ()))
        if key != prev:
            out.append(r)
            prev = key
    return out


def persistence_rows(case, last_rates, horizon_days):
    """The starter's persistence forecast (house baseline), as a fallback."""
    spread = 0.5 * math.sqrt(max(0.0, horizon_days))
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = last_rates[i]
        lg = logit(rate if rate is not None else 0.01)
        rows.append({"id": trophy["id"],
                     "q": [round4(min(max(inv_logit(lg + z * spread), 0.01), 99.99))
                           for z in Z]})
    return rows


def _earner_points(fit_reads, i):
    """(x, earners) points for trophy i, from rates * N / 100."""
    pts = []
    for x, n, rates in fit_reads:
        if i < len(rates) and rates[i] is not None:
            pts.append((x, rates[i] * n / 100.0))
    return pts


def _midpoint_profile(pts_e, m, s, a, n_inf):
    """Parameter uncertainty in the midpoint m: refit A in closed form at
    each m and take the forecast's logit range over the m values that fit
    nearly as well as the best. Flat histories leave m almost
    unidentified, so this widens exactly the trophies whose curve the
    data cannot pin down. Returns a logit half-width."""
    exs = [p[0] for p in pts_e]
    eys = [p[1] for p in pts_e]

    def sse_and_a(mm):
        fs = [sigmoid((x - mm) / s) for x in exs]
        den = sum(f * f for f in fs)
        aa = sum(y * f for y, f in zip(eys, fs)) / den if den > 0 else a
        aa = min(max(aa, max(eys)), n_inf)
        return sum((aa * f - y) ** 2 for f, y in zip(fs, eys)), aa

    best_sse, _ = sse_and_a(m)
    thresh = best_sse + max(best_sse, 2.0 * len(pts_e))
    lo_m, hi_m = m, m
    for k in range(1, 15):
        mm = m + 0.5 * k
        sse, _ = sse_and_a(mm)
        if sse <= thresh:
            hi_m = mm
        else:
            break
    for k in range(1, 15):
        mm = m - 0.5 * k
        sse, _ = sse_and_a(mm)
        if sse <= thresh:
            lo_m = mm
        else:
            break
    return lo_m, hi_m, sse_and_a


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    trophies = case["trophies"]
    history = case.get("history") or []
    if not history:
        raise ValueError(f"{name} has no history")
    horizon_days = case["horizon_minutes"] / 1440.0
    try:
        target_ts = parse_time(case["target"])
    except Exception:
        target_ts = parse_time(case["at"]) + case["horizon_minutes"] * 60.0
    x_target = x_of(target_ts)

    # Last read's rates (persistence anchor / fallback input).
    last = history[-1]
    last_rates = list(last.get("rates") or [])
    if len(last_rates) < len(trophies):
        last_rates += [None] * (len(trophies) - len(last_rates))

    reads = distinct_reads(history)
    # Reads usable for fitting: need a timestamp, N and rates.
    fit_reads = []
    for r in reads:
        try:
            ts = parse_time(r.get("at") or r.get("slot"))
        except Exception:
            continue
        n = r.get("players_tracked")
        rates = r.get("rates") or []
        if n is None or not rates:
            continue
        fit_reads.append((x_of(ts), float(n),
                          [float(v) if v is not None else None for v in rates]))
    if len(fit_reads) < 3:
        return {"trophies": persistence_rows(case, last_rates, horizon_days)}

    xs = [r[0] for r in fit_reads]
    ns = [r[1] for r in fit_reads]
    x_last = max(xs)
    n_last = ns[-1]
    x_lo, x_hi = min(xs), max(xs)

    # --- Fit the tracked base N(x): one saturating logistic in log-time. ---
    n_fit = logistic_fit(xs, ns, max(1.0, n_last),
                         50.0 * max(1.0, n_last), (x_lo + x_hi) / 2.0)
    if n_fit is not None:
        n_inf, n_m, n_s, _ = n_fit
        n_target = n_inf * sigmoid((x_target - n_m) / n_s)
    else:
        n_inf = max(ns) * 3.0
        n_target = n_last
    n_target = max(n_target, n_last)

    # --- Per-trophy earner fits. ---
    own = []  # (A, m, s, rmse) per trophy, ("full",) for the 100% trophy, None
    for i in range(len(trophies)):
        pts = _earner_points(fit_reads, i)
        if len(pts) < 3:
            own.append(None)
            continue
        ex = [p[0] for p in pts]
        ey = [p[1] for p in pts]
        if max(ey) <= 0:
            own.append(None)
            continue
        if min(ey) >= ns[0] * 0.999 and max(ey) >= max(ns) * 0.999:
            # Earners == tracked in every read (the 100% trophy): no curve.
            own.append(("full",))
            continue
        cls = trophy_class(trophies[i])
        tier = str(trophies[i].get("tier") or "bronze").lower()
        m_prior = PRIOR_M.get("platinum" if tier == "platinum" else cls, 4.3)
        own.append(logistic_fit(ex, ey, max(ey), n_inf, m_prior))

    # Tier/class priors from the pooled own-fits (shrunk toward the global
    # mean when a cell is small), for trophies whose own fit is missing.
    fitted_ms = [o[1] for o in own if o is not None and o != ("full",)]
    global_m = sum(fitted_ms) / len(fitted_ms) if fitted_ms else 4.3
    cell_ms = {}
    for i, o in enumerate(own):
        if o is None or o == ("full",):
            continue
        tier = str(trophies[i].get("tier") or "bronze").lower()
        key = "platinum" if tier == "platinum" else trophy_class(trophies[i])
        cell_ms.setdefault(key, []).append(o[1])
    cell_prior = {}
    for key, vals in cell_ms.items():
        w = len(vals) / (len(vals) + 3.0)
        cell_prior[key] = w * (sum(vals) / len(vals)) + (1.0 - w) * global_m

    rows = []
    for i, trophy in enumerate(trophies):
        rate_last = last_rates[i]
        o = own[i]
        # --- The 100% trophy: it stays (essentially) 100%. ---
        if o == ("full",) or (rate_last is not None and rate_last >= 99.99):
            med = 99.99
            sigma = 0.05
            rows.append({"id": trophy["id"],
                         "q": [round4(min(max(inv_logit(logit(med) + z * sigma),
                                              0.01), 99.99))
                               for z in Z]})
            continue
        if rate_last is None:
            rate_last = 0.01
        pts_e = _earner_points(fit_reads, i)
        # --- Structural median: E_i(x_target) / N(x_target). ---
        med_struct = None
        resid_logit = None
        param_logit = 0.0
        if o is not None:
            a, m, s, _rmse = o
            tier = str(trophy.get("tier") or "bronze").lower()
            key = "platinum" if tier == "platinum" else trophy_class(trophy)
            m_prior = cell_prior.get(key, PRIOR_M.get(key, global_m))
            n_obs = len(pts_e)
            w = n_obs / (n_obs + SHRINK_K)
            m_shrunk = w * m + (1.0 - w) * m_prior
            # Shrink the asymptote share toward the tier prior share.
            share_prior = PRIOR_SHARE.get(tier, 0.5)
            a_share = a / n_inf if n_inf > 0 else 0.0
            a_shrunk = n_inf * math.exp(
                w * math.log(max(a_share, 1e-6))
                + (1.0 - w) * math.log(share_prior))
            a_shrunk = min(max(a_shrunk, max(1.0, a * 0.5)), n_inf)
            e_target = a_shrunk * sigmoid((x_target - m_shrunk) / s)
            med_struct = 100.0 * e_target / n_target
            # Residuals of the structural fit, on the logit scale.
            res = []
            for x, n, rates in fit_reads:
                if i >= len(rates) or rates[i] is None:
                    continue
                r_fit = 100.0 * (a_shrunk * sigmoid((x - m_shrunk) / s)) / max(n, 1.0)
                res.append(logit(rates[i]) - logit(r_fit))
            if len(res) >= 3:
                mean_res = sum(res) / len(res)
                resid_logit = math.sqrt(
                    sum((v - mean_res) ** 2 for v in res) / (len(res) - 1))
            # Parameter uncertainty in the midpoint m (profile, see helper).
            if len(pts_e) >= 3:
                lo_m, hi_m, sse_and_a = _midpoint_profile(pts_e, m, s, a, n_inf)
                if hi_m > lo_m:
                    _, a_lo = sse_and_a(lo_m)
                    _, a_hi = sse_and_a(hi_m)
                    r_lo = 100.0 * a_lo * sigmoid((x_target - lo_m) / s) / n_target
                    r_hi = 100.0 * a_hi * sigmoid((x_target - hi_m) / s) / n_target
                    param_logit = 0.5 * abs(logit(r_lo) - logit(r_hi))
        # --- Logit-linear hedge: straight line of logit(rate) on x. ---
        med_direct = None
        pts = [(x, logit(rates[i])) for x, n, rates in fit_reads
               if i < len(rates) and rates[i] is not None]
        if len(pts) >= 2:
            lf = linfit([p[0] for p in pts], [p[1] for p in pts])
            if lf is not None:
                a0, b0 = lf
                med_direct = inv_logit(a0 + b0 * x_target)
        # --- Blend in logit space. ---
        if med_struct is not None and med_direct is not None:
            med_logit = (W_STRUCT * logit(med_struct)
                         + (1.0 - W_STRUCT) * logit(med_direct))
        elif med_struct is not None:
            med_logit = logit(med_struct)
        elif med_direct is not None:
            med_logit = logit(med_direct)
        else:
            med_logit = logit(rate_last)
        # --- Persistence shrink: do not run far from the last rate on a
        # thin history. The shrink factor grows with the number of
        # distinct reads (i.e. with the number of Exophase refreshes
        # actually seen).
        n_obs = len(pts_e)
        w_model = n_obs / (n_obs + PERSIST_K)
        med_logit = logit(rate_last) + w_model * (med_logit - logit(rate_last))
        # --- Spread: residuals + midpoint uncertainty + extrapolation
        # distance + refresh noise. No structural fit at all (zero
        # earners, too few reads) means maximum honest spread.
        r_var = (resid_logit or 0.30) ** 2
        e_var = (SIGMA_EXTRAP_PER_X * max(0.0, x_target - x_last)) ** 2
        sigma = math.sqrt(r_var + param_logit ** 2 + e_var + SIGMA_REFRESH ** 2)
        if o is None:
            sigma = SIGMA_MAX
        sigma = min(max(sigma, SIGMA_MIN), SIGMA_MAX)
        qs = [round4(inv_logit(med_logit + z * sigma)) for z in Z]
        qs = sorted(min(max(q, 0.01), 99.99) for q in qs)
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case, path.name)
        except Exception:
            # Never crash a case: fall back to persistence rows.
            last = (case.get("history") or [{}])[-1]
            last_rates = list(last.get("rates") or [0.01] * len(case["trophies"]))
            forecast = {"trophies": persistence_rows(
                case, last_rates, case.get("horizon_minutes", 90) / 1440.0)}
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
