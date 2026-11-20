"""gta6-trophy-rarity-forecast-01's starter in Python 3.13 (standard library only): persistence.

The same program as starter/main.ts (the house baseline), number for number: it forecasts that every trophy's earn rate
on Exophase at the case's target is the rate in the case's last read, with a spread that widens like a random walk's on
the logit scale:

    q_tau = 100 / (1 + exp(-(logit(r) + z_tau * 0.5 * sqrt(d)))), rounded to 4 decimals (half up)

for each trophy, where r is its rate in the last read of the case's history (percent, clamped to 0.01..99.99 before the
logit), d the case's horizon in days (horizon_minutes / 1440) and z_tau the standard normal's quantile at each level
(0.05, 0.25, 0.5, 0.75, 0.95): -1.645, -0.674, 0, 0.674, 1.645.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in python:3.13-slim, standard library only: no
network, no pip packages. -P keeps this file's directory OFF sys.path, so a module you ship beside main.py (say
/bot/model.py) imports only after the sys.path line below, which puts the directory first; -s drops user
site-packages.

What an entry must do:
  - read every <id>.in in the input directory (the grader runs one case at a time, with only that case's file there):
    JSON {"schema": "makefun/gta6-trophies-case/v1", "id", "at", "horizon_minutes", "target", "levels", "game",
    "trophies", "history", "notes"}, where `trophies` lists the case's trophies {id, name, description, tier, points,
    secret} and `history` every read before `at`, oldest first, each {slot, at, players_tracked, trophies_earned,
    rates}, `rates` in the order of `trophies` (null where that read lacked the trophy);
  - write <id>.out to the output directory: {"trophies": [{"id": 1, "q": [q05, q25, q50, q75, q95]}, ...]}, one row
    for each of the case's trophies, each id once, no other key, each q five finite numbers from 0 to 100 (percent),
    non-decreasing; at most 65,536 bytes. Anything else is an invalid case, scored as if every quantile equalled the
    trophy's rate in the last read, and an entry with an invalid case takes no prize.

Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR stand in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model` finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]
# The spread of a rate's move, in logit units per square-root day.
SIGMA = 0.5


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    return math.log(clamp(pct) / (100 - clamp(pct)))


def nondecreasing(values: list) -> list:
    """Force a row's quantiles to be non-decreasing.

    The sigmoid is monotone and round4 is monotone, so a row cannot invert in
    practice; this is a guarantee the grader's "each at least the one before it"
    rule holds whatever rounding or floating point does, and it keeps the output
    deterministic (a pure function of the values in).
    """
    out = []
    prev = float("-inf")
    for v in values:
        v = max(v, prev)
        out.append(v)
        prev = v
    return out


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, as JavaScript's Math.round(x * 10000) / 10000 rounds it.

    A whole number comes back as an int, so json.dumps writes 100 where JavaScript writes 100 (not 100.0).
    """
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    last = case["history"][-1]
    days = case["horizon_minutes"] / 1440
    spread = SIGMA * math.sqrt(days)
    # Observed refresh proxy: growth of the tracked-players count across the history.
    players = [h["players_tracked"] for h in case["history"]]
    growth = (players[-1] / players[0]) if players and players[0] else 1.0
    # Hours-since-epoch of each read, on ln(1 + h) scale, for the trend fit.
    import calendar
    def tslot(h):
        t = h["at"]
        ts = calendar.timegm((int(t[:4]), int(t[5:7]), int(t[8:10]), int(t[11:13]), int(t[14:16]), int(t[17:19]), 0, 0, 0))
        return math.log1p(max(0.0, (ts - 1760832000) / 3600.0))
    xs_all = [tslot(h) for h in case["history"]]
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        pts = [(xs_all[k], case["history"][k]["rates"][i]) for k in range(len(case["history"]))
               if case["history"][k]["rates"][i] is not None]
        if not pts:
            rows.append({"id": trophy["id"], "q": [round4(0.01)] * 5})
            continue
        rate = pts[-1][1]
        lg = logit(rate)
        mu, s = lg, spread
        distinct = {v for _, v in pts}
        if len(pts) >= 3 and len(distinct) > 1:
            n = len(pts)
            mx = sum(x for x, _ in pts) / n
            my = sum(logit(v) for _, v in pts) / n
            sxx = sum((x - mx) ** 2 for x, _ in pts)
            if sxx > 1e-12:
                b = sum((x - mx) * (logit(v) - my) for x, v in pts) / sxx
                a = my - b * mx
                resid = [logit(v) - (a + b * x) for x, v in pts]
                rmse = math.sqrt(sum(e * e for e in resid) / n)
                # Damped extrapolation of the fit to the target instant: the
                # trend's prediction is blended toward the last read, and the
                # blend weight is capped well short of 1 so a straight-line fit
                # is never trusted in full, however many reads back it.
                xt = math.log1p(math.expm1(xs_all[-1]) + days * 24.0)
                lam = min(0.5, n / 12.0)
                mu = (1 - lam) * lg + lam * (a + b * xt)
                s = SIGMA * math.sqrt(days) + 0.35 * rmse * math.sqrt(max(0.1, days * 4.0)) + 0.05
                s *= min(1.5, max(1.0, growth ** 0.5))
        elif growth > 1.0:
            s += 0.3 * math.log(growth)
        s = max(s, 0.12)
        rows.append({"id": trophy["id"], "q": nondecreasing([round4(100 / (1 + math.exp(-(mu + z * s)))) for z in Z])})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
