
from __future__ import annotations

import random
from typing import Any


def adversarial_xs(family: str, lo: float, hi: float, n: int = 16, seed: int = 0) -> list[float]:
    rnd = random.Random(seed)
    xs = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
    if family in ("rational", "rational11"):
        # focus near suspected poles around mid
        mid = 0.5 * (lo + hi)
        xs = sorted(xs + [mid - 1e-3, mid + 1e-3, mid - 5e-4, mid + 5e-4])
    if family in ("sin", "cos", "harm"):
        # hit flat spots (near pi/2 + k*pi for cos, etc.)
        xs += [(rnd.random() * (hi - lo) + lo) for _ in range(8)]
    return xs[: max(n, len(xs))]


def probe_family(family: str, xs: list[float], ys: list[float]) -> dict[str, Any]:
    """Fit then search for large local errors on a dense grid in [min(xs), max(xs)]."""
    try:
        from transfer.families import fit_families, predict
    except Exception:
        return {"ok": False, "error": "missing_families"}
    fits = fit_families(xs, ys, [family], [], robust=False, delta=1.0)
    if not fits:
        return {"ok": False, "error": "fit_fail"}
    p = fits[0].params
    lo = min(xs)
    hi = max(xs)
    grid = [lo + (hi - lo) * i / 200.0 for i in range(201)]
    errs = []
    for x in grid:
        yh = predict(family, p, x)
        # interpolate gt via piecewise linear on (xs,ys)
        # find segment
        j = max(0, min(len(xs) - 2, next((k for k in range(len(xs) - 1) if xs[k] <= x <= xs[k + 1]), len(xs) - 2)))
        t = 0.0 if xs[j + 1] == xs[j] else (x - xs[j]) / (xs[j + 1] - xs[j])
        yg = ys[j] * (1 - t) + ys[j + 1] * t
        errs.append(abs(yh - yg))
    # report worst slices
    worst = sorted([(errs[i], grid[i]) for i in range(len(grid))], key=lambda t: -t[0])[:5]
    return {"ok": True, "worst": [{"x": x, "abs_err": e} for e, x in worst], "max_err": max(errs) if errs else 0.0}
