
from __future__ import annotations

from typing import Any


def check_goal_ctrl_linear(
    model: dict[str, Any],
    x0: list[float],
    ref: list[list[float]],
    lo: list[float],
    hi: list[float],
    var_x0: list[float] | None = None,
    margin: float = 0.0,
) -> dict[str, Any]:
    """Feasibility via simple bound propagation on controls and linear reachability.
    If reference states exceed what is reachable within box-limited controls (|u|), flag infeasible.
    margin: extra safety slack added to ref amplitudes.
    """
    A = model["A"]
    B = model["B"]
    b = model["b"]
    d = len(x0)
    m = len(lo)
    # crude bound over horizon: assume worst-case control at limits pointing toward target
    # compute max per-step impact of controls on each state dimension
    # impact_i = sum_j |B[i][j]| * max(|lo_j|, |hi_j|)
    imp = [sum(abs(B[i][j]) * max(abs(lo[j]), abs(hi[j])) for j in range(m)) for i in range(d)]
    xs = [x0[:]]
    feas = True
    reason = "ok"
    viol_step = None
    for t in range(1, len(ref)):
        # predict with zero control for baseline
        x = xs[-1]
        base = [sum(A[i][k] * x[k] for k in range(d)) + b[i] for i in range(d)]
        # reachable band at step t: base ± t*imp (very conservative upper bound)
        lo_r = [base[i] - t * imp[i] - margin for i in range(d)]
        hi_r = [base[i] + t * imp[i] + margin for i in range(d)]
        r = ref[t]
        if any(r[i] < lo_r[i] or r[i] > hi_r[i] for i in range(d)):
            feas = False
            reason = "unreachable_ref"
            viol_step = t
            break
        xs.append(base)  # keep baseline progression for band update
    return {"feasible": feas, "reason": reason, "violation_step": viol_step, "band_impact": imp}
