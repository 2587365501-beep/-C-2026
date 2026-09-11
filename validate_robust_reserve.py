#!/usr/bin/env python3
"""Out-of-sample validation of robust SOC reserve for problem 2."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("problem2_user_reserve", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def historical_reserves(residual: np.ndarray, day: int, q_r: float, horizon: int, eta: float) -> np.ndarray:
    """SOC reserve from the quantile of maximum positive cumulative error over H slots."""
    n = residual.shape[1]
    if day == 0:
        return np.zeros(n)
    samples = residual[max(0, day - 28):day]
    paths = np.zeros_like(samples)
    for j, row in enumerate(samples):
        for t in range(n):
            end = min(n, t + horizon)
            prefix = np.cumsum(row[t:end])
            paths[j, t] = max(0.0, float(np.max(prefix))) if len(prefix) else 0.0
    return np.quantile(paths, q_r, axis=0) / eta


def solve_reserve_day(module, target, prices, start_soc, reserve_soc):
    n = module.INTERVALS
    # Variables: grid, charge, discharge, spill, SOC, reserve shortfall.
    m = 6 * n
    g0, c0, d0, w0, s0, x0 = (k * n for k in range(6))
    objective = np.zeros(m)
    objective[g0:g0+n] = prices
    objective[c0:c0+n] = module.EPS_THROUGHPUT_COST
    objective[d0:d0+n] = module.EPS_THROUGHPUT_COST
    # One internal SOC kWh can deliver eta kWh; value reserve shortage at emergency price.
    objective[x0:x0+n] = 5.0 * prices * module.ETA

    a_eq = lil_matrix((2*n + 1, m), dtype=float)
    b_eq = np.zeros(2*n + 1)
    for t in range(n):
        a_eq[t, g0+t] = 1.0; a_eq[t, d0+t] = 1.0
        a_eq[t, c0+t] = -1.0; a_eq[t, w0+t] = -1.0
        b_eq[t] = target[t]
        row = n + t
        a_eq[row, s0+t] = 1.0; a_eq[row, c0+t] = -module.ETA
        a_eq[row, d0+t] = 1.0/module.ETA
        if t == 0:
            b_eq[row] = start_soc
        else:
            a_eq[row, s0+t-1] = -1.0
    a_eq[2*n, s0+n-1] = 1.0
    b_eq[2*n] = start_soc

    # SOC + shortfall >= physical minimum + robust reserve.
    a_ub = lil_matrix((n, m), dtype=float)
    b_ub = np.zeros(n)
    for t in range(n):
        a_ub[t, s0+t] = -1.0
        a_ub[t, x0+t] = -1.0
        b_ub[t] = -(module.SOC_MIN_KWH + reserve_soc[t])
    bounds = (
        [(0.0, None)] * n
        + [(0.0, module.MAX_FLOW_KWH)] * (2*n)
        + [(0.0, None)] * n
        + [(module.SOC_MIN_KWH, module.SOC_MAX_KWH)] * n
        + [(0.0, None)] * n
    )
    result = linprog(objective, A_ub=a_ub.tocsr(), b_ub=b_ub,
                     A_eq=a_eq.tocsr(), b_eq=b_eq, bounds=bounds, method="highs")
    if not result.success:
        raise RuntimeError(result.message)
    x = result.x
    return x[g0:g0+n], x[x0:x0+n]


def run(module, inputs, base_net, residual, q_n, q_r, horizon):
    actual_net = inputs.load_kwh - inputs.pv_kwh
    current_soc = 6000.0
    rows = []
    for d, current_date in enumerate(inputs.dates):
        hist = residual[max(0, d-28):d]
        margin = np.zeros(module.INTERVALS) if d == 0 else np.quantile(hist, q_n, axis=0)
        reserve = historical_reserves(residual, d, q_r, horizon, module.ETA)
        grid, shortfall = solve_reserve_day(module, base_net[d] + margin, inputs.prices, current_soc, reserve)
        actual = module.simulate_actual(grid, actual_net[d], current_soc)
        plan_cost = float(np.dot(inputs.prices, grid))
        emergency_cost = float(np.dot(5.0 * inputs.prices, actual["emergency"]))
        rows.append({
            "date": current_date, "plan_cost": plan_cost,
            "emergency_kwh": float(actual["emergency"].sum()),
            "emergency_cost": emergency_cost,
            "curtail_kwh": float(actual["curtail"].sum()),
            "total_cost": plan_cost + emergency_cost,
            "reserve_shortfall_kwh": float(shortfall.sum()),
            "soc_floor_hits": int(np.sum(actual["soc_path"] <= module.SOC_MIN_KWH + reserve + 1e-6)),
        })
        current_soc = actual["end_soc"]
    return rows


def summarize(rows, start, end):
    use = [r for r in rows if start <= r["date"] <= end]
    keys = ("plan_cost", "emergency_kwh", "emergency_cost", "curtail_kwh",
            "total_cost", "reserve_shortfall_kwh", "soc_floor_hits")
    return {**{k: float(sum(r[k] for r in use)) for k in keys}, "days": len(use)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--module", type=Path, required=True); p.add_argument("--price", type=Path, required=True)
    p.add_argument("--actual", type=Path, required=True); p.add_argument("--template", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    module = load_module(args.module)
    inputs = module.read_inputs(args.price, args.actual, args.template)
    base_net, _, residual = module.build_causal_target_net(inputs)
    train = (date(2025,2,1), date(2025,6,30)); test = (date(2025,7,1), date(2025,12,31))
    full = (date(2025,2,1), date(2025,12,31))
    candidates = []
    for q_n in (0.65, 0.70, 0.75):
        for q_r in (0.70, 0.80, 0.90):
            for horizon in (6, 12, 24):
                rows = run(module, inputs, base_net, residual, q_n, q_r, horizon)
                candidates.append({"q_n":q_n,"q_r":q_r,"horizon":horizon,
                                   "train":summarize(rows,*train),"test":summarize(rows,*test),
                                   "full":summarize(rows,*full)})
            print(f"completed qN={q_n:.2f}, qR={q_r:.2f}", flush=True)
    best = min(candidates, key=lambda x: x["train"]["total_cost"])
    payload = {"selection":"minimum Feb-Jun total cost; Jul-Dec untouched test",
               "reserve":"qR of max positive cumulative residual over next H slots, divided by eta",
               "best":best,"candidates":candidates}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(best,ensure_ascii=False,indent=2))

if __name__ == "__main__":
    main()
