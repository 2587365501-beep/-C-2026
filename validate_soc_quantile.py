#!/usr/bin/env python3
"""Validate fixed and start-of-day-SOC-adaptive residual quantiles for problem 2.

The script imports the user's reproduction module, preserves its forecasting,
LP, and physical dispatch conventions, and uses a chronological train/test split.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("problem2_user", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def simulate(module, inputs, base_net, residual, q0: float, alpha: float) -> dict:
    actual_net = inputs.load_kwh - inputs.pv_kwh
    current_soc = 6000.0
    rows = []
    for d, current_date in enumerate(inputs.dates):
        start_soc = current_soc
        # Low SOC raises the service quantile; high SOC lowers it.
        soc_signal = (6000.0 - start_soc) / (module.SOC_MAX_KWH - 6000.0)
        q_day = float(np.clip(q0 + alpha * soc_signal, 0.50, 0.95))
        if d == 0:
            reserve = np.zeros(module.INTERVALS)
        else:
            reserve = np.quantile(residual[max(0, d - 28):d], q_day, axis=0)
        target = base_net[d] + reserve
        plan = module.solve_lp_day(target, inputs.prices, start_soc)
        actual = module.simulate_actual(plan.grid_kwh, actual_net[d], start_soc)
        current_soc = actual["end_soc"]
        plan_cost = float(np.dot(inputs.prices, plan.grid_kwh))
        emergency_cost = float(np.dot(5.0 * inputs.prices, actual["emergency"]))
        rows.append({
            "date": current_date,
            "q": q_day,
            "start_soc": start_soc,
            "plan_kwh": float(plan.grid_kwh.sum()),
            "plan_cost": plan_cost,
            "emergency_kwh": float(actual["emergency"].sum()),
            "emergency_cost": emergency_cost,
            "curtail_kwh": float(actual["curtail"].sum()),
            "total_cost": plan_cost + emergency_cost,
        })
    return {"rows": rows}


def summarize(result: dict, start: date, end: date) -> dict:
    rows = [r for r in result["rows"] if start <= r["date"] <= end]
    keys = ("plan_kwh", "plan_cost", "emergency_kwh", "emergency_cost", "curtail_kwh", "total_cost")
    out = {key: float(sum(r[key] for r in rows)) for key in keys}
    out.update({
        "days": len(rows),
        "mean_q": float(np.mean([r["q"] for r in rows])),
        "min_q": float(np.min([r["q"] for r in rows])),
        "max_q": float(np.max([r["q"] for r in rows])),
        "mean_start_soc": float(np.mean([r["start_soc"] for r in rows])),
    })
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--price", type=Path, required=True)
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fine", action="store_true", help="Fine search around the coarse SOC optimum")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    module = load_module(args.module)
    inputs = module.read_inputs(args.price, args.actual, args.template)
    base_net, _, residual = module.build_causal_target_net(inputs)

    train = (date(2025, 2, 1), date(2025, 6, 30))
    test = (date(2025, 7, 1), date(2025, 12, 31))
    full = (date(2025, 2, 1), date(2025, 12, 31))
    if args.fine:
        q_grid = [0.80, 0.87, 0.88, 0.89, 0.90, 0.91, 0.92]
        alpha_grid = [0.00, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14]
    else:
        q_grid = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
        alpha_grid = [0.00, 0.05, 0.10, 0.15, 0.20]

    candidates = []
    for q0 in q_grid:
        for alpha in alpha_grid:
            result = simulate(module, inputs, base_net, residual, q0, alpha)
            candidates.append({
                "q0": q0,
                "alpha": alpha,
                "train": summarize(result, *train),
                "test": summarize(result, *test),
                "full": summarize(result, *full),
            })
        print(f"completed q0={q0:.2f}", flush=True)

    baseline = next(x for x in candidates if x["q0"] == 0.80 and x["alpha"] == 0.0)
    best_fixed = min((x for x in candidates if x["alpha"] == 0.0), key=lambda x: x["train"]["total_cost"])
    best_soc = min(candidates, key=lambda x: x["train"]["total_cost"])

    def comparison(candidate):
        return {
            "q0": candidate["q0"],
            "alpha": candidate["alpha"],
            "train": candidate["train"],
            "test": candidate["test"],
            "full": candidate["full"],
            "test_saving_vs_q80": baseline["test"]["total_cost"] - candidate["test"]["total_cost"],
            "test_emergency_change_vs_q80": candidate["test"]["emergency_kwh"] - baseline["test"]["emergency_kwh"],
            "test_curtail_change_vs_q80": candidate["test"]["curtail_kwh"] - baseline["test"]["curtail_kwh"],
        }

    payload = {
        "rule": "q_d=clip(q0+alpha*(6000-SOC_d0)/4800,0.50,0.95)",
        "selection": "minimum Feb-Jun total cost; Jul-Dec untouched test",
        "baseline_q80": comparison(baseline),
        "train_selected_fixed": comparison(best_fixed),
        "train_selected_soc": comparison(best_soc),
        "candidates": candidates,
    }
    (args.output_dir / "soc_quantile_validation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    fixed = sorted((x for x in candidates if x["alpha"] == 0.0), key=lambda x: x["q0"])
    qs = [x["q0"] for x in fixed]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].plot(qs, [x["train"]["total_cost"] / 1e6 for x in fixed], "o-", label="Feb-Jun tuning")
    axes[0].plot(qs, [x["test"]["total_cost"] / 1e6 for x in fixed], "s-", label="Jul-Dec test")
    axes[0].axvline(0.8, color="gray", linestyle="--", linewidth=1)
    axes[0].set(xlabel="Residual quantile", ylabel="Total cost (million CNY)", title="Fixed-quantile sensitivity")
    axes[0].grid(alpha=0.25); axes[0].legend()
    axes[1].plot(qs, [x["test"]["emergency_kwh"] / 1e3 for x in fixed], "o-", label="Emergency")
    axes[1].plot(qs, [x["test"]["curtail_kwh"] / 1e3 for x in fixed], "s-", label="Curtailment")
    axes[1].set(xlabel="Residual quantile", ylabel="Energy (MWh)", title="Jul-Dec risk trade-off")
    axes[1].grid(alpha=0.25); axes[1].legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "soc_quantile_validation.png", dpi=220)
    print(json.dumps({k: payload[k] for k in ("baseline_q80", "train_selected_fixed", "train_selected_soc")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
