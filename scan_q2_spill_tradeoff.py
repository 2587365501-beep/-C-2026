from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from src.microgrid_model import (
    Q2_ADAPTIVE_CANDIDATES,
    Q2_SERVICE_QUANTILE,
    ROLLING_WINDOW,
    build_forecasts,
    load_data,
    simulate_adaptive_quantile_policy,
    summarize_year,
)


TOLERANCES = (0.0, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02)


def evaluate(tolerance: float) -> dict:
    root = Path(__file__).resolve().parent
    data = load_data(root)
    forecasts = build_forecasts(data)
    result, audit = simulate_adaptive_quantile_policy(
        data,
        forecasts["q2_point"],
        data.fixed_price,
        candidates=Q2_ADAPTIVE_CANDIDATES,
        history_window=ROLLING_WINDOW,
        default_quantile=Q2_SERVICE_QUANTILE,
        cost_tolerance=tolerance,
    )
    metrics = summarize_year(result)
    frequency = {
        f"{q:.2f}": int((audit["selected_quantile"][31:] == q).sum())
        for q in audit["candidates"]
    }
    return {"cost_tolerance": tolerance, **metrics, "selection_frequency": frequency}


def main() -> None:
    records = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(evaluate, value): value for value in TOLERANCES}
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
    records.sort(key=lambda row: row["cost_tolerance"])
    baseline_cost = 17_508_564.242372863
    feasible = [row for row in records if row["total_cost_yuan"] <= baseline_cost]
    selected = min(feasible, key=lambda row: (row["spill_kwh"], row["total_cost_yuan"]))
    payload = {
        "selection_rule": "在总成本不超过固定q=0.8基准的方案中选择弃电量最低者",
        "fixed_0_8_cost_yuan": baseline_cost,
        "selected_cost_tolerance": selected["cost_tolerance"],
        "records": records,
    }
    output = Path(__file__).resolve().parent / "outputs" / "q2_spill_tradeoff_scan.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
