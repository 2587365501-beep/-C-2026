from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.microgrid_model import (
    _emergency_rows,
    _plan_rows,
    _storage_rows,
    build_forecasts,
    load_data,
    rolling_residual_quantile,
    simulate_fixed_policy,
    simulate_lookahead_policy,
    summarize_year,
    tune_q2_quantile,
    validate_result,
)


def main() -> None:
    root = Path(__file__).resolve().parent
    output_dir = root / "outputs"
    output_dir.mkdir(exist_ok=True)
    data = load_data(root)
    forecasts = build_forecasts(data)
    selected_quantile, tuning = tune_q2_quantile(data, forecasts["q2_point"])
    robust = forecasts["q2_point"] + rolling_residual_quantile(
        data.load - data.pv, forecasts["q2_point"], selected_quantile
    )
    calibrated_q2 = simulate_fixed_policy(data, robust, data.fixed_price)
    baseline_robust = forecasts["q2_point"] + rolling_residual_quantile(
        data.load - data.pv, forecasts["q2_point"], 0.80
    )
    baseline_q2 = simulate_fixed_policy(data, baseline_robust, data.fixed_price)
    calibrated_holdout = summarize_year(calibrated_q2, start_index=181)
    baseline_holdout = summarize_year(baseline_q2, start_index=181)
    deployed_quantile = selected_quantile
    q2 = calibrated_q2
    if calibrated_holdout["total_cost_yuan"] >= baseline_holdout["total_cost_yuan"]:
        deployed_quantile = 0.80
        q2 = baseline_q2
    greedy_q2 = q2
    q2 = simulate_lookahead_policy(
        data, baseline_robust if deployed_quantile == 0.80 else robust,
        forecasts["q2_point"], data.fixed_price
    )
    validation = validate_result(data, q2, data.fixed_price, "问题2-日前LP+滚动前瞻LP")
    if not validation["passed"]:
        raise RuntimeError(f"问题2验证未通过: {validation}")

    summary_path = output_dir / "analysis_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    summary["problem2"] = summarize_year(q2)
    summary["problem2_greedy_benchmark"] = summarize_year(greedy_q2)
    lookahead_holdout = summarize_year(q2, start_index=181)
    greedy_holdout = summarize_year(greedy_q2, start_index=181)
    summary["problem2_executor_improvement"] = {
        "total_cost_saving_yuan": float(greedy_q2.total_cost[31:].sum() - q2.total_cost[31:].sum()),
        "emergency_cost_saving_yuan": float(greedy_q2.emergency_cost[31:].sum() - q2.emergency_cost[31:].sum()),
        "emergency_reduction_kwh": float(greedy_q2.emergency[31:].sum() - q2.emergency[31:].sum()),
        "relative_total_cost_saving": float(
            (greedy_q2.total_cost[31:].sum() - q2.total_cost[31:].sum())
            / greedy_q2.total_cost[31:].sum()
        ),
        "holdout_period": "2025-07-01至2025-12-31",
        "lookahead_holdout": lookahead_holdout,
        "greedy_holdout": greedy_holdout,
        "holdout_total_cost_saving_yuan": float(
            greedy_holdout["total_cost_yuan"] - lookahead_holdout["total_cost_yuan"]
        ),
    }
    summary["problem2_model_selection"] = {
        "method": "用2—6月校准风险分位数，7—12月作为独立时间外评测期",
        "objective": "计划购电成本+缺电惩罚成本最小",
        "selected_quantile": selected_quantile,
        "deployed_quantile": deployed_quantile,
        "selection_decision": (
            f"校准期候选值{selected_quantile:.0%}在独立检验期未优于80%，最终采用80%以避免过拟合"
            if deployed_quantile == 0.80 and selected_quantile != 0.80
            else "采用校准期最优分位数"
        ),
        "candidates": tuning,
        "holdout_period": "2025-07-01至2025-12-31",
        "calibrated_holdout": calibrated_holdout,
        "baseline_80_holdout": baseline_holdout,
        "calibrated_holdout_cost_saving_vs_80_yuan": baseline_holdout["total_cost_yuan"] - calibrated_holdout["total_cost_yuan"],
    }
    summary.setdefault("assumptions", {})["optimization_model_q2"] = (
        "两层连续线性规划：日前LP确定计划购电，实时滚动前瞻LP依据当期实测偏差、历史残差持续性和剩余时段电价分配储能"
    )
    summary["assumptions"]["q2_realtime_dispatch"] = (
        "逐10分钟重求剩余日内LP并只执行首个动作；允许为更高价值的未来缺口保留电量，紧急购电仅用于负荷缺口，不主动给电池充电；日末SOC结转次日"
    )
    summary["assumptions"]["q2_quantile"] = deployed_quantile
    summary["validation"] = {**summary.get("validation", {}), "问题2": validation}
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(tuning).to_csv(output_dir / "q2_quantile_sensitivity.csv", index=False, encoding="utf-8-sig")

    payload_path = output_dir / "workbook_payload.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["result2"] = {
        "plan_rows": _plan_rows(q2, data.fixed_price),
        "storage_rows": _storage_rows(q2),
        "emergency_rows": _emergency_rows(q2, data.slot_labels),
    }
    payload_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    if "--compute-only" not in sys.argv:
        node = shutil.which("node")
        if node is None:
            node = str(Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe")
        subprocess.run([node, str(root / "scripts/build_result_workbooks.mjs"), "result2"], cwd=root, check=True)
    print(json.dumps({"problem2": summary["problem2"], "model_selection": summary["problem2_model_selection"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
