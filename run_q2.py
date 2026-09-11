from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.microgrid_model import (
    Q2_ADAPTIVE_CANDIDATES,
    Q2_SERVICE_QUANTILE,
    Q2_SPILL_COST_TOLERANCE,
    ROLLING_WINDOW,
    _emergency_rows,
    _plan_rows,
    _storage_rows,
    build_forecasts,
    load_data,
    simulate_adaptive_quantile_policy,
    summarize_year,
    validate_result,
)


def _candidate_summary(audit: dict, start_index: int = 31) -> list[dict]:
    rows = []
    for q in audit["candidates"]:
        metrics = summarize_year(audit["candidate_results"][q], start_index=start_index)
        rows.append({"quantile": q, **metrics})
    return rows


def main() -> None:
    root = Path(__file__).resolve().parent
    output_dir = root / "outputs"
    output_dir.mkdir(exist_ok=True)
    data = load_data(root)
    forecasts = build_forecasts(data)

    adaptive_q2, audit = simulate_adaptive_quantile_policy(
        data,
        forecasts["q2_point"],
        data.fixed_price,
        candidates=Q2_ADAPTIVE_CANDIDATES,
        history_window=ROLLING_WINDOW,
        default_quantile=Q2_SERVICE_QUANTILE,
        cost_tolerance=Q2_SPILL_COST_TOLERANCE,
    )
    fixed_q2 = audit["candidate_results"][Q2_SERVICE_QUANTILE]

    adaptive_validation = validate_result(
        data, adaptive_q2, data.fixed_price, "问题2-逐日自适应分位数LP"
    )
    fixed_validation = validate_result(
        data, fixed_q2, data.fixed_price, "问题2-固定80%分位数LP基准"
    )
    selected = audit["selected_quantile"]
    scores = audit["history_cost_yuan"]
    spill_scores = audit["history_spill_kwh"]
    candidates = np.asarray(audit["candidates"], dtype=float)
    for day in range(ROLLING_WINDOW, len(data.dates)):
        minimum = float(np.min(scores[day]))
        eligible = np.flatnonzero(
            scores[day] <= minimum * (1.0 + audit["cost_tolerance"]) + 1.0e-8
        )
        expected_index = min(
            eligible,
            key=lambda index: (
                spill_scores[day, index],
                scores[day, index],
                abs(candidates[index] - Q2_SERVICE_QUANTILE),
                index,
            ),
        )
        expected = candidates[expected_index]
        if not np.isclose(selected[day], expected, atol=1.0e-12):
            raise RuntimeError(f"{data.dates[day]:%Y-%m-%d} 自适应分位数选择审计失败")
    leakage_audit = {
        "passed": bool(
            np.all(audit["history_days"][ROLLING_WINDOW:] == ROLLING_WINDOW)
            and np.all(
                audit["history_start_index"][ROLLING_WINDOW:]
                == np.arange(ROLLING_WINDOW, len(data.dates)) - ROLLING_WINDOW
            )
        ),
        "rule": "目标日d只汇总[d-28,d)的完整历史日成本；历史日h的预测和分位数裕度只使用h之前残差",
    }
    frozen_plan_audit = {
        "passed": bool(np.allclose(adaptive_q2.final_grid, adaptive_q2.plan_grid)),
        "max_plan_revision_kwh": float(
            np.max(np.abs(adaptive_q2.final_grid - adaptive_q2.plan_grid))
        ),
        "official_daily_lp_solves": int(audit["official_lp_solves"]),
        "intraday_reoptimizations": int(audit["intraday_reoptimizations"]),
    }
    if not (
        adaptive_validation["passed"]
        and fixed_validation["passed"]
        and leakage_audit["passed"]
        and frozen_plan_audit["passed"]
    ):
        raise RuntimeError(
            f"问题2验证未通过: adaptive={adaptive_validation}, fixed={fixed_validation}, "
            f"leakage={leakage_audit}, frozen={frozen_plan_audit}"
        )

    adaptive_summary = summarize_year(adaptive_q2)
    fixed_summary = summarize_year(fixed_q2)
    cost_saving = fixed_summary["total_cost_yuan"] - adaptive_summary["total_cost_yuan"]
    saving_rate = cost_saving / fixed_summary["total_cost_yuan"]
    comparison = {
        "fixed_0_8": fixed_summary,
        "adaptive": adaptive_summary,
        "cost_saving_yuan": float(cost_saving),
        "saving_rate": float(saving_rate),
    }

    evaluation = slice(31, None)
    selection_frequency = {
        f"{q:.2f}": int(np.sum(np.isclose(selected[evaluation], q)))
        for q in audit["candidates"]
    }
    summary_path = output_dir / "analysis_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    summary["problem2"] = adaptive_summary
    summary["problem2_fixed_0_8_baseline"] = fixed_summary
    summary["problem2_adaptive_comparison"] = comparison
    summary["problem2_model_selection"] = {
        "method": "每天0:00用最近28个完整历史日回测；在总成本不超过历史最低值0.25%的候选中，选择历史弃电量最低的当天统一分位数",
        "objective": "在近最优成本约束下最小化历史弃电量",
        "candidates": list(audit["candidates"]),
        "cost_tolerance": float(audit["cost_tolerance"]),
        "default_quantile_before_full_window": Q2_SERVICE_QUANTILE,
        "history_window_days": ROLLING_WINDOW,
        "selection_frequency_2025_02_01_to_2025_12_31": selection_frequency,
        "candidate_fixed_policy_summary": _candidate_summary(audit),
    }
    summary.pop("problem2_greedy_benchmark", None)
    summary.pop("problem2_executor_improvement", None)
    summary.setdefault("assumptions", {})["optimization_model_q2"] = (
        "逐日自适应分位数+连续LP；每天仅0:00选择一次分位数并求解一次144时段计划，日内不重预测、不重优化、不修改购电计划"
    )
    summary["assumptions"]["q2_realtime_dispatch"] = (
        "固定计划购电与实际光伏先供负荷；富余充电后弃电，缺口放电后紧急购电；实际日末SOC结转次日"
    )
    summary["assumptions"]["q2_adaptive_candidates"] = list(audit["candidates"])
    prior_validation = summary.get("validation", {})
    prior_validation.pop("问题2", None)
    summary["validation"] = {
        **prior_validation,
        "问题2-Adaptive": adaptive_validation,
        "问题2-Fixed0.8": fixed_validation,
        "问题2-无未来泄漏": leakage_audit,
        "问题2-日内计划冻结": frozen_plan_audit,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    daily_rows = []
    for day in range(31, len(data.dates)):
        row = {
            "日期": data.dates[day].strftime("%Y-%m-%d"),
            "选择分位数": float(selected[day]),
            "历史窗口开始": data.dates[audit["history_start_index"][day]].strftime("%Y-%m-%d"),
            "历史窗口结束": data.dates[day - 1].strftime("%Y-%m-%d"),
            "历史天数": int(audit["history_days"][day]),
        }
        for index, q in enumerate(audit["candidates"]):
            row[f"历史总成本_q{q:.2f}_元"] = float(scores[day, index])
            row[f"历史弃电量_q{q:.2f}_kWh"] = float(spill_scores[day, index])
        daily_rows.append(row)
    pd.DataFrame(daily_rows).to_csv(
        output_dir / "q2_adaptive_daily_quantile.csv", index=False, encoding="utf-8-sig"
    )
    comparison_rows = []
    for strategy, metrics in (("Fixed q=0.8", fixed_summary), ("Adaptive q_d", adaptive_summary)):
        comparison_rows.append({"策略": strategy, **metrics})
    pd.DataFrame(comparison_rows).to_csv(
        output_dir / "q2_adaptive_comparison.csv", index=False, encoding="utf-8-sig"
    )
    (output_dir / "q2_adaptive_summary.json").write_text(
        json.dumps(
            {
                "comparison": comparison,
                "selection_frequency": selection_frequency,
                "validation": {
                    "adaptive": adaptive_validation,
                    "fixed_0_8": fixed_validation,
                    "no_future_leakage": leakage_audit,
                    "frozen_intraday_plan": frozen_plan_audit,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    payload_path = output_dir / "workbook_payload.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["result2"] = {
        "plan_rows": _plan_rows(adaptive_q2, data.fixed_price),
        "storage_rows": _storage_rows(adaptive_q2),
        "emergency_rows": _emergency_rows(adaptive_q2, data.slot_labels),
    }
    payload_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    if "--compute-only" not in sys.argv:
        node = shutil.which("node")
        if node is None:
            node = str(
                Path.home()
                / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe"
            )
        subprocess.run(
            [node, str(root / "scripts/build_result_workbooks.mjs"), "result2"],
            cwd=root,
            check=True,
        )
    print(
        json.dumps(
            {
                "comparison": comparison,
                "selection_frequency": selection_frequency,
                "validation": summary["validation"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
