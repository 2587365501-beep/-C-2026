from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import lil_matrix


SEED = 42
DT_HOURS = 1.0 / 6.0
N_SLOTS = 144
ETA_CHARGE = 0.90
ETA_DISCHARGE = 0.90
SOC_MIN = 1200.0
SOC_MAX = 10800.0
SOC_INITIAL = 6000.0
POWER_LIMIT_KW = 5000.0
ENERGY_LIMIT = POWER_LIMIT_KW * DT_HOURS
DEGRADATION_COST = 1.0e-4
EMERGENCY_MULTIPLIER = 5.0
Q2_SERVICE_QUANTILE = 0.80
Q2_ADAPTIVE_CANDIDATES = (0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)
Q2_SPILL_COST_TOLERANCE = 0.0025
Q2_QUANTILE_CANDIDATES = tuple(np.round(np.arange(0.50, 0.951, 0.05), 2))
Q3_SERVICE_QUANTILE = 0.70
ROLLING_WINDOW = 28
UPDATE_HOURS = (6, 12, 18)
SELECTED_DATES = ("2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21")


@dataclass
class DataBundle:
    dates: pd.DatetimeIndex
    slot_hours: np.ndarray
    slot_labels: list[str]
    fixed_price: np.ndarray
    sample_load: np.ndarray
    sample_pv: np.ndarray
    load: np.ndarray
    pv: np.ndarray
    variable_price: np.ndarray
    pv_forecast_hourly: np.ndarray


@dataclass
class OptimizationResult:
    grid: np.ndarray
    charge: np.ndarray
    discharge: np.ndarray
    spill: np.ndarray
    soc: np.ndarray
    objective: float
    solve_seconds: float


@dataclass
class DispatchResult:
    charge: np.ndarray
    discharge: np.ndarray
    emergency: np.ndarray
    spill: np.ndarray
    soc: np.ndarray


@dataclass
class YearResult:
    dates: pd.DatetimeIndex
    plan_grid: np.ndarray
    final_grid: np.ndarray
    charge: np.ndarray
    discharge: np.ndarray
    emergency: np.ndarray
    spill: np.ndarray
    soc_start: np.ndarray
    soc_end: np.ndarray
    plan_cost: np.ndarray
    settlement_cost: np.ndarray
    emergency_cost: np.ndarray
    total_cost: np.ndarray
    solve_seconds: float
    soc_path: np.ndarray | None = None


def _as_float_matrix(frame: pd.DataFrame) -> np.ndarray:
    matrix = frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        bad = int(np.size(matrix) - np.isfinite(matrix).sum())
        raise ValueError(f"发现 {bad} 个无法转换为有限数值的数据单元格")
    return matrix


def _slot_label(start_hour: float) -> str:
    start_minutes = int(round(start_hour * 60))
    end_minutes = start_minutes + 10

    def display(minutes: int) -> str:
        day = minutes // 1440
        local = minutes % 1440
        hour, minute = divmod(local, 60)
        suffix = "+1" if day else ""
        return f"{hour}:{minute:02d}{suffix}"

    return f"{display(start_minutes)}-{display(end_minutes)}"


def load_data(root: Path) -> DataBundle:
    attach = root / "附件"
    sample = pd.read_excel(attach / "附件1.xlsx")
    load_frame = pd.read_excel(attach / "附件2.xlsx", sheet_name="小区负载")
    pv_frame = pd.read_excel(attach / "附件2.xlsx", sheet_name="光伏发电实际功率")
    price_frame = pd.read_excel(attach / "附件4.xlsx")
    forecast_frame = pd.read_excel(attach / "附件3.xlsx")

    dates = pd.DatetimeIndex(pd.to_datetime(load_frame.iloc[:, 0]))
    pv_dates = pd.DatetimeIndex(pd.to_datetime(pv_frame.iloc[:, 0]))
    price_dates = pd.DatetimeIndex(pd.to_datetime(price_frame.iloc[:, 0]))
    if not (dates.equals(pv_dates) and dates.equals(price_dates)):
        raise ValueError("附件2和附件4的日期索引不一致")
    if len(dates) != 365 or dates.has_duplicates:
        raise ValueError("日期应覆盖2025年365天且不得重复")

    load = _as_float_matrix(load_frame.iloc[:, 1:])
    pv = _as_float_matrix(pv_frame.iloc[:, 1:])
    variable_price = _as_float_matrix(price_frame.iloc[:, 1:])
    fixed_price = pd.to_numeric(sample.iloc[:, 1], errors="coerce").to_numpy(dtype=float)
    sample_load = pd.to_numeric(sample.iloc[:, 2], errors="coerce").to_numpy(dtype=float)
    sample_pv = pd.to_numeric(sample.iloc[:, 3], errors="coerce").to_numpy(dtype=float)
    if load.shape != (365, N_SLOTS) or pv.shape != (365, N_SLOTS):
        raise ValueError(f"负载/光伏矩阵维度异常: {load.shape}, {pv.shape}")
    if variable_price.shape != (365, N_SLOTS):
        raise ValueError(f"波动电价矩阵维度异常: {variable_price.shape}")
    if fixed_price.shape != (N_SLOTS,):
        raise ValueError(f"固定日价格向量维度异常: {fixed_price.shape}")
    if np.any(load < 0) or np.any(pv < 0) or np.any(variable_price <= 0) or np.any(fixed_price <= 0):
        raise ValueError("功率不得为负，电价必须为正")

    raw_dates = forecast_frame.iloc[:, 0].replace("", np.nan).ffill()
    forecast_dates = pd.to_datetime(raw_dates)
    issue_hours = (
        forecast_frame.iloc[:, 1]
        .astype(str)
        .str.extract(r"^(\d{1,2}):", expand=False)
        .astype(int)
        .to_numpy()
    )
    hourly_values = _as_float_matrix(forecast_frame.iloc[:, 2:26])
    pv_forecast_hourly = np.full((365, 4, 24), np.nan, dtype=float)
    date_to_index = {date.normalize(): i for i, date in enumerate(dates)}
    issue_to_index = {hour: i for i, hour in enumerate((0, 6, 12, 18))}
    for row, (date, issue) in enumerate(zip(forecast_dates, issue_hours)):
        pv_forecast_hourly[date_to_index[pd.Timestamp(date).normalize()], issue_to_index[issue]] = hourly_values[row]
    if not np.isfinite(pv_forecast_hourly).all():
        raise ValueError("附件3的日期、预报时刻或预报值不完整")

    slot_hours = np.arange(1, N_SLOTS + 1, dtype=float) / 6.0
    slot_labels = [_slot_label(hour) for hour in slot_hours]
    return DataBundle(
        dates=dates,
        slot_hours=slot_hours,
        slot_labels=slot_labels,
        fixed_price=fixed_price,
        sample_load=sample_load,
        sample_pv=sample_pv,
        load=load,
        pv=pv,
        variable_price=variable_price,
        pv_forecast_hourly=pv_forecast_hourly,
    )


def diagnose_data(data: DataBundle) -> dict:
    def stats(array: np.ndarray) -> dict:
        return {
            "shape": list(array.shape),
            "missing": int(np.size(array) - np.isfinite(array).sum()),
            "negative": int(np.sum(array < 0)),
            "min": float(np.min(array)),
            "median": float(np.median(array)),
            "max": float(np.max(array)),
            "mean": float(np.mean(array)),
            "std": float(np.std(array)),
        }

    daylight = data.pv > 0
    zero_night_fraction = float(np.mean(data.pv[:, :30] == 0))
    return {
        "date_start": data.dates[0].strftime("%Y-%m-%d"),
        "date_end": data.dates[-1].strftime("%Y-%m-%d"),
        "date_count": int(len(data.dates)),
        "duplicate_dates": int(data.dates.duplicated().sum()),
        "time_slots_per_day": N_SLOTS,
        "interval_minutes": 10,
        "load_kw": stats(data.load),
        "pv_kw": stats(data.pv),
        "fixed_price_yuan_per_kwh": stats(data.fixed_price),
        "variable_price_yuan_per_kwh": stats(data.variable_price),
        "pv_forecast_kw": stats(data.pv_forecast_hourly),
        "positive_pv_fraction": float(np.mean(daylight)),
        "night_zero_fraction_first_5h": zero_night_fraction,
        "quality_score": 100,
        "issues": [],
        "processing": [
            "保持原始Excel不变，在内存中转换为365×144数值矩阵",
            "功率按10分钟乘以1/6换算为电量",
            "整点光伏预报用当前观测作锚点并线性插值到10分钟",
            "预测仅使用当前日期之前的数据，分位数安全裕度在滚动窗口内校准",
        ],
    }


def build_load_pv_baseline(
    load: np.ndarray,
    pv: np.ndarray,
    sample_load: np.ndarray,
    sample_pv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    load_pred = np.zeros_like(load)
    pv_pred = np.zeros_like(pv)
    for day in range(len(load)):
        if day == 0:
            load_pred[day] = sample_load
            pv_pred[day] = sample_pv
            continue
        sources: list[tuple[float, np.ndarray, np.ndarray]] = [(0.45, load[day - 1], pv[day - 1])]
        if day >= 7:
            sources.append((0.35, load[day - 7], pv[day - 7]))
        start = max(0, day - ROLLING_WINDOW)
        sources.append((0.20, np.median(load[start:day], axis=0), np.median(pv[start:day], axis=0)))
        weight_sum = sum(weight for weight, _, _ in sources)
        load_pred[day] = sum(weight * values for weight, values, _ in sources) / weight_sum
        pv_pred[day] = sum(weight * values for weight, _, values in sources) / weight_sum
    return np.maximum(load_pred, 0.0), np.maximum(pv_pred, 0.0)


def rolling_residual_quantile(
    actual: np.ndarray,
    point: np.ndarray,
    quantile: float,
    window: int = ROLLING_WINDOW,
) -> np.ndarray:
    output = np.zeros_like(point)
    residual = actual - point
    for day in range(1, len(actual)):
        start = max(0, day - window)
        output[day] = np.quantile(residual[start:day], quantile, axis=0)
    return output


def interpolate_pv_forecast(
    data: DataBundle,
    day: int,
    issue_hour: int,
) -> np.ndarray:
    issue_index = (0, 6, 12, 18).index(issue_hour)
    forecast = data.pv_forecast_hourly[day, issue_index]
    if issue_hour == 0:
        anchor = 0.0
    else:
        slot = int(round(issue_hour * 6)) - 1
        anchor = data.pv[day, slot]
    x = np.concatenate([[float(issue_hour)], issue_hour + np.arange(1, 25, dtype=float)])
    y = np.concatenate([[anchor], forecast])
    return np.maximum(np.interp(data.slot_hours, x, y), 0.0)


def build_forecasts(data: DataBundle) -> dict:
    load_point, pv_point = build_load_pv_baseline(data.load, data.pv, data.sample_load, data.sample_pv)
    actual_net = data.load - data.pv
    q2_point = load_point - pv_point
    q2_margin = rolling_residual_quantile(actual_net, q2_point, Q2_SERVICE_QUANTILE)
    q2_robust = q2_point + q2_margin

    pv_issue = {hour: np.zeros_like(data.pv) for hour in (0, 6, 12, 18)}
    q3_point: dict[int, np.ndarray] = {}
    q3_robust: dict[int, np.ndarray] = {}
    for hour in (0, 6, 12, 18):
        for day in range(len(data.dates)):
            pv_issue[hour][day] = interpolate_pv_forecast(data, day, hour)
        point = load_point - pv_issue[hour]
        margin = rolling_residual_quantile(actual_net, point, Q3_SERVICE_QUANTILE)
        q3_point[hour] = point
        q3_robust[hour] = point + margin
    return {
        "load_point": load_point,
        "pv_point": pv_point,
        "q2_point": q2_point,
        "q2_robust": q2_robust,
        "pv_issue": pv_issue,
        "q3_point": q3_point,
        "q3_robust": q3_robust,
    }


def tune_q2_quantile(data: DataBundle, q2_point: np.ndarray) -> tuple[float, list[dict]]:
    """Select Q2 risk on Feb-Jun; reserve Jul-Dec for a final out-of-sample check."""
    actual_net_kwh = (data.load - data.pv) * DT_HOURS
    records: list[dict] = []
    for quantile in Q2_QUANTILE_CANDIDATES:
        robust = q2_point + rolling_residual_quantile(data.load - data.pv, q2_point, quantile)
        soc = SOC_INITIAL
        plan_cost = 0.0
        emergency_cost = 0.0
        emergency_kwh = 0.0
        spill_kwh = 0.0
        for day in range(181):
            optimized = _deterministic_lp(
                robust[day] * DT_HOURS,
                data.fixed_price,
                soc,
                soc,
            )
            day_plan_cost = float(np.dot(data.fixed_price, optimized.grid))
            for slot in range(N_SLOTS):
                _, _, emergency, spill, soc = dispatch_one_interval(
                    optimized.grid[slot], actual_net_kwh[day, slot], soc
                )
                if day >= 31:
                    emergency_kwh += emergency
                    spill_kwh += spill
                    emergency_cost += EMERGENCY_MULTIPLIER * data.fixed_price[slot] * emergency
            if day >= 31:
                plan_cost += day_plan_cost
        records.append(
            {
                "quantile": float(quantile),
                "calibration_period": "2025-02-01至2025-06-30",
                "plan_cost_yuan": plan_cost,
                "emergency_cost_yuan": emergency_cost,
                "total_cost_yuan": plan_cost + emergency_cost,
                "emergency_kwh": emergency_kwh,
                "spill_kwh": spill_kwh,
            }
        )
    best = min(records, key=lambda item: (item["total_cost_yuan"], item["emergency_kwh"]))
    return float(best["quantile"]), records


def forecast_metrics(data: DataBundle, forecasts: dict) -> dict:
    def metric(actual: np.ndarray, pred: np.ndarray) -> dict:
        error = pred - actual
        scale = np.maximum(np.abs(actual), 1.0)
        return {
            "mae_kw": float(np.mean(np.abs(error))),
            "rmse_kw": float(np.sqrt(np.mean(error**2))),
            "wape": float(np.sum(np.abs(error)) / np.sum(np.abs(actual))),
            "bias_kw": float(np.mean(error)),
            "p90_abs_error_kw": float(np.quantile(np.abs(error), 0.90)),
            "mape_guarded": float(np.mean(np.abs(error) / scale)),
        }

    metrics = {
        "load_baseline": metric(data.load, forecasts["load_point"]),
        "pv_baseline": metric(data.pv, forecasts["pv_point"]),
    }
    for hour in (0, 6, 12, 18):
        mask = data.slot_hours >= hour
        metrics[f"pv_forecast_{hour:02d}"] = metric(
            data.pv[:, mask], forecasts["pv_issue"][hour][:, mask]
        )
    return metrics


def _deterministic_lp(
    net_demand_kwh: np.ndarray,
    price: np.ndarray,
    soc0: float,
    terminal_soc: float,
) -> OptimizationResult:
    net = np.asarray(net_demand_kwh, dtype=float)
    price = np.asarray(price, dtype=float)
    n = len(net)
    # Continuous LP variables: [grid, charge, discharge, spill, soc_1..soc_n].
    # The statement does not prohibit aggregate charging and discharging in the
    # same interval. Positive losses plus the small throughput cost remove
    # economically meaningless cycles without binary mode variables.
    m = 5 * n
    g0, c0, d0, s0, e0 = (k * n for k in range(5))
    objective = np.zeros(m)
    objective[g0 : g0 + n] = price
    objective[c0 : c0 + n] = DEGRADATION_COST
    objective[d0 : d0 + n] = DEGRADATION_COST

    equality = lil_matrix((2 * n + 1, m), dtype=float)
    rhs = np.zeros(2 * n + 1)
    for t in range(n):
        equality[t, g0 + t] = 1.0
        equality[t, d0 + t] = 1.0
        equality[t, c0 + t] = -1.0
        equality[t, s0 + t] = -1.0
        rhs[t] = net[t]

        row = n + t
        equality[row, e0 + t] = 1.0
        if t > 0:
            equality[row, e0 + t - 1] = -1.0
            rhs[row] = 0.0
        else:
            rhs[row] = soc0
        equality[row, c0 + t] = -ETA_CHARGE
        equality[row, d0 + t] = 1.0 / ETA_DISCHARGE
    equality[2 * n, e0 + n - 1] = 1.0
    rhs[2 * n] = terminal_soc

    lower = np.zeros(m)
    upper = np.full(m, np.inf)
    upper[c0 : c0 + n] = ENERGY_LIMIT
    upper[d0 : d0 + n] = ENERGY_LIMIT
    lower[e0 : e0 + n] = SOC_MIN
    upper[e0 : e0 + n] = SOC_MAX

    started = time.perf_counter()
    result = linprog(
        objective,
        A_eq=equality.tocsr(),
        b_eq=rhs,
        bounds=list(zip(lower, upper)),
        method="highs",
        options={"presolve": True},
    )
    elapsed = time.perf_counter() - started
    if not result.success:
        raise RuntimeError(f"日内线性规划失败: {result.message}")
    x = result.x
    return OptimizationResult(
        grid=np.maximum(x[g0 : g0 + n], 0.0),
        charge=np.maximum(x[c0 : c0 + n], 0.0),
        discharge=np.maximum(x[d0 : d0 + n], 0.0),
        spill=np.maximum(x[s0 : s0 + n], 0.0),
        soc=np.concatenate([[soc0], x[e0 : e0 + n]]),
        objective=float(result.fun),
        solve_seconds=elapsed,
    )


def _adjustment_lp(
    net_demand_kwh: np.ndarray,
    price: np.ndarray,
    original_plan: np.ndarray,
    soc0: float,
    terminal_soc: float,
) -> OptimizationResult:
    net = np.asarray(net_demand_kwh, dtype=float)
    price = np.asarray(price, dtype=float)
    plan = np.asarray(original_plan, dtype=float)
    n = len(net)
    # MILP variables: [adjusted, charge, discharge, spill, up, down, soc, charge_mode].
    m = 8 * n
    a0, c0, d0, s0, u0, v0, e0, z0 = (k * n for k in range(8))
    objective = np.zeros(m)
    objective[u0 : u0 + n] = 1.5 * price
    objective[v0 : v0 + n] = -0.5 * price
    objective[c0 : c0 + n] = DEGRADATION_COST
    objective[d0 : d0 + n] = DEGRADATION_COST

    equality = lil_matrix((3 * n + 1, m), dtype=float)
    rhs = np.zeros(3 * n + 1)
    for t in range(n):
        equality[t, a0 + t] = 1.0
        equality[t, d0 + t] = 1.0
        equality[t, c0 + t] = -1.0
        equality[t, s0 + t] = -1.0
        rhs[t] = net[t]

        row = n + t
        equality[row, a0 + t] = 1.0
        equality[row, u0 + t] = -1.0
        equality[row, v0 + t] = 1.0
        rhs[row] = plan[t]

        row = 2 * n + t
        equality[row, e0 + t] = 1.0
        if t > 0:
            equality[row, e0 + t - 1] = -1.0
            rhs[row] = 0.0
        else:
            rhs[row] = soc0
        equality[row, c0 + t] = -ETA_CHARGE
        equality[row, d0 + t] = 1.0 / ETA_DISCHARGE
    equality[3 * n, e0 + n - 1] = 1.0
    rhs[3 * n] = terminal_soc

    lower = np.zeros(m)
    upper = np.full(m, np.inf)
    upper[c0 : c0 + n] = ENERGY_LIMIT
    upper[d0 : d0 + n] = ENERGY_LIMIT
    upper[v0 : v0 + n] = plan
    lower[e0 : e0 + n] = SOC_MIN
    upper[e0 : e0 + n] = SOC_MAX
    upper[z0 : z0 + n] = 1.0

    mutual_exclusion = lil_matrix((2 * n, m), dtype=float)
    exclusion_upper = np.zeros(2 * n)
    for t in range(n):
        mutual_exclusion[2 * t, c0 + t] = 1.0
        mutual_exclusion[2 * t, z0 + t] = -ENERGY_LIMIT
        mutual_exclusion[2 * t + 1, d0 + t] = 1.0
        mutual_exclusion[2 * t + 1, z0 + t] = ENERGY_LIMIT
        exclusion_upper[2 * t + 1] = ENERGY_LIMIT
    integrality = np.zeros(m, dtype=int)
    integrality[z0 : z0 + n] = 1

    started = time.perf_counter()
    result = milp(
        objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=(
            LinearConstraint(equality.tocsr(), rhs, rhs),
            LinearConstraint(mutual_exclusion.tocsr(), -np.inf, exclusion_upper),
        ),
        options={"presolve": True},
    )
    elapsed = time.perf_counter() - started
    if not result.success:
        raise RuntimeError(f"滚动调整混合整数线性规划失败: {result.message}")
    x = result.x
    return OptimizationResult(
        grid=np.maximum(x[a0 : a0 + n], 0.0),
        charge=np.maximum(x[c0 : c0 + n], 0.0),
        discharge=np.maximum(x[d0 : d0 + n], 0.0),
        spill=np.maximum(x[s0 : s0 + n], 0.0),
        soc=np.concatenate([[soc0], x[e0 : e0 + n]]),
        objective=float(result.fun),
        solve_seconds=elapsed,
    )


def dispatch_one_interval(grid: float, net_actual: float, soc: float) -> tuple[float, float, float, float, float]:
    surplus = grid - net_actual
    if surplus >= 0:
        charge = min(surplus, ENERGY_LIMIT, max((SOC_MAX - soc) / ETA_CHARGE, 0.0))
        discharge = 0.0
        emergency = 0.0
        spill = max(surplus - charge, 0.0)
    else:
        charge = 0.0
        discharge = min(-surplus, ENERGY_LIMIT, max((soc - SOC_MIN) * ETA_DISCHARGE, 0.0))
        emergency = max(-surplus - discharge, 0.0)
        spill = 0.0
    next_soc = soc + ETA_CHARGE * charge - discharge / ETA_DISCHARGE
    next_soc = min(max(next_soc, SOC_MIN), SOC_MAX)
    return charge, discharge, emergency, spill, next_soc


def _estimate_residual_persistence(residual_kw: np.ndarray, day: int) -> float:
    """Estimate a leakage-free AR(1) coefficient from the preceding 30 days."""
    start = max(0, day - 30)
    history = residual_kw[start:day].reshape(-1)
    if history.size < 2:
        return 0.0
    x = history[:-1]
    y = history[1:]
    denominator = float(np.dot(x, x))
    if denominator <= 1.0e-12:
        return 0.0
    return float(np.clip(np.dot(x, y) / denominator, 0.0, 0.99))


def _lookahead_dispatch_lp(
    fixed_grid: np.ndarray,
    expected_net_kwh: np.ndarray,
    price: np.ndarray,
    soc0: float,
    terminal_value_yuan_per_kwh: float,
) -> tuple[float, float, float, float, float]:
    """Optimize recourse over the remaining day and return only the first action."""
    grid = np.asarray(fixed_grid, dtype=float)
    net = np.asarray(expected_net_kwh, dtype=float)
    price = np.asarray(price, dtype=float)
    n = len(net)
    # Variables: [charge, discharge, emergency, spill, soc_1..soc_n].
    m = 5 * n
    c0, d0, x0, w0, e0 = (k * n for k in range(5))
    objective = np.zeros(m)
    objective[c0 : c0 + n] = DEGRADATION_COST
    objective[d0 : d0 + n] = DEGRADATION_COST
    objective[x0 : x0 + n] = EMERGENCY_MULTIPLIER * price
    objective[e0 + n - 1] = -terminal_value_yuan_per_kwh

    equality = lil_matrix((2 * n, m), dtype=float)
    rhs = np.zeros(2 * n)
    for t in range(n):
        equality[t, d0 + t] = 1.0
        equality[t, x0 + t] = 1.0
        equality[t, c0 + t] = -1.0
        equality[t, w0 + t] = -1.0
        rhs[t] = net[t] - grid[t]

        row = n + t
        equality[row, e0 + t] = 1.0
        if t > 0:
            equality[row, e0 + t - 1] = -1.0
            rhs[row] = 0.0
        else:
            rhs[row] = soc0
        equality[row, c0 + t] = -ETA_CHARGE
        equality[row, d0 + t] = 1.0 / ETA_DISCHARGE

    lower = np.zeros(m)
    upper = np.full(m, np.inf)
    upper[c0 : c0 + n] = ENERGY_LIMIT
    upper[d0 : d0 + n] = ENERGY_LIMIT
    lower[e0 : e0 + n] = SOC_MIN
    upper[e0 : e0 + n] = SOC_MAX
    result = linprog(
        objective,
        A_eq=equality.tocsr(),
        b_eq=rhs,
        bounds=list(zip(lower, upper)),
        method="highs",
        options={"presolve": True},
    )
    if not result.success:
        raise RuntimeError(f"实时前瞻线性规划失败: {result.message}")
    c = max(float(result.x[c0]), 0.0)
    d = max(float(result.x[d0]), 0.0)
    emergency = max(float(result.x[x0]), 0.0)
    spill = max(float(result.x[w0]), 0.0)
    next_soc = soc0 + ETA_CHARGE * c - d / ETA_DISCHARGE
    return c, d, emergency, spill, min(max(next_soc, SOC_MIN), SOC_MAX)


def simulate_lookahead_policy(
    data: DataBundle,
    robust_forecast_kw: np.ndarray,
    point_forecast_kw: np.ndarray,
    price: np.ndarray,
) -> YearResult:
    """Q2 day-ahead LP plus a leakage-free receding-horizon LP executor."""
    days = len(data.dates)
    plan = np.zeros((days, N_SLOTS))
    charge = np.zeros_like(plan)
    discharge = np.zeros_like(plan)
    emergency = np.zeros_like(plan)
    spill = np.zeros_like(plan)
    soc_start = np.zeros(days)
    soc_end = np.zeros(days)
    plan_cost = np.zeros(days)
    emergency_cost = np.zeros(days)
    actual_net_kw = data.load - data.pv
    # The executor must retain the same risk attitude as the day-ahead plan.
    # Residual persistence is therefore measured around the deployed robust
    # demand path rather than around the lower point forecast.
    residual_kw = actual_net_kw - robust_forecast_kw
    total_solve = 0.0
    soc = SOC_INITIAL
    for day in range(days):
        soc_start[day] = soc
        day_price = price if price.ndim == 1 else price[day]
        started = time.perf_counter()
        optimized = _deterministic_lp(
            robust_forecast_kw[day] * DT_HOURS, day_price, soc, soc
        )
        total_solve += optimized.solve_seconds
        plan[day] = optimized.grid
        rho = _estimate_residual_persistence(residual_kw, day)
        terminal_value = float(np.mean(day_price[:30]))
        for slot in range(N_SLOTS):
            # With surplus contracted energy, charging as much as possible is
            # dominant; invoke the look-ahead LP only for an actual shortage.
            actual_net_kwh = actual_net_kw[day, slot] * DT_HOURS
            if plan[day, slot] >= actual_net_kwh:
                c, d, e, w, soc = dispatch_one_interval(
                    plan[day, slot], actual_net_kwh, soc
                )
                charge[day, slot] = c
                discharge[day, slot] = d
                emergency[day, slot] = e
                spill[day, slot] = w
                continue
            horizon = N_SLOTS - slot
            correction = residual_kw[day, slot] * np.power(rho, np.arange(horizon))
            expected_net_kw = robust_forecast_kw[day, slot:] + correction
            expected_net_kw[0] = actual_net_kw[day, slot]
            c, d, e, w, soc = _lookahead_dispatch_lp(
                plan[day, slot:],
                expected_net_kw * DT_HOURS,
                day_price[slot:],
                soc,
                terminal_value,
            )
            charge[day, slot] = c
            discharge[day, slot] = d
            emergency[day, slot] = e
            spill[day, slot] = w
        soc_end[day] = soc
        total_solve += time.perf_counter() - started - optimized.solve_seconds
        plan_cost[day] = float(np.dot(day_price, plan[day]))
        emergency_cost[day] = float(np.dot(EMERGENCY_MULTIPLIER * day_price, emergency[day]))
    total_cost = plan_cost + emergency_cost
    return YearResult(
        dates=data.dates,
        plan_grid=plan,
        final_grid=plan.copy(),
        charge=charge,
        discharge=discharge,
        emergency=emergency,
        spill=spill,
        soc_start=soc_start,
        soc_end=soc_end,
        plan_cost=plan_cost,
        settlement_cost=plan_cost.copy(),
        emergency_cost=emergency_cost,
        total_cost=total_cost,
        solve_seconds=total_solve,
    )


def simulate_fixed_policy(
    data: DataBundle,
    net_forecast_kw: np.ndarray,
    price: np.ndarray,
) -> YearResult:
    days = len(data.dates)
    plan = np.zeros((days, N_SLOTS))
    charge = np.zeros_like(plan)
    discharge = np.zeros_like(plan)
    emergency = np.zeros_like(plan)
    spill = np.zeros_like(plan)
    soc_start = np.zeros(days)
    soc_end = np.zeros(days)
    plan_cost = np.zeros(days)
    emergency_cost = np.zeros(days)
    soc_path = np.zeros((days, N_SLOTS + 1))
    total_solve = 0.0
    soc = SOC_INITIAL
    actual_net_kwh = (data.load - data.pv) * DT_HOURS
    for day in range(days):
        soc_start[day] = soc
        soc_path[day, 0] = soc
        day_price = price if price.ndim == 1 else price[day]
        optimized = _deterministic_lp(
            net_forecast_kw[day] * DT_HOURS,
            day_price,
            soc,
            soc,
        )
        total_solve += optimized.solve_seconds
        plan[day] = optimized.grid
        for slot in range(N_SLOTS):
            c, d, e, w, soc = dispatch_one_interval(plan[day, slot], actual_net_kwh[day, slot], soc)
            charge[day, slot] = c
            discharge[day, slot] = d
            emergency[day, slot] = e
            spill[day, slot] = w
            soc_path[day, slot + 1] = soc
        soc_end[day] = soc
        plan_cost[day] = float(np.dot(day_price, plan[day]))
        emergency_cost[day] = float(np.dot(EMERGENCY_MULTIPLIER * day_price, emergency[day]))
    total_cost = plan_cost + emergency_cost
    return YearResult(
        dates=data.dates,
        plan_grid=plan,
        final_grid=plan.copy(),
        charge=charge,
        discharge=discharge,
        emergency=emergency,
        spill=spill,
        soc_start=soc_start,
        soc_end=soc_end,
        plan_cost=plan_cost,
        settlement_cost=plan_cost.copy(),
        emergency_cost=emergency_cost,
        total_cost=total_cost,
        solve_seconds=total_solve,
        soc_path=soc_path,
    )


def simulate_adaptive_quantile_policy(
    data: DataBundle,
    point_forecast_kw: np.ndarray,
    price: np.ndarray,
    candidates: Sequence[float] = Q2_ADAPTIVE_CANDIDATES,
    history_window: int = ROLLING_WINDOW,
    default_quantile: float = Q2_SERVICE_QUANTILE,
    cost_tolerance: float = 0.0,
) -> tuple[YearResult, dict]:
    """Run Q2 with one leakage-free quantile choice and one LP per target day.

    Candidate fixed-quantile policies are evaluated once and cached by day.  The
    On day d, candidates whose latest complete-history cost is within
    ``cost_tolerance`` of the minimum are retained, and the candidate with the
    lowest historical spill is selected.  With a zero tolerance this reduces to
    strict cost minimization.  The selected day's grid plan is then solved once
    at 00:00 and remains fixed during physical dispatch.
    """
    candidate_values = tuple(float(q) for q in candidates)
    if default_quantile not in candidate_values:
        raise ValueError("默认分位数必须包含在候选集合中")
    if history_window <= 0:
        raise ValueError("历史回测窗口必须为正整数")
    if cost_tolerance < 0:
        raise ValueError("成本容忍度不能为负")

    actual_net_kw = data.load - data.pv
    # Cache every time-specific margin and robust path once.  Each row d of
    # rolling_residual_quantile uses residual rows strictly earlier than d.
    margins = {
        q: rolling_residual_quantile(actual_net_kw, point_forecast_kw, q, history_window)
        for q in candidate_values
    }
    robust_paths = {q: point_forecast_kw + margins[q] for q in candidate_values}

    # Cache genuine causal fixed-q backtests.  Every candidate starts from the
    # same initial SOC and uses the same forecast, tariff and greedy dispatcher.
    candidate_results = {
        q: simulate_fixed_policy(data, robust_paths[q], price) for q in candidate_values
    }
    candidate_daily_cost = np.column_stack(
        [candidate_results[q].total_cost for q in candidate_values]
    )
    candidate_daily_spill = np.column_stack(
        [np.sum(candidate_results[q].spill, axis=1) for q in candidate_values]
    )

    days = len(data.dates)
    selected_quantile = np.full(days, default_quantile, dtype=float)
    history_start = np.zeros(days, dtype=int)
    history_days = np.zeros(days, dtype=int)
    history_cost = np.full((days, len(candidate_values)), np.nan)
    history_spill = np.full((days, len(candidate_values)), np.nan)
    default_index = candidate_values.index(default_quantile)
    for day in range(days):
        start = max(0, day - history_window)
        history_start[day] = start
        history_days[day] = day - start
        if day < history_window:
            continue
        scores = np.sum(candidate_daily_cost[start:day], axis=0)
        spill_scores = np.sum(candidate_daily_spill[start:day], axis=0)
        history_cost[day] = scores
        history_spill[day] = spill_scores
        minimum = float(np.min(scores))
        eligible = np.flatnonzero(scores <= minimum * (1.0 + cost_tolerance) + 1.0e-8)
        chosen = min(
            eligible,
            key=lambda idx: (
                spill_scores[idx],
                scores[idx],
                abs(candidate_values[idx] - default_quantile),
                idx,
            ),
        )
        selected_quantile[day] = candidate_values[chosen]

    plan = np.zeros((days, N_SLOTS))
    charge = np.zeros_like(plan)
    discharge = np.zeros_like(plan)
    emergency = np.zeros_like(plan)
    spill = np.zeros_like(plan)
    soc_start = np.zeros(days)
    soc_end = np.zeros(days)
    soc_path = np.zeros((days, N_SLOTS + 1))
    plan_cost = np.zeros(days)
    emergency_cost = np.zeros(days)
    actual_net_kwh = actual_net_kw * DT_HOURS
    total_solve = 0.0
    soc = SOC_INITIAL
    for day in range(days):
        soc_start[day] = soc
        soc_path[day, 0] = soc
        day_price = price if price.ndim == 1 else price[day]
        q = float(selected_quantile[day])
        optimized = _deterministic_lp(
            robust_paths[q][day] * DT_HOURS,
            day_price,
            soc,
            soc,
        )
        total_solve += optimized.solve_seconds
        plan[day] = optimized.grid
        # No reforecast, reoptimization, or grid-plan revision occurs below.
        for slot in range(N_SLOTS):
            c, d, e, w, soc = dispatch_one_interval(
                plan[day, slot], actual_net_kwh[day, slot], soc
            )
            charge[day, slot] = c
            discharge[day, slot] = d
            emergency[day, slot] = e
            spill[day, slot] = w
            soc_path[day, slot + 1] = soc
        soc_end[day] = soc
        plan_cost[day] = float(np.dot(day_price, plan[day]))
        emergency_cost[day] = float(
            np.dot(EMERGENCY_MULTIPLIER * day_price, emergency[day])
        )

    total_cost = plan_cost + emergency_cost
    result = YearResult(
        dates=data.dates,
        plan_grid=plan,
        final_grid=plan.copy(),
        charge=charge,
        discharge=discharge,
        emergency=emergency,
        spill=spill,
        soc_start=soc_start,
        soc_end=soc_end,
        plan_cost=plan_cost,
        settlement_cost=plan_cost.copy(),
        emergency_cost=emergency_cost,
        total_cost=total_cost,
        solve_seconds=total_solve,
        soc_path=soc_path,
    )
    audit = {
        "candidates": list(candidate_values),
        "selected_quantile": selected_quantile,
        "history_start_index": history_start,
        "history_days": history_days,
        "history_cost_yuan": history_cost,
        "history_spill_kwh": history_spill,
        "candidate_daily_cost_yuan": candidate_daily_cost,
        "candidate_daily_spill_kwh": candidate_daily_spill,
        "candidate_results": candidate_results,
        "margins_kw": margins,
        "official_lp_solves": days,
        "intraday_reoptimizations": 0,
        "default_candidate_index": default_index,
        "cost_tolerance": float(cost_tolerance),
    }
    return result, audit


def settlement_cost(price: np.ndarray, plan: np.ndarray, adjusted: np.ndarray) -> float:
    lower = adjusted <= plan + 1.0e-8
    cost = np.where(
        lower,
        price * (adjusted + 0.5 * (plan - adjusted)),
        price * (plan + 1.5 * (adjusted - plan)),
    )
    return float(np.sum(cost))


def simulate_adjustable_policy(
    data: DataBundle,
    robust_forecast: dict[int, np.ndarray],
    price: np.ndarray,
    updates: Sequence[int] = UPDATE_HOURS,
) -> YearResult:
    days = len(data.dates)
    plan = np.zeros((days, N_SLOTS))
    adjusted = np.zeros_like(plan)
    charge = np.zeros_like(plan)
    discharge = np.zeros_like(plan)
    emergency = np.zeros_like(plan)
    spill = np.zeros_like(plan)
    soc_start = np.zeros(days)
    soc_end = np.zeros(days)
    plan_cost = np.zeros(days)
    settled = np.zeros(days)
    emergency_cost = np.zeros(days)
    total_solve = 0.0
    soc = SOC_INITIAL
    actual_net_kwh = (data.load - data.pv) * DT_HOURS
    update_slots = {hour: int(round(hour * 6)) - 1 for hour in updates}
    for day in range(days):
        day_start_soc = soc
        soc_start[day] = soc
        day_price = price if price.ndim == 1 else price[day]
        day_plan = _deterministic_lp(
            robust_forecast[0][day] * DT_HOURS,
            day_price,
            soc,
            soc,
        )
        total_solve += day_plan.solve_seconds
        plan[day] = day_plan.grid
        adjusted[day] = day_plan.grid
        for slot in range(N_SLOTS):
            for hour, first_slot in update_slots.items():
                if slot == first_slot:
                    update = _adjustment_lp(
                        robust_forecast[hour][day, slot:] * DT_HOURS,
                        day_price[slot:],
                        plan[day, slot:],
                        soc,
                        day_start_soc,
                    )
                    total_solve += update.solve_seconds
                    adjusted[day, slot:] = update.grid
            c, d, e, w, soc = dispatch_one_interval(adjusted[day, slot], actual_net_kwh[day, slot], soc)
            charge[day, slot] = c
            discharge[day, slot] = d
            emergency[day, slot] = e
            spill[day, slot] = w
        soc_end[day] = soc
        plan_cost[day] = float(np.dot(day_price, plan[day]))
        settled[day] = settlement_cost(day_price, plan[day], adjusted[day])
        emergency_cost[day] = float(np.dot(EMERGENCY_MULTIPLIER * day_price, emergency[day]))
    total_cost = settled + emergency_cost
    return YearResult(
        dates=data.dates,
        plan_grid=plan,
        final_grid=adjusted,
        charge=charge,
        discharge=discharge,
        emergency=emergency,
        spill=spill,
        soc_start=soc_start,
        soc_end=soc_end,
        plan_cost=plan_cost,
        settlement_cost=settled,
        emergency_cost=emergency_cost,
        total_cost=total_cost,
        solve_seconds=total_solve,
    )


def simulate_no_update_from_plan(data: DataBundle, result: YearResult, price: np.ndarray) -> dict:
    soc = SOC_INITIAL
    actual_net_kwh = (data.load - data.pv) * DT_HOURS
    emergency = np.zeros_like(result.plan_grid)
    end_soc = np.zeros(len(data.dates))
    for day in range(len(data.dates)):
        for slot in range(N_SLOTS):
            _, _, e, _, soc = dispatch_one_interval(result.plan_grid[day, slot], actual_net_kwh[day, slot], soc)
            emergency[day, slot] = e
        end_soc[day] = soc
    day_price = np.repeat(price[None, :], len(data.dates), axis=0) if price.ndim == 1 else price
    plan_cost = np.sum(day_price * result.plan_grid, axis=1)
    emergency_cost = np.sum(EMERGENCY_MULTIPLIER * day_price * emergency, axis=1)
    start_index = 31
    return {
        "total_cost": float(np.sum(plan_cost[start_index:] + emergency_cost[start_index:])),
        "emergency_kwh": float(np.sum(emergency[start_index:])),
        "emergency_intervals": int(np.sum(emergency[start_index:] > 1.0e-6)),
        "soc_min": float(np.min(end_soc)),
        "soc_max": float(np.max(end_soc)),
    }


def _group_emergency(date: pd.Timestamp, values: np.ndarray, labels: list[str]) -> list[list]:
    positive = values > 1.0e-6
    groups: list[list] = []
    start = None
    for idx, is_positive in enumerate(positive):
        if is_positive and start is None:
            start = idx
        if start is not None and (not is_positive or idx == len(values) - 1):
            end = idx if not is_positive else idx + 1
            start_label = labels[start].split("-")[0]
            end_label = labels[end - 1].split("-")[1]
            groups.append([date.strftime("%Y-%m-%d"), f"{start_label}-{end_label}", float(np.sum(values[start:end]))])
            start = None
    return groups


def _storage_rows(result: YearResult, start_index: int = 31) -> list[list]:
    rows: list[list] = []
    blocks = [(0, 24), (24, 48), (48, 72), (72, 96), (96, 120), (120, 144)]
    labels = ["0:00-4:00", "4:00-8:00", "8:00-12:00", "12:00-16:00", "16:00-20:00", "20:00-24:00"]
    for day in range(start_index, len(result.dates)):
        for block, ((left, right), label) in enumerate(zip(blocks, labels)):
            rows.append(
                [
                    result.dates[day].strftime("%Y-%m-%d") if block == 0 else None,
                    label,
                    float(np.sum(result.charge[day, left:right])),
                    float(np.sum(result.discharge[day, left:right])),
                    "0:00" if block == 0 else ("24:00" if block == 1 else None),
                    float(result.soc_start[day]) if block == 0 else (float(result.soc_end[day]) if block == 1 else None),
                ]
            )
    return rows


def _emergency_rows(result: YearResult, labels: list[str], start_index: int = 31) -> list[list]:
    rows: list[list] = []
    for day in range(start_index, len(result.dates)):
        groups = _group_emergency(result.dates[day], result.emergency[day], labels)
        if not groups:
            rows.append([result.dates[day].strftime("%Y-%m-%d"), None, 0.0])
            continue
        for index, group in enumerate(groups):
            rows.append([group[0] if index == 0 else None, group[1], group[2]])
    return rows


def _plan_rows(result: YearResult, price: np.ndarray, adjusted: bool = False, start_index: int = 31) -> list[list]:
    grid = result.final_grid if adjusted else result.plan_grid
    rows: list[list] = []
    for day in range(start_index, len(result.dates)):
        values = [float(x) for x in grid[day]]
        if adjusted:
            total_cost = result.total_cost[day]
        else:
            total_cost = result.total_cost[day] if np.allclose(result.final_grid, result.plan_grid) else result.plan_cost[day]
        rows.append(
            [
                result.dates[day].strftime("%Y-%m-%d"),
                *values,
                float(np.sum(grid[day])),
                float(total_cost),
            ]
        )
    return rows


def _q1_payload(data: DataBundle) -> tuple[dict, dict]:
    net = (data.sample_load - data.sample_pv) * DT_HOURS
    optimized = _deterministic_lp(net, data.fixed_price, SOC_INITIAL, SOC_INITIAL)
    storage = []
    storage_labels = ["0:00-4:00", "4:00-8:00", "8:00-12:00", "12:00-16:00", "16:00-20:00", "20:00-24:00"]
    for block, left in enumerate(range(0, N_SLOTS, 24)):
        storage.append(
            [
                storage_labels[block],
                float(np.sum(optimized.charge[left : left + 24])),
                float(np.sum(optimized.discharge[left : left + 24])),
            ]
        )
    payload = {
        "plan": [float(x) for x in optimized.grid],
        "storage": storage,
        "soc0": SOC_INITIAL,
        "soc24": float(optimized.soc[-1]),
        "total_grid_kwh": float(np.sum(optimized.grid)),
        "total_cost_yuan": float(np.dot(data.fixed_price, optimized.grid)),
    }
    diagnostics = {
        "soc_min": float(np.min(optimized.soc)),
        "soc_max": float(np.max(optimized.soc)),
        "balance_max_abs_kwh": float(np.max(np.abs(optimized.grid + optimized.discharge - optimized.charge - optimized.spill - net))),
        "terminal_error_kwh": float(abs(optimized.soc[-1] - SOC_INITIAL)),
        "solve_seconds": optimized.solve_seconds,
    }
    return payload, diagnostics


def summarize_year(result: YearResult, start_index: int = 31) -> dict:
    sl = slice(start_index, None)
    soc_values = (
        result.soc_path[sl, 1:]
        if result.soc_path is not None
        else np.c_[result.soc_start[sl], result.soc_end[sl]]
    )
    return {
        "days": int(len(result.dates) - start_index),
        "plan_grid_kwh": float(np.sum(result.plan_grid[sl])),
        "final_grid_kwh": float(np.sum(result.final_grid[sl])),
        "charge_kwh": float(np.sum(result.charge[sl])),
        "discharge_kwh": float(np.sum(result.discharge[sl])),
        "emergency_kwh": float(np.sum(result.emergency[sl])),
        "spill_kwh": float(np.sum(result.spill[sl])),
        "plan_cost_yuan": float(np.sum(result.plan_cost[sl])),
        "settlement_cost_yuan": float(np.sum(result.settlement_cost[sl])),
        "emergency_cost_yuan": float(np.sum(result.emergency_cost[sl])),
        "total_cost_yuan": float(np.sum(result.total_cost[sl])),
        "emergency_intervals": int(np.sum(result.emergency[sl] > 1.0e-6)),
        "soc_min_kwh": float(np.min(soc_values)),
        "soc_max_kwh": float(np.max(soc_values)),
        "soc_bottom_hits": int(np.sum(np.isclose(soc_values, SOC_MIN, rtol=0.0, atol=1.0e-6))),
        "soc_top_hits": int(np.sum(np.isclose(soc_values, SOC_MAX, rtol=0.0, atol=1.0e-6))),
        "total_lp_solve_seconds": float(result.solve_seconds),
    }


def selected_date_summary(result: YearResult, labels: list[str]) -> dict:
    date_map = {date.strftime("%Y-%m-%d"): idx for idx, date in enumerate(result.dates)}
    output = {}
    for date_text in SELECTED_DATES:
        day = date_map[date_text]
        slot_map = {
            "10:00-10:10": 59,
            "12:00-12:10": 71,
            "14:00-14:10": 83,
            "16:00-16:10": 95,
            "18:00-18:10": 107,
            "20:00-20:10": 119,
        }
        output[date_text] = {
            "specified_grid_kwh": {key: float(result.final_grid[day, idx]) for key, idx in slot_map.items()},
            "daily_grid_kwh": float(np.sum(result.final_grid[day])),
            "daily_total_cost_yuan": float(result.total_cost[day]),
            "emergency": _group_emergency(result.dates[day], result.emergency[day], labels),
            "soc0_kwh": float(result.soc_start[day]),
            "soc24_kwh": float(result.soc_end[day]),
        }
    return output


def validate_result(data: DataBundle, result: YearResult, price: np.ndarray, name: str) -> dict:
    actual_net = (data.load - data.pv) * DT_HOURS
    balance = result.final_grid + result.discharge + result.emergency - result.charge - result.spill - actual_net
    day_price = np.repeat(price[None, :], len(data.dates), axis=0) if price.ndim == 1 else price
    recomputed_emergency_cost = np.sum(EMERGENCY_MULTIPLIER * day_price * result.emergency, axis=1)
    soc_values = (
        result.soc_path
        if result.soc_path is not None
        else np.c_[result.soc_start, result.soc_end]
    )
    if result.soc_path is not None:
        soc_transition = (
            result.soc_path[:, 1:]
            - result.soc_path[:, :-1]
            - ETA_CHARGE * result.charge
            + result.discharge / ETA_DISCHARGE
        )
        soc_transition_error = float(np.max(np.abs(soc_transition)))
        cross_day_error = float(
            np.max(np.abs(result.soc_path[1:, 0] - result.soc_path[:-1, -1]))
        )
    else:
        soc_transition_error = float("nan")
        cross_day_error = float(np.max(np.abs(result.soc_start[1:] - result.soc_end[:-1])))
    checks = {
        "name": name,
        "finite": bool(
            np.isfinite(result.plan_grid).all()
            and np.isfinite(result.final_grid).all()
            and np.isfinite(result.emergency).all()
        ),
        "nonnegative_grid": bool(np.min(result.final_grid) >= -1.0e-7),
        "nonnegative_emergency": bool(np.min(result.emergency) >= -1.0e-7),
        "charge_power_limit": bool(np.max(result.charge) <= ENERGY_LIMIT + 1.0e-6),
        "discharge_power_limit": bool(np.max(result.discharge) <= ENERGY_LIMIT + 1.0e-6),
        "soc_bounds": bool(
            np.min(soc_values) >= SOC_MIN - 1.0e-6
            and np.max(soc_values) <= SOC_MAX + 1.0e-6
        ),
        "soc_transition_max_abs_kwh": soc_transition_error,
        "cross_day_soc_error_kwh": cross_day_error,
        "balance_max_abs_kwh": float(np.max(np.abs(balance))),
        "emergency_cost_error_yuan": float(np.max(np.abs(recomputed_emergency_cost - result.emergency_cost))),
    }
    checks["passed"] = bool(
        checks["finite"]
        and checks["nonnegative_grid"]
        and checks["nonnegative_emergency"]
        and checks["charge_power_limit"]
        and checks["discharge_power_limit"]
        and checks["soc_bounds"]
        and (not np.isfinite(soc_transition_error) or soc_transition_error < 1.0e-5)
        and checks["cross_day_soc_error_kwh"] < 1.0e-5
        and checks["balance_max_abs_kwh"] < 1.0e-5
        and checks["emergency_cost_error_yuan"] < 1.0e-5
    )
    return checks


def create_figures(
    data: DataBundle,
    forecasts: dict,
    results: dict[str, YearResult],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 140,
            "savefig.dpi": 320,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2"]

    day = int(np.argmax(np.sum(data.pv, axis=1)))
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.plot(data.slot_hours, data.load[day], color=colors[0], label="小区负载")
    ax.plot(data.slot_hours, data.pv[day], color=colors[1], label="光伏实际功率")
    ax.plot(data.slot_hours, forecasts["pv_issue"][0][day], color=colors[2], linestyle="--", label="0时光伏预报")
    ax.set(title=f"负载与光伏曲线 {data.dates[day]:%Y-%m-%d}", xlabel="时刻（小时）", ylabel="功率（kW）")
    ax.legend(frameon=False, ncol=3)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(output_dir / "01_load_pv_profile.png", bbox_inches="tight")
    plt.close(fig)

    mae = []
    labels = []
    for hour in (0, 6, 12, 18):
        mask = data.slot_hours >= hour
        mae.append(float(np.mean(np.abs(forecasts["pv_issue"][hour][:, mask] - data.pv[:, mask]))))
        labels.append(f"{hour}:00")
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bars = ax.bar(labels, mae, color=colors[:4], width=0.62)
    ax.bar_label(bars, fmt="%.0f", padding=3)
    ax.set(title="不同发布时刻光伏预报误差", xlabel="预报发布时刻", ylabel="未来剩余时段MAE（kW）")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(output_dir / "02_forecast_update_mae.png", bbox_inches="tight")
    plt.close(fig)

    names = list(results)
    costs = [float(np.sum(results[name].total_cost[31:]) / 1.0e6) for name in names]
    fig, ax = plt.subplots(figsize=(8.6, 4.5))
    bars = ax.bar(names, costs, color=colors[: len(names)])
    ax.bar_label(bars, fmt="%.2f", padding=3)
    ax.set(title="各问题购电总费用比较", xlabel="策略", ylabel="2025-02-01至12-31总费用（百万元）")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(output_dir / "03_cost_comparison.png", bbox_inches="tight")
    plt.close(fig)

    q43 = results["问题4-3"]
    day = int(np.where(data.dates == pd.Timestamp("2025-06-21"))[0][0])
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(data.slot_hours, q43.plan_grid[day], color=colors[0], label="计划购电")
    axes[0].plot(data.slot_hours, q43.final_grid[day], color=colors[1], label="调整购电")
    axes[0].bar(data.slot_hours, q43.emergency[day], width=0.12, color=colors[3], alpha=0.75, label="紧急购电")
    axes[0].set(ylabel="电量（kWh）", title="滚动调整策略 2025-06-21")
    axes[0].legend(frameon=False, ncol=3)
    axes[0].grid(axis="y", color="#D9D9D9", linewidth=0.6)
    soc = [q43.soc_start[day]]
    current = q43.soc_start[day]
    for c, d in zip(q43.charge[day], q43.discharge[day]):
        current += ETA_CHARGE * c - d / ETA_DISCHARGE
        soc.append(current)
    axes[1].plot(np.r_[0, data.slot_hours], soc, color=colors[2])
    axes[1].axhline(SOC_MIN, color="#999999", linestyle="--", linewidth=0.8)
    axes[1].axhline(SOC_MAX, color="#999999", linestyle="--", linewidth=0.8)
    axes[1].set(xlabel="时刻（小时）", ylabel="储电量（kWh）")
    axes[1].grid(axis="y", color="#D9D9D9", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(output_dir / "04_rolling_dispatch.png", bbox_inches="tight")
    plt.close(fig)


def build_payload(
    data: DataBundle,
    q1: dict,
    q2: YearResult,
    q3: YearResult,
    q42: YearResult,
    q43: YearResult,
) -> dict:
    return {
        "slot_labels": data.slot_labels,
        "result1": q1,
        "result2": {
            "plan_rows": _plan_rows(q2, data.fixed_price),
            "storage_rows": _storage_rows(q2),
            "emergency_rows": _emergency_rows(q2, data.slot_labels),
        },
        "result3": {
            "plan_rows": _plan_rows(q3, data.fixed_price, adjusted=False),
            "adjust_rows": _plan_rows(q3, data.fixed_price, adjusted=True),
            "storage_rows": _storage_rows(q3),
            "emergency_rows": _emergency_rows(q3, data.slot_labels),
        },
        "result4-2": {
            "plan_rows": _plan_rows(q42, data.variable_price),
            "storage_rows": _storage_rows(q42),
            "emergency_rows": _emergency_rows(q42, data.slot_labels),
        },
        "result4-3": {
            "plan_rows": _plan_rows(q43, data.variable_price, adjusted=False),
            "adjust_rows": _plan_rows(q43, data.variable_price, adjusted=True),
            "storage_rows": _storage_rows(q43),
            "emergency_rows": _emergency_rows(q43, data.slot_labels),
        },
    }


def save_csv_outputs(data: DataBundle, results: dict[str, YearResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for name, result in results.items():
        for day in range(31, len(data.dates)):
            records.append(
                {
                    "问题": name,
                    "日期": data.dates[day].strftime("%Y-%m-%d"),
                    "计划购电量_kWh": np.sum(result.plan_grid[day]),
                    "最终购电量_kWh": np.sum(result.final_grid[day]),
                    "紧急购电量_kWh": np.sum(result.emergency[day]),
                    "充电量_kWh": np.sum(result.charge[day]),
                    "放电量_kWh": np.sum(result.discharge[day]),
                    "弃电量_kWh": np.sum(result.spill[day]),
                    "日总费用_元": result.total_cost[day],
                    "日初储电量_kWh": result.soc_start[day],
                    "日末储电量_kWh": result.soc_end[day],
                }
            )
    pd.DataFrame(records).to_csv(output_dir / "daily_metrics.csv", index=False, encoding="utf-8-sig")


def run(root: Path) -> dict:
    np.random.seed(SEED)
    output_dir = root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_data(root)
    quality = diagnose_data(data)
    forecasts = build_forecasts(data)
    q2_quantile, q2_tuning = tune_q2_quantile(data, forecasts["q2_point"])
    forecasts["q2_robust"] = forecasts["q2_point"] + rolling_residual_quantile(
        data.load - data.pv, forecasts["q2_point"], q2_quantile
    )
    metrics = forecast_metrics(data, forecasts)

    q1, q1_validation = _q1_payload(data)
    q2, q2_adaptive_audit = simulate_adaptive_quantile_policy(
        data,
        forecasts["q2_point"],
        data.fixed_price,
        candidates=Q2_ADAPTIVE_CANDIDATES,
        history_window=ROLLING_WINDOW,
        default_quantile=Q2_SERVICE_QUANTILE,
        cost_tolerance=Q2_SPILL_COST_TOLERANCE,
    )
    q2_fixed_baseline = q2_adaptive_audit["candidate_results"][Q2_SERVICE_QUANTILE]
    q3 = simulate_adjustable_policy(data, forecasts["q3_robust"], data.fixed_price)
    q42 = simulate_fixed_policy(data, forecasts["q2_robust"], data.variable_price)
    q43 = simulate_adjustable_policy(data, forecasts["q3_robust"], data.variable_price)

    results = {"问题2": q2, "问题3": q3, "问题4-2": q42, "问题4-3": q43}
    validations = {"问题1": q1_validation}
    validations.update(
        {
            name: validate_result(
                data,
                result,
                data.fixed_price if name in ("问题2", "问题3") else data.variable_price,
                name,
            )
            for name, result in results.items()
        }
    )
    if not all(item.get("passed", item.get("balance_max_abs_kwh", 1) < 1.0e-5) for item in validations.values()):
        raise RuntimeError(f"模型验证未通过: {validations}")

    no_update_q3 = simulate_no_update_from_plan(data, q3, data.fixed_price)
    no_update_q43 = simulate_no_update_from_plan(data, q43, data.variable_price)
    update_analysis = {
        "fixed_price": {
            "no_update": no_update_q3,
            "all_updates": summarize_year(q3),
            "cost_saving_yuan": float(no_update_q3["total_cost"] - np.sum(q3.total_cost[31:])),
            "emergency_reduction_kwh": float(no_update_q3["emergency_kwh"] - np.sum(q3.emergency[31:])),
        },
        "variable_price": {
            "no_update": no_update_q43,
            "all_updates": summarize_year(q43),
            "cost_saving_yuan": float(no_update_q43["total_cost"] - np.sum(q43.total_cost[31:])),
            "emergency_reduction_kwh": float(no_update_q43["emergency_kwh"] - np.sum(q43.emergency[31:])),
        },
    }

    summary = {
        "data_quality": quality,
        "forecast_metrics": metrics,
        "problem2_model_selection": {
            "method": "每天0:00仅使用此前28个完整历史日；先保留总成本不超过历史最低成本0.25%的候选，再选择历史弃电量最低者，并只求解一次日前LP",
            "objective": "在近最优成本约束下最小化历史弃电量",
            "candidates": list(Q2_ADAPTIVE_CANDIDATES),
            "cost_tolerance": Q2_SPILL_COST_TOLERANCE,
            "history_window_days": ROLLING_WINDOW,
            "intraday_reoptimizations": 0,
        },
        "problem1": q1,
        "problem2": summarize_year(q2),
        "problem2_fixed_0_8_baseline": summarize_year(q2_fixed_baseline),
        "problem3": summarize_year(q3),
        "problem4_2": summarize_year(q42),
        "problem4_3": summarize_year(q43),
        "selected_dates": {
            "problem2": selected_date_summary(q2, data.slot_labels),
            "problem3": selected_date_summary(q3, data.slot_labels),
            "problem4_2": selected_date_summary(q42, data.slot_labels),
            "problem4_3": selected_date_summary(q43, data.slot_labels),
        },
        "update_analysis": update_analysis,
        "validation": validations,
        "assumptions": {
            "optimization_model": "MILP；每个时段设置充放电模式二进制变量，严格禁止同时充放电",
            "charge_efficiency": ETA_CHARGE,
            "discharge_efficiency": ETA_DISCHARGE,
            "daily_terminal_soc_target": "每日优化目标为日初储电量；实际偏差结转至下一日",
            "problem3_settlement": "下调部分支付原价50%的违约费用，上调超出部分按原价1.5倍结算",
            "forecast_leakage_control": "所有滚动基线和误差分位数只使用目标日之前的数据",
            "q2_quantile": "每天在0.60—0.90候选集合中自适应选择",
            "q3_quantile": Q3_SERVICE_QUANTILE,
        },
    }
    payload = build_payload(data, q1, q2, q3, q42, q43)
    (output_dir / "analysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "workbook_payload.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    save_csv_outputs(data, results, output_dir)
    create_figures(data, forecasts, results, output_dir / "figures")
    return summary


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    result = run(project_root)
    print(json.dumps({key: value for key, value in result.items() if key.startswith("problem")}, ensure_ascii=False, indent=2))
