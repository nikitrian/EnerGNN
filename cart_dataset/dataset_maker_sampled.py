"""Generate detailed patient-level CART datasets with sampled facility masks.

Solve an unrestricted detailed reference, propose a diverse mask pool, score
each fixed mask with the shared reduced detailed solver, and stratify the
completed results for training without enumerating every binary mask.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    gp = GRB = None
import numpy as np


from .config import FACILITY_UTIL_CAP, PER_DAY_CAP, TIME_HORIZON, TMFE
from . import dataset_maker as dm
from .candidate_scoring import (
    IncompleteCandidatePool, _finite_attr, is_complete, label_candidate_results,
    resolve_formulation, solve_mask,
)
from .demand import draw_inc_for_instance
from .facility_sampling import _expand_facilities, _static_config
from .model_cart_detailed import build_detailed_data_from_aggregate
from .patient_dats import load_static_from_config, write_patient_dat

DEMAND_DISTRIBUTIONS = ("poisson", "seasonal_poisson", "bursty")
SPLITS = ("train", "val", "test")
EPS = 1e-8


class IncumbentCollector:
    def __init__(self, z_vars: Sequence[gp.Var]) -> None:
        self.z_vars = list(z_vars)
        self.rows: list[tuple[np.ndarray, float]] = []
        self.seen: set[tuple[int, ...]] = set()

    def __call__(self, model: gp.Model, where: int) -> None:
        if where != GRB.Callback.MIPSOL:
            return
        vals = model.cbGetSolution(self.z_vars)
        mask = np.asarray([1 if float(v) > 0.5 else 0 for v in vals], dtype=np.int8)
        key = _mask_key(mask)
        if key in self.seen:
            return
        self.seen.add(key)
        self.rows.append((mask, float(model.cbGet(GRB.Callback.MIPSOL_OBJ))))


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row))
            fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    temporary.replace(path)


def _write_json(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as fh:
        json.dump(row, fh, indent=2)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    temporary.replace(path)


def _append_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _load_mask_checkpoints(path: Path) -> dict[tuple[int, ...], dict]:
    results = {}
    if not path.exists():
        return results
    # A killed append can leave a partial final line; retain every finished solve.
    with path.open("r+b") as fh:
        while True:
            start = fh.tell()
            line = fh.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                fh.truncate(start)
                break
            row = json.loads(line)
            results[_mask_key(row["mask"])] = row
    return results


def _mask_key(mask: Sequence[int]) -> tuple[int, ...]:
    return tuple(int(v) for v in np.asarray(mask, dtype=np.int8).reshape(-1).tolist())


def _mask_to_dict(mask: Sequence[int], facilities: Sequence[str]) -> Dict[str, int]:
    return {str(facilities[idx]): int(mask[idx]) for idx in range(len(facilities))}


def _size_bucket(total_therapies: int) -> str:
    if int(total_therapies) <= 224:
        return "small"
    if int(total_therapies) <= 437:
        return "medium"
    return "large"


def _sample_demand_kwargs(
    rng: np.random.Generator,
    distribution: str,
    per_day_cap: int,
) -> Dict[str, object]:
    if distribution == "poisson":
        return {
            "distribution": "poisson",
            "lambda_demand": float(rng.uniform(2.0, 3.5)),
            "per_day_cap": int(per_day_cap),
        }
    if distribution == "seasonal_poisson":
        return {
            "distribution": "seasonal_poisson",
            "lambda_demand": float(rng.uniform(2.0, 3.5)),
            "season_amp": float(rng.uniform(0.3, 1.0)),
            "season_period": int(rng.choice([7, 14, 30, 60, 90])),
            "per_day_cap": int(per_day_cap),
        }
    if distribution == "bursty":
        return {
            "distribution": "bursty",
            "lambda_demand": float(rng.uniform(2.0, 3.5)),
            "n_bursts": int(rng.integers(1, 4)),
            "burst_len": int(rng.choice([3, 5, 7])),
            "burst_mult": float(rng.uniform(2.0, 4.0)),
            "per_day_cap": int(per_day_cap),
        }
    raise ValueError(f"Unsupported distribution: {distribution}")


def _tame_large_demand(total_therapies: int, demand_kwargs: Dict[str, object]) -> Dict[str, object]:
    kw = dict(demand_kwargs)
    if _size_bucket(total_therapies) != "large":
        return kw
    dist = kw.get("distribution")
    if dist == "seasonal_poisson":
        kw["lambda_demand"] = float(min(float(kw.get("lambda_demand", 3.5)), 2.4))
        kw["season_amp"] = float(min(float(kw.get("season_amp", 1.0)), 0.5))
    elif dist == "bursty":
        kw["lambda_demand"] = float(min(float(kw.get("lambda_demand", 3.5)), 2.4))
        kw["burst_mult"] = float(min(float(kw.get("burst_mult", 4.0)), 2.5))
        kw["n_bursts"] = int(min(int(kw.get("n_bursts", 1)), 2))
    return kw


def _distribution_for_instance(rng: np.random.Generator, mode: str, idx: int) -> str:
    mode_key = str(mode).strip().lower()
    if mode_key == "mixed":
        return DEMAND_DISTRIBUTIONS[int(idx) % len(DEMAND_DISTRIBUTIONS)]
    if mode_key == "random":
        return str(rng.choice(DEMAND_DISTRIBUTIONS))
    if mode_key in DEMAND_DISTRIBUTIONS:
        return mode_key
    raise ValueError(f"Unsupported distribution mode '{mode}'.")


def _make_split_specs(
    split_counts: Dict[str, int],
    total_low: int,
    total_high: int,
    distribution_mode: str,
    per_day_cap: int,
    seed: int,
) -> Dict[str, list[dict]]:
    total_instances = sum(int(v) for v in split_counts.values())
    totals = [int(v) for v in dm.sobol_totals(total_instances, total_low, total_high, seed=seed)]
    rng = np.random.default_rng(int(seed))
    specs: Dict[str, list[dict]] = {split: [] for split in SPLITS}
    offset = 0
    for split in SPLITS:
        for local_id in range(int(split_counts[split])):
            global_idx = offset + local_id
            total = totals[global_idx]
            distribution = _distribution_for_instance(rng, distribution_mode, global_idx)
            demand_kwargs = _sample_demand_kwargs(rng, distribution, per_day_cap)
            demand_kwargs = _tame_large_demand(total, demand_kwargs)
            specs[split].append(
                {
                    "instance_id": int(local_id),
                    "global_demand_index": int(global_idx),
                    "total_therapies": int(total),
                    "seed_demand": int(rng.integers(1, 10**9)),
                    "distribution": distribution,
                    "demand_kwargs": demand_kwargs,
                    "size_bucket": _size_bucket(total),
                }
            )
        offset += int(split_counts[split])
    return specs


def _nested_inc(inc_raw: Dict[tuple, int], clinics: Sequence[str]) -> Dict[str, Dict[int, int]]:
    inc = {str(c): {} for c in clinics}
    for (clinic, day), value in inc_raw.items():
        inc[str(clinic)][int(day)] = int(value)
    return inc


def _compute_peak_active_demand(inc: Dict[str, Dict[int, int]], t_horizon: int, tmfe: int) -> float:
    daily = np.zeros(int(t_horizon) + 1, dtype=float)
    for days in inc.values():
        for day, value in days.items():
            day_int = int(day)
            if 1 <= day_int <= int(t_horizon):
                daily[day_int] += float(value)
    peak = 0.0
    for day in range(1, int(t_horizon) + 1):
        start = max(1, day - int(tmfe) + 1)
        peak = max(peak, float(daily[start : day + 1].sum()))
    return peak


def _demand_by_clinic(inc: Dict[str, Dict[int, int]], clinics: Sequence[str]) -> np.ndarray:
    values = [float(sum((inc.get(str(c), {}) or {}).values())) for c in clinics]
    arr = np.asarray(values, dtype=float)
    total = float(arr.sum())
    if total <= 0.0:
        return np.full(len(clinics), 1.0 / max(1, len(clinics)), dtype=float)
    return arr / total


def _meta_row(
    instance_id: int,
    static_config: Dict[str, object],
    inc: Dict[str, Dict[int, int]],
    spec: Dict[str, object],
) -> Dict[str, object]:
    return {
        "static": {
            "c": list(static_config["C"]),
            "m": list(static_config["M"]),
            "FCAP": static_config["FCAP"],
            "U1": static_config["U1"],
        },
        "inc": inc,
        "objective_model": "cart_detailed",
        "instance_id": int(instance_id),
        "demand_kwargs": dict(spec["demand_kwargs"]),
        "total_therapies": int(spec["total_therapies"]),
        "seed_demand": int(spec["seed_demand"]),
        "size_bucket": str(spec["size_bucket"]),
        "distribution": str(spec["distribution"]),
        "global_demand_index": int(spec["global_demand_index"]),
    }


def _instance_input(static_config: dict, spec: dict) -> dict:
    clinics = [str(c) for c in static_config["C"]]
    demand_horizon = int(static_config.get("demand_horizon", TIME_HORIZON))
    inc_raw = draw_inc_for_instance(
        clinics, demand_horizon, int(spec["total_therapies"]), int(spec["seed_demand"]),
        **dict(spec["demand_kwargs"]),
    )
    meta = _meta_row(int(spec["instance_id"]), static_config, _nested_inc(inc_raw, clinics), spec)
    meta.update(
        time_horizon=int(static_config.get("time_horizon", 130)),
        demand_horizon=demand_horizon,
        turnaround_limit=float(static_config.get("turnaround_limit", 17.0)),
        formulation=resolve_formulation(static_config.get("formulation", "detailed"),
                                        static_config.get("turnaround_limit", 17.0)),
    )
    return meta


class DetailedCartScorer:
    def __init__(
        self,
        static_config: Dict[str, object],
        inc: Dict[str, Dict[int, int]],
        *,
        seed: int,
        threads: int,
        full_time_limit: float,
        fixed_time_limit: float,
        mip_gap: float,
    ) -> None:
        self.C = [str(c) for c in static_config["C"]]
        self.M = [str(m) for m in static_config["M"]]
        self.n = len(self.M)
        capacities = static_config.get("CAPMAX", static_config.get("FCAP"))
        self.fcap = np.asarray([float(capacities[m]) for m in self.M], dtype=float)
        self.util_cap = float(static_config.get("util_cap", FACILITY_UTIL_CAP))
        self.T = int(static_config.get("time_horizon", 130))
        self.turnaround_limit = float(static_config.get("turnaround_limit", 17.0))
        self.formulation = resolve_formulation(static_config.get("formulation", "detailed"), self.turnaround_limit)
        self.seed = int(seed)
        self.threads = int(threads)
        self.full_time_limit = float(full_time_limit)
        self.fixed_time_limit = float(fixed_time_limit)
        self.mip_gap = float(mip_gap)
        self.reference_result = None
        self.reference_mask = None
        self.peak_active_demand = _compute_peak_active_demand(
            inc, self.T, int(static_config.get("TMFE", TMFE)),
        )
        params = {
            "CAPMAX": capacities,
            "CIM": static_config["CIM"],
            "CVM": static_config["CVM"],
            "U1": {(c, m): float(static_config["U1"][c][m]) for c in self.C for m in self.M},
            "TMFE": int(static_config.get("TMFE", TMFE)),
        }
        inc_raw = {(c, int(day)): int(value) for c, days in inc.items() for day, value in days.items()}
        self.data = build_detailed_data_from_aggregate(
            {"c": self.C, "m": self.M}, params, inc_raw, time_horizon=self.T,
        )

    def solve_full(self) -> tuple[np.ndarray, Optional[float], str, bool, float, list[tuple[np.ndarray, float]]]:
        if gp is None:
            raise RuntimeError("Sampled CART generation requires gurobipy and a working Gurobi license.")
        from .model_cart_detailed_gurobi import build_detailed_gurobi_model, detailed_data_from_mapping

        started = time.perf_counter()
        model, artifacts = build_detailed_gurobi_model(
            detailed_data_from_mapping(self.data, time_horizon=self.T),
            turnaround_limit=self.turnaround_limit,
            max_open_facilities=None,
            solver_threads=self.threads,
            time_limit=self.full_time_limit if self.full_time_limit > 0 else None,
            mip_gap=self.mip_gap,
            solver_options={"Seed": self.seed, "MIPGapAbs": 0.0},
        )
        build_time = time.perf_counter() - started
        try:
            z_vars = [artifacts.E1[m] for m in self.M]
            collector = IncumbentCollector(z_vars)
            started = time.perf_counter()
            model.optimize(collector)
            runtime = time.perf_counter() - started
            status = int(model.Status)
            has_solution = int(model.SolCount) > 0
            result = {
                "status": _gurobi_status_name(status),
                "gurobi_status_code": status,
                "feasible": True if has_solution else False if status == GRB.INFEASIBLE else None,
                "proven_optimal": bool(status == GRB.OPTIMAL and has_solution),
                "objective": float(model.ObjVal) if has_solution else None,
                "best_bound": _finite_attr(model, "ObjBound"),
                "mip_gap_rel": _finite_attr(model, "MIPGap") if has_solution else None,
                "build_time_sec": build_time,
                "solve_wall_time_sec": runtime,
                "solver_time_sec": float(model.Runtime),
                "model_num_variables": int(model.NumVars),
                "model_num_constraints": int(model.NumConstrs),
                "model_num_binary_variables": int(model.NumBinVars),
                "source": f"{self.formulation}_reference",
                "formulation": self.formulation, "turnaround_limit": self.turnaround_limit,
            }
            self.reference_result = result
            if not has_solution:
                return np.zeros(self.n, dtype=np.int8), None, result["status"], False, runtime, []
            mask = np.asarray([int(round(var.X)) for var in z_vars], dtype=np.int8)
            self.reference_mask = _mask_key(mask)
            if self.reference_mask not in collector.seen:
                collector.rows.append((mask.copy(), result["objective"]))
            collector.rows.sort(key=lambda row: (float(row[1]), _mask_key(row[0])))
            return (mask, result["objective"], result["status"], is_complete(result, self.mip_gap),
                    runtime, collector.rows)
        finally:
            model.dispose()
            artifacts.env.dispose()

    def score_fixed(self, mask: Sequence[int], source: str) -> Dict[str, object]:
        bits = [int(bit) for bit in mask]
        if len(bits) != self.n or any(bit not in (0, 1) for bit in bits):
            raise ValueError("Mask must contain one binary value per facility.")
        if (_mask_key(bits) == self.reference_mask and self.reference_result is not None
                and is_complete(self.reference_result, self.mip_gap)):
            result = dict(self.reference_result)
        else:
            result = solve_mask(
                self.data, bits, time_horizon=self.T, max_open_facilities=None,
                time_limit=self.fixed_time_limit, threads=self.threads, mip_gap=self.mip_gap,
                formulation=self.formulation, turnaround_limit=self.turnaround_limit,
            )
        if not is_complete(result, self.mip_gap):
            raise IncompleteCandidatePool(
                f"Detailed fixed-mask solve is unresolved ({result['status']}, mask={bits}); "
                "increase --fixed-time-limit."
            )
        capacity = float(np.dot(self.fcap * self.util_cap, bits))
        # Capacity estimates guide sampling only; patient scheduling determines feasibility.
        violation = max(0.0, self.peak_active_demand - capacity) / max(self.peak_active_demand, 1.0)
        return dict(
            result, mask=bits, solver_source=result.get("source"), source=source,
            capacity=capacity, capacity_heuristic_violation=violation,
        )


def _gurobi_status_name(status: int) -> str:
    mapping = {
        GRB.OPTIMAL: "optimal",
        GRB.INFEASIBLE: "infeasible",
        GRB.INF_OR_UNBD: "inf_or_unbd",
        GRB.TIME_LIMIT: "time_limit",
        GRB.INTERRUPTED: "interrupted",
        GRB.SUBOPTIMAL: "suboptimal",
    }
    return mapping.get(int(status), f"STATUS_{int(status)}")


def _add_candidate(
    candidates: Dict[tuple[int, ...], str],
    mask: Sequence[int],
    source: str,
) -> None:
    arr = np.asarray(mask, dtype=np.int8).reshape(-1)
    key = _mask_key(arr)
    if key not in candidates:
        candidates[key] = str(source)


def _demand_weighted_transport(static_config: Dict[str, object], inc: Dict[str, Dict[int, int]]) -> np.ndarray:
    clinics = [str(c) for c in static_config["C"]]
    facilities = [str(m) for m in static_config["M"]]
    weights = _demand_by_clinic(inc, clinics)
    scores = []
    for m in facilities:
        score = 0.0
        for weight, c in zip(weights, clinics):
            score += float(weight) * float(static_config["U1"][c][m])
        scores.append(score)
    return np.asarray(scores, dtype=float)


def _ranked_mask(
    ranking: Sequence[int],
    capacities: np.ndarray,
    peak_active_demand: float,
    *,
    extra: int = 0,
    stop_fraction: float = 1.0,
) -> np.ndarray:
    n = len(capacities)
    mask = np.zeros(n, dtype=np.int8)
    cap = 0.0
    target = float(peak_active_demand) * float(stop_fraction)
    used = 0
    for idx in ranking:
        if used >= n:
            break
        j = int(idx)
        if mask[j]:
            continue
        mask[j] = 1
        cap += float(capacities[j])
        used += 1
        if cap + EPS >= target:
            break
    for idx in ranking:
        if extra <= 0:
            break
        j = int(idx)
        if mask[j]:
            continue
        mask[j] = 1
        extra -= 1
    return mask


def _sample_exact_k(rng: np.random.Generator, n: int, k: int) -> np.ndarray:
    k_eff = max(0, min(int(n), int(k)))
    mask = np.zeros(int(n), dtype=np.int8)
    if k_eff:
        idx = rng.choice(np.arange(int(n)), size=k_eff, replace=False)
        mask[np.asarray(idx, dtype=int)] = 1
    return mask


def _mutate_mask(
    rng: np.random.Generator,
    base_mask: np.ndarray,
    *,
    n_flip: int,
) -> np.ndarray:
    mask = np.asarray(base_mask, dtype=np.int8).copy()
    n = int(mask.size)
    k = max(1, min(int(n_flip), n))
    idx = rng.choice(np.arange(n), size=k, replace=False)
    mask[np.asarray(idx, dtype=int)] = 1 - mask[np.asarray(idx, dtype=int)]
    return mask


def _swap_mask(rng: np.random.Generator, base_mask: np.ndarray, n_swaps: int = 1) -> Optional[np.ndarray]:
    mask = np.asarray(base_mask, dtype=np.int8).copy()
    open_idx = np.flatnonzero(mask == 1)
    closed_idx = np.flatnonzero(mask == 0)
    if open_idx.size == 0 or closed_idx.size == 0:
        return None
    k = min(int(n_swaps), int(open_idx.size), int(closed_idx.size))
    out_idx = rng.choice(open_idx, size=k, replace=False)
    in_idx = rng.choice(closed_idx, size=k, replace=False)
    mask[np.asarray(out_idx, dtype=int)] = 0
    mask[np.asarray(in_idx, dtype=int)] = 1
    return mask


def _propose_candidates(
    static_config: Dict[str, object],
    inc: Dict[str, Dict[int, int]],
    best_mask: np.ndarray,
    incumbents: Sequence[tuple[np.ndarray, float]],
    peak_active_demand: float,
    target_temp_size: int,
    seed: int,
) -> Dict[tuple[int, ...], str]:
    rng = np.random.default_rng(int(seed))
    facilities = [str(m) for m in static_config["M"]]
    n = len(facilities)
    caps = np.asarray([float(static_config["FCAP"][m]) * float(static_config.get("util_cap", 1.0)) for m in facilities])
    fixed = np.asarray([float(static_config["CIM"][m]) + float(static_config["CVM"][m]) for m in facilities])
    transport = _demand_weighted_transport(static_config, inc)
    opt_k = int(np.asarray(best_mask, dtype=np.int8).sum())

    candidates: Dict[tuple[int, ...], str] = {}
    _add_candidate(candidates, best_mask, "reference")
    _add_candidate(candidates, np.zeros(n, dtype=np.int8), "all_zero")
    _add_candidate(candidates, np.ones(n, dtype=np.int8), "all_one")
    for mask, _obj in incumbents:
        _add_candidate(candidates, mask, "incumbent")

    normalized_fixed = fixed / max(float(np.max(fixed)), 1.0)
    normalized_transport = transport / max(float(np.max(transport)), 1.0)
    score_vectors = {
        "transport": transport,
        "fixed": fixed,
        "fixed_per_cap": fixed / np.maximum(caps, 1.0),
        "blend": 0.55 * normalized_transport + 0.45 * normalized_fixed,
        "capacity": -caps,
        "cap_per_blend": (0.35 * normalized_fixed + 0.65 * normalized_transport) / np.maximum(caps, 1.0),
    }
    for name, scores in score_vectors.items():
        ranking = np.argsort(scores, kind="stable")
        for stop_fraction in (0.75, 0.9, 1.0, 1.1):
            for extra in (0, 1, 2, 3):
                mask = _ranked_mask(
                    ranking,
                    caps,
                    peak_active_demand,
                    extra=extra,
                    stop_fraction=stop_fraction,
                )
                _add_candidate(candidates, mask, f"heuristic_{name}")
        for k in range(max(0, opt_k - 3), min(n, opt_k + 5) + 1):
            mask = np.zeros(n, dtype=np.int8)
            if k:
                mask[ranking[:k]] = 1
            _add_candidate(candidates, mask, f"heuristic_topk_{name}")

    for h in (1, 2, 3, 4, 5, 6, max(1, n // 2)):
        attempts = max(12, 3 * h)
        for _ in range(attempts):
            _add_candidate(candidates, _mutate_mask(rng, best_mask, n_flip=h), f"hamming_{h}")

    for swaps in (1, 2, 3):
        for _ in range(24):
            mask = _swap_mask(rng, best_mask, n_swaps=swaps)
            if mask is not None:
                _add_candidate(candidates, mask, f"swap_{swaps}")

    open_idx = np.flatnonzero(best_mask == 1)
    closed_idx = np.flatnonzero(best_mask == 0)
    for add_count in (1, 2, 3, 4):
        if closed_idx.size >= add_count:
            for _ in range(12):
                mask = best_mask.copy()
                chosen = rng.choice(closed_idx, size=add_count, replace=False)
                mask[np.asarray(chosen, dtype=int)] = 1
                _add_candidate(candidates, mask, f"add_{add_count}")
    for drop_count in (1, 2, 3, 4):
        if open_idx.size >= drop_count:
            for _ in range(12):
                mask = best_mask.copy()
                chosen = rng.choice(open_idx, size=drop_count, replace=False)
                mask[np.asarray(chosen, dtype=int)] = 0
                _add_candidate(candidates, mask, f"drop_{drop_count}")

    low_cap_order = np.argsort(caps, kind="stable")
    high_transport_order = np.argsort(-transport, kind="stable")
    for ranking, name in ((low_cap_order, "low_capacity"), (high_transport_order, "bad_transport")):
        cap = 0.0
        mask = np.zeros(n, dtype=np.int8)
        for idx in ranking:
            j = int(idx)
            if cap + float(caps[j]) >= peak_active_demand:
                break
            mask[j] = 1
            cap += float(caps[j])
            _add_candidate(candidates, mask, f"infeasible_{name}")

    density_values = sorted(
        set(
            [
                1.0 / max(n, 1),
                max(0.05, (opt_k - 3) / max(n, 1)),
                max(0.05, opt_k / max(n, 1)),
                min(0.95, (opt_k + 3) / max(n, 1)),
                0.25,
                0.5,
                0.75,
            ]
        )
    )
    attempts = 0
    max_attempts = max(5000, int(target_temp_size) * 80)
    while len(candidates) < int(target_temp_size) and attempts < max_attempts:
        attempts += 1
        if rng.random() < 0.55:
            k_lo = max(0, opt_k - max(3, n // 5))
            k_hi = min(n, opt_k + max(4, n // 4))
            k = int(rng.integers(k_lo, k_hi + 1))
            mask = _sample_exact_k(rng, n, k)
            source = "random_exact_k"
        else:
            density = float(density_values[int(rng.integers(0, len(density_values)))])
            mask = (rng.random(n) < density).astype(np.int8)
            source = "random_density"
        _add_candidate(candidates, mask, source)

    return candidates


def _row_regret(row: Dict[str, object]) -> float:
    val = row.get("regret_rel")
    if val is None:
        return float("inf")
    try:
        return float(val)
    except Exception:
        return float("inf")


def _row_hamming(row: Dict[str, object]) -> int:
    val = row.get("hamming_r")
    if val is None:
        return 10**9
    return int(val)


def _row_open_count(row: Dict[str, object]) -> int:
    extras = row.get("extras") or {}
    return int(extras.get("open_count", sum(int(v) for v in (row.get("e1") or {}).values())))


def _banded_select(
    rows: Sequence[Dict[str, object]],
    target: int,
    band_fn,
    band_order: Sequence[str],
) -> list[Dict[str, object]]:
    if int(target) <= 0:
        return []
    by_band: Dict[str, list[Dict[str, object]]] = {band: [] for band in band_order}
    for row in rows:
        by_band.setdefault(str(band_fn(row)), []).append(row)
    for band_rows in by_band.values():
        band_rows.sort(key=lambda r: (_row_regret(r), _row_hamming(r), _row_open_count(r), str(r.get("source", ""))))

    selected: list[Dict[str, object]] = []
    seen: set[tuple[int, ...]] = set()
    while len(selected) < int(target):
        progressed = False
        for band in band_order:
            bucket = by_band.get(str(band), [])
            while bucket:
                row = bucket.pop(0)
                key = tuple(int(v) for _m, v in sorted((row.get("e1") or {}).items()))
                if key in seen:
                    continue
                seen.add(key)
                selected.append(row)
                progressed = True
                break
            if len(selected) >= int(target):
                break
        if not progressed:
            break
    return selected[: int(target)]


def _feasible_regret_band(row: Dict[str, object]) -> str:
    regret = _row_regret(row)
    if regret <= 0.001 + EPS:
        return "hard"
    if regret <= 0.01 + EPS:
        return "medium"
    if regret <= 0.05 + EPS:
        return "easy"
    return "far"


def _infeasible_violation_band(row: Dict[str, object]) -> str:
    extras = row.get("extras") or {}
    viol = float(extras.get("violation", 0.0))
    if viol <= 0.10 + EPS:
        return "near"
    if viol <= 0.35 + EPS:
        return "medium"
    return "large"


def _sample_to_row(
    instance_id: int,
    scored: Dict[str, object],
    best_mask: np.ndarray,
    facilities: Sequence[str],
    spec: Dict[str, object],
    reference: Dict[str, object],
    full_optimal: bool,
    time_horizon: int,
    demand_horizon: int,
) -> Dict[str, object]:
    result = dict(scored)
    mask = np.asarray(result.pop("mask"), dtype=np.int8)
    hamming_r = int(np.count_nonzero(mask != best_mask))
    return dict(
        result,
        instance_id=int(instance_id),
        objective_model="cart_detailed",
        e1=_mask_to_dict(mask, facilities),
        hamming_r=hamming_r,
        seed_demand=int(spec["seed_demand"]),
        seed_solver=None,
        fixed_mask_optimal=bool(result["proven_optimal"]),
        optimal=bool(full_optimal and result["label_type"] == "positive"),
        demand_kwargs=dict(spec["demand_kwargs"]),
        total_therapies=int(spec["total_therapies"]),
        extras={
            "source": result["source"],
            "solver_source": result["solver_source"],
            "hamming_dist": hamming_r,
            "open_count": int(mask.sum()),
            "capacity": result["capacity"],
            "violation": result["capacity_heuristic_violation"],
            "solver_runtime": result.get("solver_time_sec"),
            "from_incumbent": result["source"] == "incumbent",
            "time_horizon": int(time_horizon),
            "demand_horizon": int(demand_horizon),
            "full_milp_status": reference["status"],
            "full_milp_optimal": bool(full_optimal),
            "reference_objective": reference["objective"],
            "reference_best_bound": reference["best_bound"],
            "reference_mip_gap_rel": reference["mip_gap_rel"],
        },
    )


def _select_training_pool(
    rows: Sequence[Dict[str, object]],
    target_candidates: int,
    max_positives: int,
    infeasible_fraction: float,
) -> list[Dict[str, object]]:
    positives = [r for r in rows if r.get("label_type") == "positive"]
    positives.sort(key=lambda r: (_row_regret(r), _row_hamming(r), _row_open_count(r)))
    selected = positives[: max(1, min(int(max_positives), len(positives)))]
    selected_keys = {
        tuple(int(v) for _m, v in sorted((row.get("e1") or {}).items()))
        for row in selected
    }

    remaining = max(0, int(target_candidates) - len(selected))
    feasible_negs = [
        r for r in rows
        if r.get("feasible") and r.get("label_type") != "positive"
    ]
    infeasible = [r for r in rows if not r.get("feasible")]
    feasible_negs = [
        r for r in feasible_negs
        if tuple(int(v) for _m, v in sorted((r.get("e1") or {}).items())) not in selected_keys
    ]
    infeasible = [
        r for r in infeasible
        if tuple(int(v) for _m, v in sorted((r.get("e1") or {}).items())) not in selected_keys
    ]

    infeasible_target = int(round(remaining * float(infeasible_fraction)))
    infeasible_target = min(infeasible_target, len(infeasible))
    feasible_target = remaining - infeasible_target
    feasible_selected = _banded_select(
        feasible_negs,
        feasible_target,
        _feasible_regret_band,
        ("hard", "medium", "easy", "far"),
    )
    infeasible_selected = _banded_select(
        infeasible,
        infeasible_target,
        _infeasible_violation_band,
        ("near", "medium", "large"),
    )
    selected.extend(feasible_selected)
    selected.extend(infeasible_selected)

    if len(selected) < int(target_candidates):
        used = {
            tuple(int(v) for _m, v in sorted((row.get("e1") or {}).items()))
            for row in selected
        }
        top_up = sorted(
            [r for r in rows if tuple(int(v) for _m, v in sorted((r.get("e1") or {}).items())) not in used],
            key=lambda r: (
                0 if r.get("feasible") else 1,
                _row_regret(r),
                _row_hamming(r),
                _row_open_count(r),
            ),
        )
        selected.extend(top_up[: int(target_candidates) - len(selected)])

    selected.sort(
        key=lambda r: (
            int(r["instance_id"]),
            0 if r.get("label_type") == "positive" else 1,
            0 if r.get("feasible") else 2,
            _row_regret(r),
            _row_hamming(r),
            _row_open_count(r),
            str(r.get("source", "")),
        )
    )
    return selected[: int(target_candidates)]


def _generate_one_instance(
    static_config: Dict[str, object],
    spec: Dict[str, object],
    target_candidates: int,
    oversample_factor: int,
    max_positives: int,
    pos_threshold: float,
    infeasible_fraction: float,
    full_time_limit: float,
    fixed_time_limit: float,
    threads: int,
    seed: int,
    mip_gap: float,
    instance_meta: Optional[dict] = None,
    checkpoint_dir: Optional[Path] = None,
) -> Dict[str, object]:
    instance_id = int(spec["instance_id"])
    facilities = [str(m) for m in static_config["M"]]
    demand_horizon = int(static_config.get("demand_horizon", TIME_HORIZON))
    meta = dict(instance_meta if instance_meta is not None else _instance_input(static_config, spec))
    inc = meta["inc"]
    scorer = DetailedCartScorer(
        static_config, inc, seed=int(seed), threads=int(threads),
        full_time_limit=float(full_time_limit), fixed_time_limit=float(fixed_time_limit),
        mip_gap=float(mip_gap),
    )
    plan_path = checkpoint_dir / "plan.json" if checkpoint_dir is not None else None
    plan = json.loads(plan_path.read_text(encoding="utf-8")) if plan_path is not None and plan_path.exists() else None
    if plan is None:
        reference_mask, reference_obj, full_status, full_optimal, full_runtime, incumbents = scorer.solve_full()
        reference = scorer.reference_result
    else:
        reference_mask = np.asarray(plan["reference_mask"], dtype=np.int8)
        reference = plan["reference"]
        reference_obj = reference["objective"]
        full_status = reference["status"]
        full_optimal = bool(plan["full_optimal"])
        full_runtime = float(plan["full_runtime"])
        scorer.reference_mask = _mask_key(reference_mask)
        scorer.reference_result = reference
    meta["reference_solve"] = dict(reference, e1=_mask_to_dict(reference_mask, facilities))
    if reference_obj is None:
        if full_status != "infeasible":
            raise IncompleteCandidatePool(
                f"Detailed reference solve has no incumbent ({full_status}); increase --full-time-limit."
            )
        row = dict(
            reference, instance_id=instance_id, objective_model="cart_detailed",
            formulation=meta["formulation"], turnaround_limit=meta["turnaround_limit"],
            label_type="inst_infeas", e1={m: 0 for m in facilities},
            hamming_r=None, regret_abs=None, regret_rel=None,
            seed_demand=int(spec["seed_demand"]), seed_solver=int(seed), optimal=False,
            demand_kwargs=dict(spec["demand_kwargs"]), total_therapies=int(spec["total_therapies"]),
            extras={"source": "full_milp_no_incumbent", "violation": 1.0},
        )
        return {
            "instance_id": instance_id, "meta": meta, "samples": [row],
            "summary": {
                "instance_id": instance_id, "status": full_status, "optimal": False,
                "samples": 1, "positive": 0, "feasible": 0, "infeasible": 1,
                "full_runtime": full_runtime,
            },
        }

    if plan is None:
        target_temp = max(int(target_candidates) * max(1, int(oversample_factor)), int(target_candidates) + 64)
        proposed = _propose_candidates(
            static_config, inc, best_mask=reference_mask, incumbents=incumbents,
            peak_active_demand=float(scorer.peak_active_demand),
            target_temp_size=target_temp, seed=int(seed) + 17,
        )
        if plan_path is not None:
            _write_json(plan_path, {
                "reference_mask": reference_mask.tolist(), "reference": reference,
                "full_optimal": bool(full_optimal), "full_runtime": float(full_runtime),
                "proposed": [{"mask": list(_mask_key(key)), "source": source} for key, source in proposed.items()],
            })
    else:
        proposed = {_mask_key(row["mask"]): row["source"] for row in plan["proposed"]}
    mask_path = checkpoint_dir / "scored_masks.jsonl" if checkpoint_dir is not None else None
    cached = _load_mask_checkpoints(mask_path) if mask_path is not None else {}
    results = []
    for key, source in proposed.items():
        result = cached.get(key)
        if result is None or not is_complete(result, mip_gap):
            result = scorer.score_fixed(key, source=source)
            if mask_path is not None:
                _append_jsonl(mask_path, [result])
        results.append(result)
    labeled = label_candidate_results(results, mip_gap=mip_gap, tie_tolerance=pos_threshold)
    best = min((result for result in labeled if result["feasible"]), key=lambda result: result["objective"])
    best_mask = np.asarray(best["mask"], dtype=np.int8)
    scored_rows = [
        _sample_to_row(
            instance_id, result, best_mask, facilities, spec, reference, full_optimal,
            scorer.T, demand_horizon,
        )
        for result in labeled
    ]
    for row in scored_rows:
        row["extras"]["peak_active_demand"] = float(scorer.peak_active_demand)
        row.update(formulation=meta["formulation"], turnaround_limit=meta["turnaround_limit"])
    selected = _select_training_pool(
        scored_rows, target_candidates=int(target_candidates),
        max_positives=int(max_positives), infeasible_fraction=float(infeasible_fraction),
    )
    return {
        "instance_id": instance_id, "meta": meta, "samples": selected,
        "summary": {
            "instance_id": instance_id, "status": full_status, "optimal": bool(full_optimal),
            "samples": len(selected), "proposed": len(proposed), "scored": len(scored_rows),
            "positive": sum(1 for row in selected if row["label_type"] == "positive"),
            "feasible": sum(1 for row in selected if row["feasible"]),
            "infeasible": sum(1 for row in selected if not row["feasible"]),
            "best_obj": float(best["objective"]), "best_open_count": int(best_mask.sum()),
            "reference_objective": reference_obj,
            "reference_best_bound": reference["best_bound"],
            "reference_mip_gap_rel": reference["mip_gap_rel"],
            "peak_active_demand": float(scorer.peak_active_demand),
            "full_runtime": float(full_runtime),
        },
    }


def _target_candidates_for_facility_count(facility_count: int, args: argparse.Namespace) -> int:
    if args.candidates_per_instance is not None:
        return int(args.candidates_per_instance)
    defaults = {12: 128, 16: 160, 20: 192}
    return int(defaults.get(int(facility_count), 192))


def _split_counts(args: argparse.Namespace) -> Dict[str, int]:
    return {
        "train": int(args.train_count),
        "val": int(args.val_count),
        "test": int(args.test_count),
    }


def _copy_static_to_split(scale_dir: Path, split: str, static_config: Dict[str, object]) -> None:
    split_dir = scale_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    _write_json(split_dir / "static_config.json", static_config)


def _split_summary(split: str, results: Sequence[dict], input_count: int) -> dict:
    label_counts: Dict[str, int] = {}
    for result in results:
        for row in result["samples"]:
            label = str(row.get("label_type"))
            label_counts[label] = label_counts.get(label, 0) + 1
    sample_count = sum(len(result["samples"]) for result in results)
    return {
        "split": split,
        "instances": len(results),
        "input_instances": input_count,
        "complete": len(results) == input_count,
        "samples": sample_count,
        "avg_samples_per_instance": float(sample_count / max(1, len(results))),
        "label_counts": label_counts,
        "full_milp_optimal": sum(1 for result in results if result["summary"].get("optimal")),
        "avg_full_runtime": float(np.mean([float(result["summary"].get("full_runtime", 0.0)) for result in results]))
        if results else 0.0,
    }


def _write_split_dataset(
    scale_dir: Path,
    split: str,
    static_config: Dict[str, object],
    results: Sequence[Dict[str, object]],
    input_metas: Sequence[dict],
) -> Dict[str, object]:
    split_dir = scale_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    _write_json(split_dir / "static_config.json", static_config)
    metas = {int(meta["instance_id"]): meta for meta in input_metas}
    metas.update({int(result["instance_id"]): result["meta"] for result in results})
    samples = [
        row
        for result in sorted(results, key=lambda r: int(r["instance_id"]))
        for row in result["samples"]
    ]
    _write_jsonl(split_dir / "instances_meta.jsonl", (metas[key] for key in sorted(metas)))
    _write_jsonl(split_dir / "samples.jsonl", samples)
    summary = _split_summary(split, results, len(input_metas))
    _write_json(split_dir / "summary.json", summary)
    return summary


def _write_manifest(scale_dir: Path, scale_name: str, results_by_split: dict[str, Sequence[dict]]) -> None:
    manifest_rows: list[dict] = []
    for split in SPLITS:
        for result in sorted(results_by_split[split], key=lambda row: int(row["instance_id"])):
            s = result["summary"]
            manifest_rows.append(
                {
                    "scale": scale_name,
                    "split": split,
                    "instance_id": int(result["instance_id"]),
                    "samples": int(s.get("samples", 0)),
                    "positive": int(s.get("positive", 0)),
                    "feasible": int(s.get("feasible", 0)),
                    "infeasible": int(s.get("infeasible", 0)),
                    "best_open_count": s.get("best_open_count"),
                    "peak_active_demand": s.get("peak_active_demand"),
                    "full_milp_status": s.get("status"),
                    "full_milp_optimal": bool(s.get("optimal", False)),
                    "full_runtime": s.get("full_runtime"),
                }
            )
    manifest_path = scale_dir / "manifest.csv"
    temporary = manifest_path.with_name(manifest_path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = list(manifest_rows[0].keys()) if manifest_rows else ["scale", "split", "instance_id"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    temporary.replace(manifest_path)


def _resume_static_config(static_config: dict) -> dict:
    generation = dict(static_config["candidate_generation"])
    for key in ("full_time_limit", "fixed_time_limit_per_mask"):
        generation.pop(key, None)
    return dict(static_config, candidate_generation=generation)


def _prepare_scale(
    facility_count: int,
    static_sets: Dict[str, list],
    static_params: Dict[str, dict],
    split_specs: Dict[str, list[dict]],
    args: argparse.Namespace,
) -> tuple[dict, dict[str, list[dict]]]:
    scale_name = f"m{int(facility_count)}"
    scale_dir = args.out_dir / scale_name
    if scale_dir.exists() and args.overwrite:
        shutil.rmtree(scale_dir)

    expanded_sets, expanded_params, metadata = _expand_facilities(
        static_sets,
        static_params,
        target_facilities=int(facility_count),
        seed=int(args.seed),
    )
    static_config = _static_config(expanded_sets, expanded_params, metadata)
    static_config["max_facilities"] = None
    static_config["objective_model"] = "cart_detailed"
    static_config["CAPMAX"] = dict(static_config["FCAP"])
    static_config["time_horizon"] = int(args.time_horizon)
    static_config["demand_horizon"] = int(args.demand_horizon)
    static_config["turnaround_limit"] = float(args.turnaround_limit)
    static_config["formulation"] = resolve_formulation(args.formulation, args.turnaround_limit)
    static_config["candidate_generation"] = {
        "strategy": "sampled_detailed_cart_pool",
        "candidates_per_instance": _target_candidates_for_facility_count(facility_count, args),
        "oversample_factor": int(args.oversample_factor),
        "pos_threshold": float(args.pos_threshold),
        "infeasible_fraction": float(args.infeasible_fraction),
        "mip_gap": float(args.mip_gap),
        "full_time_limit": float(args.full_time_limit),
        "fixed_time_limit_per_mask": float(args.fixed_time_limit),
        "formulation": static_config["formulation"],
        "turnaround_limit": static_config["turnaround_limit"],
    }
    identity = {
        "format_version": 1,
        "static_config": _resume_static_config(static_config),
        "split_specs": split_specs,
        "seed": int(args.seed),
        "max_positives": int(args.max_positives),
    }
    state_path = scale_dir / "generation_state.json"
    if scale_dir.exists():
        if not args.resume:
            raise SystemExit(f"{scale_dir} already exists; use --resume, a new --out-dir or --overwrite.")
        if state_path.exists():
            saved_identity = json.loads(state_path.read_text(encoding="utf-8"))
            if saved_identity != identity:
                raise SystemExit(f"Cannot resume {scale_dir}: generation settings or base data differ from the saved run.")
        else:
            # Adopt old failed runs only when they contain static configs and no results.
            allowed = {Path("static_config.json"), *(Path(split) / "static_config.json" for split in SPLITS)}
            for path in scale_dir.rglob("*"):
                if path.is_file() and (
                    path.relative_to(scale_dir) not in allowed
                    or _resume_static_config(json.loads(path.read_text(encoding="utf-8"))) != identity["static_config"]
                ):
                    raise SystemExit(f"Cannot resume {scale_dir}: existing files have no compatible generation checkpoint.")
    scale_dir.mkdir(parents=True, exist_ok=True)
    if not state_path.exists():
        _write_json(state_path, identity)
    static_path = scale_dir / "static_config.json"
    if not static_path.exists():
        _write_json(static_path, static_config)
    patient_sets, patient_params = load_static_from_config(scale_dir / "static_config.json")
    inputs = {}
    for split in SPLITS:
        split_dir = scale_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        split_static_path = split_dir / "static_config.json"
        if not split_static_path.exists():
            _write_json(split_static_path, static_config)
        input_path = split_dir / "instances_input.jsonl"
        if input_path.exists():
            metas = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if len(metas) != len(split_specs[split]) or any(
                any(meta.get(key) != value for key, value in spec.items())
                for meta, spec in zip(metas, split_specs[split])
            ):
                raise SystemExit(f"Cannot resume {split_dir}: saved input metadata does not match the run.")
        else:
            metas = [_instance_input(static_config, spec) for spec in split_specs[split]]
            _write_jsonl(input_path, metas)
        inputs[split] = metas
        if not (split_dir / "instances_meta.jsonl").exists():
            _write_jsonl(split_dir / "instances_meta.jsonl", metas)
        if not (split_dir / "samples.jsonl").exists():
            _write_jsonl(split_dir / "samples.jsonl", [])
        if not (split_dir / "summary.json").exists():
            _write_json(split_dir / "summary.json", _split_summary(split, [], len(metas)))
        patient_dir = scale_dir / f"{split}_patient_dats"
        for meta in metas:
            instance_id = int(meta["instance_id"])
            patient_path = patient_dir / f"instance_{instance_id}.dat"
            if patient_path.exists():
                continue
            inc = {(str(c), int(day)): int(value) for c, days in meta["inc"].items() for day, value in days.items()}
            data = build_detailed_data_from_aggregate(
                patient_sets, patient_params, inc, time_horizon=int(args.time_horizon),
            )
            write_patient_dat(
                patient_path, data,
                title=f"Instance {instance_id}; horizon={args.time_horizon}; turnaround_limit={args.turnaround_limit}",
            )
    print(f"[{scale_name}] Saved {sum(len(rows) for rows in inputs.values())} instance inputs to {scale_dir}", flush=True)
    return static_config, inputs


def _generate_scale(
    facility_count: int,
    static_config: dict,
    input_metas: dict[str, list[dict]],
    split_specs: Dict[str, list[dict]],
    args: argparse.Namespace,
) -> Dict[str, object]:
    scale_name = f"m{int(facility_count)}"
    scale_dir = args.out_dir / scale_name
    target_candidates = _target_candidates_for_facility_count(facility_count, args)
    scale_summary = {
        "facility_count": int(facility_count),
        "target_candidates_per_instance": int(target_candidates),
        "splits": {},
    }
    results_by_split = {split: [] for split in SPLITS}
    for split in SPLITS:
        for spec in split_specs[split]:
            result_path = scale_dir / split / "checkpoints" / f"instance_{int(spec['instance_id']):04d}" / "result.json"
            if result_path.exists():
                result = json.loads(result_path.read_text(encoding="utf-8"))
                if int(result["instance_id"]) != int(spec["instance_id"]):
                    raise SystemExit(f"Checkpoint instance ID does not match {result_path}.")
                results_by_split[split].append(result)
        _write_split_dataset(scale_dir, split, static_config, results_by_split[split], input_metas[split])
    try:
        for split in SPLITS:
            results = results_by_split[split]
            completed = {int(result["instance_id"]) for result in results}
            if completed:
                print(f"[{scale_name}/{split}] Resuming with {len(completed)}/{len(split_specs[split])} completed instances", flush=True)
            for spec_idx, (spec, meta) in enumerate(zip(split_specs[split], input_metas[split])):
                if int(spec["instance_id"]) in completed:
                    continue
                seed = int(args.seed) + int(facility_count) * 100_000 + _split_seed_offset(split) + int(spec["instance_id"])
                checkpoint_dir = scale_dir / split / "checkpoints" / f"instance_{int(spec['instance_id']):04d}"
                result = _generate_one_instance(
                    static_config=static_config,
                    spec=spec,
                    target_candidates=target_candidates,
                    oversample_factor=int(args.oversample_factor),
                    max_positives=int(args.max_positives),
                    pos_threshold=float(args.pos_threshold),
                    infeasible_fraction=float(args.infeasible_fraction),
                    full_time_limit=float(args.full_time_limit),
                    fixed_time_limit=float(args.fixed_time_limit),
                    threads=int(args.solver_threads),
                    seed=seed,
                    mip_gap=float(args.mip_gap),
                    instance_meta=meta,
                    checkpoint_dir=checkpoint_dir,
                )
                _write_json(checkpoint_dir / "result.json", result)
                results.append(result)
                _append_jsonl(scale_dir / split / "samples.jsonl", result["samples"])
                s = result["summary"]
                if (spec_idx + 1) % max(1, int(args.progress_every)) == 0 or spec_idx + 1 == len(split_specs[split]):
                    _write_json(scale_dir / split / "summary.json", _split_summary(split, results, len(input_metas[split])))
                    print(
                        f"[{scale_name}/{split}] {len(results)}/{len(split_specs[split])} "
                        f"inst={int(spec['instance_id']):04d} "
                        f"pos={int(s.get('positive', 0))} feas={int(s.get('feasible', 0))} "
                        f"infeas={int(s.get('infeasible', 0))} "
                        f"open={s.get('best_open_count')} status={s.get('status')} "
                        f"runtime={float(s.get('full_runtime', 0.0)):.2f}s saved",
                        flush=True,
                    )
    finally:
        for split in SPLITS:
            results = results_by_split[split]
            scale_summary["splits"][split] = _write_split_dataset(
                scale_dir, split, static_config, results, input_metas[split],
            )
        scale_summary["complete"] = all(summary["complete"] for summary in scale_summary["splits"].values())
        _write_manifest(scale_dir, scale_name, results_by_split)
        _write_json(scale_dir / "summary.json", scale_summary)
    return scale_summary


def _split_seed_offset(split: str) -> int:
    return {"train": 0, "val": 20_000, "test": 40_000}[split]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate detailed CART training datasets from sampled facility masks.")
    parser.add_argument("--base-dat", type=Path, default=Path(__file__).with_name("Data_CART_og.dat"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/cart_sampled_detailed"))
    parser.add_argument("--facility-counts", type=int, nargs="+", default=[16])
    parser.add_argument("--train-count", type=int, default=800)
    parser.add_argument("--val-count", type=int, default=100)
    parser.add_argument("--test-count", type=int, default=100)
    parser.add_argument("--candidates-per-instance", type=int, default=None)
    parser.add_argument("--oversample-factor", type=int, default=2)
    parser.add_argument("--max-positives", type=int, default=3)
    parser.add_argument("--pos-threshold", type=float, default=1e-6)
    parser.add_argument("--infeasible-fraction", type=float, default=0.30)
    parser.add_argument("--total-low", type=int, default=10)
    parser.add_argument("--total-high", type=int, default=650)
    parser.add_argument("--distribution-mode", type=str, default="mixed")
    parser.add_argument("--seed", type=int, default=515151)
    parser.add_argument("--demand-horizon", type=int, default=TIME_HORIZON)
    parser.add_argument("--time-horizon", type=int, default=130)
    parser.add_argument("--turnaround-limit", type=float, default=17.0)
    parser.add_argument("--formulation", choices=("auto", "detailed"), default="detailed",
                        help="Detailed patient-level Gurobi formulation.")
    parser.add_argument("--mip-gap", type=float, default=1e-4)
    parser.add_argument("--full-time-limit", type=float, default=3600.0,
                        help="Detailed reference solve limit in seconds per instance; zero means unlimited.")
    parser.add_argument("--fixed-time-limit", type=float, default=3600.0,
                        help="Detailed fixed-mask solve limit in seconds per proposed mask.")
    parser.add_argument("--solver-threads", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=10)
    existing = parser.add_mutually_exclusive_group()
    existing.add_argument("--overwrite", action="store_true", help="Delete existing scales and start again.")
    existing.add_argument("--resume", action="store_true", help="Reuse saved inputs, masks and completed instances.")
    parser.add_argument("--prepare-only", action="store_true", help="Save all instance inputs without running any solves.")
    args = parser.parse_args(argv)
    counts = _split_counts(args)
    try:
        formulation = resolve_formulation(args.formulation, args.turnaround_limit)
    except ValueError as exc:
        parser.error(str(exc))
    if not 0 < args.demand_horizon <= args.time_horizon:
        parser.error("Demand horizon must be positive and within the detailed model horizon.")
    if not math.isfinite(args.mip_gap) or args.mip_gap < 0:
        parser.error("--mip-gap must be finite and nonnegative.")
    if any(count < 0 for count in counts.values()) or sum(counts.values()) == 0:
        parser.error("Split counts must be nonnegative and include at least one instance.")
    if args.total_low < 1 or args.total_high < args.total_low:
        parser.error("Demand totals must be positive, with --total-high >= --total-low.")
    if args.oversample_factor < 1 or args.max_positives < 1 or args.solver_threads < 1:
        parser.error("Oversampling, maximum positives and solver threads must be positive.")
    if args.candidates_per_instance is not None and args.candidates_per_instance < 1:
        parser.error("--candidates-per-instance must be positive.")
    if not math.isfinite(args.infeasible_fraction) or not 0 <= args.infeasible_fraction <= 1:
        parser.error("--infeasible-fraction must be between zero and one.")
    if not math.isfinite(args.pos_threshold) or args.pos_threshold < 0:
        parser.error("--pos-threshold must be finite and nonnegative.")
    if (not math.isfinite(args.full_time_limit) or args.full_time_limit < 0
            or not math.isfinite(args.fixed_time_limit) or args.fixed_time_limit <= 0):
        parser.error("Full solve limit must be nonnegative; fixed-mask limit must be positive.")
    if len(set(args.facility_counts)) != len(args.facility_counts):
        parser.error("Facility counts must be distinct.")
    if args.distribution_mode not in ("mixed", "random", *DEMAND_DISTRIBUTIONS):
        parser.error("Unknown --distribution-mode.")
    static_sets, static_params = dm.load_static_from_dat(args.base_dat.as_posix())
    if any(count < len(static_sets["m"]) for count in args.facility_counts):
        parser.error("Facility counts must be at least the base network size.")
    for count in args.facility_counts:
        scale_dir = args.out_dir / f"m{count}"
        if scale_dir.exists() and not (args.overwrite or args.resume):
            parser.error(f"{scale_dir} already exists; use --resume, a new --out-dir or --overwrite.")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    split_specs = _make_split_specs(
        _split_counts(args),
        total_low=int(args.total_low),
        total_high=int(args.total_high),
        distribution_mode=str(args.distribution_mode),
        per_day_cap=int(PER_DAY_CAP),
        seed=int(args.seed),
    )

    root_summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_dat": args.base_dat.as_posix(),
        "facility_counts": [int(v) for v in args.facility_counts],
        "split_counts": _split_counts(args),
        "objective_model": "cart_detailed",
        "formulation": formulation, "turnaround_limit": float(args.turnaround_limit),
        "generation_config": {
            key: value.as_posix() if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "strategy": "solve unrestricted detailed CART; sample, score, and stratify fixed-E1 masks",
        "scales": {},
    }
    prepared = {}
    for facility_count in args.facility_counts:
        static_config, inputs = _prepare_scale(int(facility_count), static_sets, static_params, split_specs, args)
        prepared[int(facility_count)] = (static_config, inputs)
        root_summary["scales"][f"m{int(facility_count)}"] = {
            "facility_count": int(facility_count),
            "target_candidates_per_instance": _target_candidates_for_facility_count(facility_count, args),
            "splits": {
                split: json.loads((args.out_dir / f"m{int(facility_count)}" / split / "summary.json").read_text(encoding="utf-8"))
                for split in SPLITS
            },
        }
    _write_json(args.out_dir / "summary.json", root_summary)
    if args.prepare_only:
        print(f"Prepared all CART instance inputs in {args.out_dir}; rerun with --resume to solve.")
        return 0

    try:
        for facility_count in args.facility_counts:
            static_config, inputs = prepared[int(facility_count)]
            summary = _generate_scale(
                facility_count=int(facility_count),
                static_config=static_config,
                input_metas=inputs,
                split_specs=split_specs,
                args=args,
            )
            root_summary["scales"][f"m{int(facility_count)}"] = summary
            _write_json(args.out_dir / "summary.json", root_summary)
    finally:
        for facility_count in args.facility_counts:
            scale_summary_path = args.out_dir / f"m{int(facility_count)}" / "summary.json"
            if scale_summary_path.exists():
                root_summary["scales"][f"m{int(facility_count)}"] = json.loads(scale_summary_path.read_text(encoding="utf-8"))
        _write_json(args.out_dir / "summary.json", root_summary)

    print(f"Wrote sampled CART facility training datasets to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
