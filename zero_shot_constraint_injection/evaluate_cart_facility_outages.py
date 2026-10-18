"""Evaluate CART facility constraints for trained EnerGNN checkpoints.

For each test instance, solve the original reference MIP, deactivate facilities
from that optimum or inject targeted facility states and count limits, and
compare inference against a reference with the same constraints. References
and mask scoring use the detailed Gurobi model.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import statistics
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from gnn.data.dataset import InstancePoolDataset
from gnn.inference.checkpoint import load_trained_model
from gnn.inference.run_milp import (
    PAPER_SOLVER_THREADS,
    PAPER_TIME_LIMIT,
    _load_eval_data,
    _relative_opt_gap,
    _resolve_model_type,
    _solve_cart_detailed_milp,
    resolve_fixing_inference_defaults,
    solve_instance_simple,
)


def _load_config(path: Optional[Path]) -> dict:
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"Missing config: {path}")
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("YAML config files require PyYAML. Use JSON or install PyYAML.") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {} if data is None else dict(data)


def _as_path(value) -> Optional[Path]:
    if value is None or str(value).strip() == "" or str(value).lower() == "null":
        return None
    return Path(str(value))


def _as_bool(value, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value}")


def _default_split_dirs(facility_count: int) -> Tuple[Path, Path]:
    if int(facility_count) != 16:
        raise ValueError("CART outage evaluation requires 16 facilities.")
    return Path("data/CART/m16/train"), Path("data/CART/m16/test")


def _load_run_config(checkpoint: Path) -> dict:
    path = checkpoint.parent / "run_config.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _load_cart_model(
    checkpoint: Path,
    *,
    model_type: str,
    metas: Dict[int, object],
    tmfe: int,
    dh: int,
    nlayers: int,
    dropout: float,
    ca_head_variant: str,
) -> object:
    probe_ds = InstancePoolDataset(metas, {k: [] for k in metas.keys()}, TMFE=tmfe)
    graph, _, _, _, _ = probe_ds[0]
    dc = int(graph["Xc"].shape[1])
    dm = int(graph["Xm"].shape[1])
    de = int(graph["edge_attr"].shape[1])
    model_cfg = {
        "dc": dc,
        "dm": dm,
        "de": de,
        "dh": int(dh),
        "nlayers": int(nlayers),
        "dropout": float(dropout),
        "problem": "cart",
        "ca_readout_variant": "sum_only_plus",
        "qkp_readout_variant": "sum_only",
        "ca_head_variant": str(ca_head_variant),
    }
    model = load_trained_model(
        _resolve_model_type(model_type),
        model_cfg,
        checkpoint,
        torch.device("cpu"),
    )
    return model


def _select_instance_ids(
    metas: Dict[int, object],
    *,
    instance_id: Optional[int],
    limit: int,
    num_shards: int,
    shard_index: int,
) -> List[int]:
    if instance_id is not None:
        inst_id = int(instance_id)
        if inst_id not in metas:
            raise ValueError(f"instance_id {inst_id} not found in test data.")
        ids = [inst_id]
    else:
        ids = sorted(int(k) for k in metas.keys())
        if int(limit) > 0:
            ids = ids[: int(limit)]

    n_shards = max(1, int(num_shards))
    idx = int(shard_index)
    if not (0 <= idx < n_shards):
        raise ValueError(f"--shard-index must be in [0, {n_shards}), got {idx}.")
    if n_shards > 1:
        ids = [inst_id for pos, inst_id in enumerate(ids) if pos % n_shards == idx]
    if not ids:
        raise FileNotFoundError("No test instances selected for this shard.")
    return ids


def _mask_key(mask: Optional[Sequence[int]]) -> Optional[Tuple[int, ...]]:
    if mask is None:
        return None
    return tuple(int(v) for v in mask)


def _hamming(a: Optional[Sequence[int]], b: Optional[Sequence[int]]) -> Optional[int]:
    if a is None or b is None:
        return None
    if len(a) != len(b):
        return None
    return sum(1 for x, y in zip(a, b) if int(x) != int(y))


def _facility_names(meta, indices: Sequence[int]) -> List[str]:
    return [str(meta.M[int(idx)]) for idx in indices]


def _fixed_assignments(off_indices: Sequence[int], on_indices: Sequence[int]) -> Dict[int, int]:
    forced: Dict[int, int] = {int(idx): 0 for idx in off_indices}
    for idx in on_indices:
        idx = int(idx)
        if idx in forced and forced[idx] != 1:
            raise ValueError(f"Conflicting outage scenario for facility index {idx}.")
        forced[idx] = 1
    return forced


def _sample_combinations(
    combos: List[Tuple[int, ...]],
    *,
    max_count: Optional[int],
    seed: int,
    instance_id: int,
    outage_size: int,
) -> List[Tuple[int, ...]]:
    if max_count is None or int(max_count) <= 0 or len(combos) <= int(max_count):
        return combos
    rng = np.random.default_rng(int(seed) + 1009 * int(instance_id) + 100003 * int(outage_size))
    picked = sorted(int(i) for i in rng.choice(len(combos), size=int(max_count), replace=False).tolist())
    return [combos[i] for i in picked]


def _generate_scenarios(
    original_mask: Sequence[int],
    *,
    outage_sizes: Sequence[int],
    max_pairs_per_instance: Optional[int],
    max_triples_per_instance: Optional[int],
    active_off_inactive_on: bool,
    seed: int,
    instance_id: int,
    force_off: Sequence[int] = (),
    force_on: Sequence[int] = (),
    max_facilities: Sequence[int] = (),
) -> List[dict]:
    off = tuple(sorted({int(v) for v in force_off}))
    on = tuple(sorted({int(v) for v in force_on}))
    _fixed_assignments(off, on)
    if any(idx < 0 or idx >= len(original_mask) for idx in (*off, *on)):
        raise ValueError("Forced facility indices must be within the instance's facility range.")
    caps = sorted({int(v) for v in max_facilities})
    if any(cap < 1 or cap > len(original_mask) for cap in caps):
        raise ValueError("Maximum facility counts must be between 1 and the facility count.")
    if any(len(on) > cap for cap in caps):
        raise ValueError("The number of forced active facilities exceeds a maximum facility count.")
    active = [idx for idx, bit in enumerate(original_mask) if int(bit) == 1]
    inactive = [idx for idx, bit in enumerate(original_mask) if int(bit) == 0]
    scenarios: List[dict] = []

    for size in sorted({int(v) for v in outage_sizes}):
        if size <= 0 or size > len(active):
            continue
        combos = list(itertools.combinations(active, size))
        if size == 2:
            combos = _sample_combinations(
                combos,
                max_count=max_pairs_per_instance,
                seed=seed,
                instance_id=instance_id,
                outage_size=size,
            )
        elif size == 3:
            combos = _sample_combinations(
                combos,
                max_count=max_triples_per_instance,
                seed=seed,
                instance_id=instance_id,
                outage_size=size,
            )
        for combo in combos:
            scenarios.append(
                {
                    "scenario_type": "active_off",
                    "outage_size": int(size),
                    "off_indices": tuple(int(i) for i in combo),
                    "on_indices": tuple(),
                }
            )

    if active_off_inactive_on:
        for off_idx in active:
            for on_idx in inactive:
                scenarios.append(
                    {
                        "scenario_type": "active_off_inactive_on",
                        "outage_size": 1,
                        "off_indices": (int(off_idx),),
                        "on_indices": (int(on_idx),),
                    }
                )
    if off or on or caps:
        for cap in caps or [None]:
            if off and on:
                scenario_type = "forced_active_inactive"
            elif on:
                scenario_type = "forced_active"
            elif off:
                scenario_type = "forced_inactive"
            else:
                scenario_type = "max_facilities"
            scenarios.append({
                "scenario_type": scenario_type,
                "outage_size": len(off),
                "off_indices": off,
                "on_indices": on,
                "max_facilities": cap,
            })
    return scenarios


def _scenario_max_k(scenario: dict, max_k: int) -> int:
    cap = scenario.get("max_facilities")
    if cap is None:
        return max_k
    return min(int(cap), max_k) if max_k > 0 else int(cap)


def _solve_cart_details(
    meta,
    *,
    min_k: int,
    max_k: int,
    time_limit: float,
    threads: Optional[int],
    mip_gap: Optional[float],
    fixed_assignments: Optional[Dict[int, int]] = None,
    fixed_mask: Optional[Sequence[int]] = None,
    formulation: str = "detailed",
) -> dict:
    record = _solve_cart_detailed_milp(
        meta, mask=fixed_mask, min_open_facilities=min_k,
        max_open_facilities=max_k if max_k > 0 else None,
        time_limit=time_limit, solver_threads=threads, mip_gap=mip_gap,
        fixed_assignments=fixed_assignments, formulation=formulation,
    )
    opened = set(record.get("opened_facilities") or [])
    return {
        "configuration": [int(m in opened) for m in meta.M]
        if record.get("objective") is not None else None,
        "objective": record.get("objective"), "status": record.get("status"),
        "runtime": record.get("solver_time_sec"), "bound": record.get("best_bound"),
        "solver_gap": record.get("mip_gap_rel"),
        "proven_optimal": bool(record.get("proven_optimal")),
    }


def _validate_mask_against_fixed(mask: Optional[Sequence[int]], fixed: Dict[int, int]) -> bool:
    if mask is None:
        return False
    for idx, val in fixed.items():
        if int(mask[int(idx)]) != int(val):
            return False
    return True


def _validate_mask_constraints(mask, fixed, min_k, max_k) -> bool:
    if not _validate_mask_against_fixed(mask, fixed):
        return False
    opened = sum(int(v) for v in mask)
    return opened >= min_k and (max_k <= 0 or opened <= max_k)


def _summarise_group(rows: List[dict]) -> dict:
    n = len(rows)
    gaps = [float(r["gap_to_exact_constrained_optimum"]) for r in rows if r.get("gap_to_exact_constrained_optimum") is not None]
    d_shift = [int(r["d_shift"]) for r in rows if r.get("d_shift") is not None]
    d_error = [int(r["d_error"]) for r in rows if r.get("d_error") is not None]
    energnn_times = [
        float(r["energnn_total_inference_time_sec"])
        for r in rows
        if r.get("energnn_total_inference_time_sec") is not None
    ]
    method_times = [
        float(r["method_total_inference_time_sec"])
        for r in rows
        if r.get("method_total_inference_time_sec") is not None
    ]
    reference_times = [
        float(r["reference_runtime_sec"])
        for r in rows
        if r.get("reference_runtime_sec") is not None
    ]

    def rate(key: str) -> Optional[float]:
        if n <= 0:
            return None
        return sum(1 for row in rows if bool(row.get(key))) / n

    return {
        "cases": n,
        "energnn_feasibility_rate": rate("energnn_feasible"),
        "method_feasibility_rate": rate("method_feasible"),
        "mean_gap": None if not gaps else float(sum(gaps) / len(gaps)),
        "median_gap": None if not gaps else float(statistics.median(gaps)),
        "max_gap": None if not gaps else float(max(gaps)),
        "gap_known_cases": len(gaps),
        "exact_configuration_rate": rate("exact_configuration_match"),
        "within_1pct": rate("within_1pct"),
        "within_5pct": rate("within_5pct"),
        "within_10pct": rate("within_10pct"),
        "mean_d_shift": None if not d_shift else float(sum(d_shift) / len(d_shift)),
        "mean_d_error": None if not d_error else float(sum(d_error) / len(d_error)),
        "mean_energnn_runtime_sec": None if not energnn_times else float(sum(energnn_times) / len(energnn_times)),
        "mean_method_runtime_sec": None if not method_times else float(sum(method_times) / len(method_times)),
        "mean_reference_runtime_sec": None if not reference_times else float(sum(reference_times) / len(reference_times)),
    }


def _shift_bucket(d_shift: Optional[int]) -> str:
    if d_shift is None:
        return "unknown"
    value = int(d_shift)
    if value >= 4:
        return ">=4"
    return str(value)


def _write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _json_safe(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_safe) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, separators=(",", ":"), default=_json_safe))
        fh.write("\n")


def _build_summary(rows: List[dict], skipped: List[dict], resolved: dict) -> dict:
    by_constraint = []
    keys = sorted({(row["scenario_type"], row["max_facilities"]) for row in rows})
    for scenario_type, max_facilities in keys:
        group = [row for row in rows if row["scenario_type"] == scenario_type and row["max_facilities"] == max_facilities]
        by_constraint.append({"scenario_type": scenario_type, "max_facilities": max_facilities, **_summarise_group(group)})
    by_size = []
    for size in sorted({int(row["outage_size"]) for row in rows}):
        group = [row for row in rows if int(row["outage_size"]) == size]
        by_size.append({"outage_size": size, **_summarise_group(group)})

    by_shift = []
    bucket_order = {"0": 0, "1": 1, "2": 2, "3": 3, ">=4": 4, "unknown": 5}
    for bucket in sorted({_shift_bucket(row.get("d_shift")) for row in rows}, key=lambda x: bucket_order.get(x, 99)):
        group = [row for row in rows if _shift_bucket(row.get("d_shift")) == bucket]
        by_shift.append({"d_shift_bucket": bucket, **_summarise_group(group)})

    return {
        "evaluation": {
            "name": "cart_facility_constraints",
            "facility_count": resolved["facility_count"],
            "checkpoint": resolved["checkpoint"],
            "test_dir": resolved["test_dir"],
            "outage_sizes": resolved["outage_sizes"],
            "force_off": resolved["force_off"],
            "force_on": resolved["force_on"],
            "max_facilities": resolved["max_facilities"],
            "active_off_inactive_on": resolved["active_off_inactive_on"],
            "max_pairs_per_instance": resolved["max_pairs_per_instance"],
            "max_triples_per_instance": resolved["max_triples_per_instance"],
            "seed": resolved["seed"],
            "cart_symmetry_breaking": resolved["cart_symmetry_breaking"],
        },
        "solver": {
            "time_limit": resolved["time_limit"],
            "threads": resolved["threads"],
            "mip_gap": resolved["mip_gap"],
            "min_k": resolved["min_k"],
            "max_k": resolved["max_k"],
        },
        "counts": {
            "evaluated_scenarios": len(rows),
            "skipped_scenarios": len(skipped),
            "reference_optimal": sum(1 for row in rows if bool(row.get("reference_proven_optimal"))),
            "energnn_feasible": sum(1 for row in rows if bool(row.get("energnn_feasible"))),
            "method_feasible": sum(1 for row in rows if bool(row.get("method_feasible"))),
        },
        "summary_by_outage_size": by_size,
        "summary_by_constraint": by_constraint,
        "summary_by_shift_distance": by_shift,
        "skipped": skipped,
    }


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CART facility activation, deactivation, limits, and outages.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--facility-count", type=int, default=None)
    parser.add_argument("--model-type", "--model_type", dest="model_type", type=str, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--test-dir", type=Path, default=None)
    parser.add_argument("--train-dir", type=Path, default=None, help="Accepted for command compatibility; test-dir is used for feature dimensions.")
    parser.add_argument("--static-config", type=Path, default=None)
    parser.add_argument("--dat-path", type=Path, default=None)
    parser.add_argument("--outage-sizes", type=int, nargs="*", default=None)
    parser.add_argument("--force-off", type=int, nargs="*", default=None,
                        help="Zero-based facility indices to deactivate in a targeted scenario.")
    parser.add_argument("--force-on", type=int, nargs="*", default=None,
                        help="Zero-based facility indices to activate in a targeted scenario.")
    parser.add_argument("--max-facilities", type=int, nargs="*", default=None,
                        help="Maximum active facility counts; evaluate a separate scenario for each count.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--active-off-inactive-on", action="store_true", default=None)
    parser.add_argument("--max-pairs-per-instance", type=int, default=None)
    parser.add_argument("--max-triples-per-instance", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--time-limit", type=float, default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--mip-gap", type=float, default=None)
    parser.add_argument("--min-k", type=int, default=None)
    parser.add_argument("--max-k", type=int, default=None)
    parser.add_argument("--dh", type=int, default=None)
    parser.add_argument("--nlayers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--ca-head-variant", type=str, default=None)
    parser.add_argument("--reference-formulation", choices=["detailed"],
                        default=None, help="Formulation used for original/constrained references and mask scoring.")
    parser.add_argument("--fixing-ratio", "--fixing_ratio", dest="fixing_ratio", type=float, default=None)
    parser.add_argument("--fixing-threshold-high", "--fixing_threshold_high", dest="fixing_threshold_high", type=float, default=None)
    parser.add_argument("--fixing-threshold-low", "--fixing_threshold_low", dest="fixing_threshold_low", type=float, default=None)
    parser.add_argument("--n-dives", "--n_dives", dest="n_dives", type=int, default=None)
    parser.add_argument("--parallel-dives", "--parallel_dives", dest="parallel_dives", action="store_true", default=None)
    parser.add_argument("--dive-workers", "--dive_workers", dest="dive_workers", type=int, default=None)
    parser.add_argument("--conpas-k0", "--conpas_k0", dest="conpas_k0", type=int, default=None)
    parser.add_argument("--conpas-k1", "--conpas_k1", dest="conpas_k1", type=int, default=None)
    parser.add_argument("--conpas-delta", "--conpas_delta", dest="conpas_delta", type=int, default=None)
    parser.add_argument("--conpas-benchmark-name", "--conpas_benchmark_name", dest="conpas_benchmark_name", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--instance-id", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--cart-symmetry-breaking", action="store_true", default=None,
                        help="Enable CART equal-capacity ordering constraints in EnerGNN inference. Disabled by default.")
    parser.add_argument("--quiet", action="store_true", default=None)
    parser.add_argument("--no-instance-results", action="store_true", default=None)
    parser.add_argument("--no-summary-tables", action="store_true", default=None)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    cfg = _load_config(args.config)
    eval_cfg = dict(cfg.get("evaluation", {}) or {})
    solver_cfg = dict(cfg.get("solver", {}) or {})
    output_cfg = dict(cfg.get("output", {}) or {})

    facility_count = int(args.facility_count if args.facility_count is not None else eval_cfg.get("facility_count", 16))
    default_train_dir, default_test_dir = _default_split_dirs(facility_count)
    checkpoint = args.checkpoint or _as_path(eval_cfg.get("checkpoint"))
    if checkpoint is None:
        raise ValueError("Pass --checkpoint or set evaluation.checkpoint in the config.")
    checkpoint = Path(checkpoint)

    test_dir = args.test_dir or _as_path(eval_cfg.get("test_dir")) or default_test_dir
    _ = args.train_dir or _as_path(eval_cfg.get("train_dir")) or default_train_dir
    static_config = args.static_config or _as_path(eval_cfg.get("static_config")) or (Path(test_dir) / "static_config.json")
    dat_path = args.dat_path or _as_path(eval_cfg.get("dat_path")) or Path("cart_dataset/Data_CART_og.dat")
    output_dir = args.output_dir or _as_path(output_cfg.get("output_dir")) or Path(f"results/cart_ood_m{facility_count}")
    force_off = args.force_off if args.force_off is not None else eval_cfg.get("force_off", [])
    force_on = args.force_on if args.force_on is not None else eval_cfg.get("force_on", [])
    max_facilities = args.max_facilities if args.max_facilities is not None else eval_cfg.get("max_facilities", [])
    force_off, force_on, max_facilities = ([int(v) for v in values] for values in (force_off, force_on, max_facilities))
    default_outages = [] if force_off or force_on or max_facilities else [1, 2]
    outage_sizes = args.outage_sizes if args.outage_sizes is not None else eval_cfg.get("outage_sizes", default_outages)
    outage_sizes = [int(v) for v in outage_sizes]

    active_off_inactive_on = _as_bool(
        args.active_off_inactive_on,
        default=_as_bool(eval_cfg.get("active_off_inactive_on"), False),
    )
    if not outage_sizes and not active_off_inactive_on and not (force_off or force_on or max_facilities):
        raise ValueError("Select facility states, maximum counts, or outage scenarios to evaluate.")
    cart_symmetry_breaking = _as_bool(
        args.cart_symmetry_breaking,
        default=_as_bool(eval_cfg.get("cart_symmetry_breaking"), False),
    )
    save_instance_results = not _as_bool(
        args.no_instance_results,
        default=not _as_bool(output_cfg.get("save_instance_results"), True),
    )
    save_summary_tables = not _as_bool(
        args.no_summary_tables,
        default=not _as_bool(output_cfg.get("save_summary_tables"), True),
    )

    max_pairs = args.max_pairs_per_instance
    if max_pairs is None:
        max_pairs = eval_cfg.get("max_pairs_per_instance")
    max_triples = args.max_triples_per_instance
    if max_triples is None:
        max_triples = eval_cfg.get("max_triples_per_instance")

    seed = int(args.seed if args.seed is not None else eval_cfg.get("seed", 0))
    time_limit = float(
        args.time_limit
        if args.time_limit is not None
        else solver_cfg.get("time_limit", PAPER_TIME_LIMIT)
    )
    threads = (
        args.threads
        if args.threads is not None
        else solver_cfg.get("threads", PAPER_SOLVER_THREADS)
    )
    threads = None if threads is None else int(threads)
    mip_gap = args.mip_gap if args.mip_gap is not None else solver_cfg.get("mip_gap", 0.0)
    mip_gap = None if mip_gap is None else float(mip_gap)
    min_k = int(args.min_k if args.min_k is not None else solver_cfg.get("min_k", 0))
    max_k = int(args.max_k if args.max_k is not None else solver_cfg.get("max_k", 0))
    if min_k < 0 or max_k < 0 or min_k > facility_count or (max_k > 0 and (max_k < min_k or max_k > facility_count)):
        raise ValueError("Invalid baseline facility-count bounds.")
    _generate_scenarios([0] * facility_count, outage_sizes=[], max_pairs_per_instance=None,
                        max_triples_per_instance=None, active_off_inactive_on=False,
                        seed=seed, instance_id=0, force_off=force_off, force_on=force_on,
                        max_facilities=max_facilities)
    if any(cap < min_k for cap in max_facilities) or (max_k > 0 and len(set(force_on)) > max_k):
        raise ValueError("Targeted facility constraints conflict with the baseline facility-count bounds.")
    reference_formulation = args.reference_formulation or eval_cfg.get("reference_formulation", "detailed")
    if reference_formulation != "detailed":
        raise ValueError("CART references require the detailed Gurobi formulation.")
    limit = int(args.limit if args.limit is not None else eval_cfg.get("limit", 0))
    num_shards = int(args.num_shards if args.num_shards is not None else eval_cfg.get("num_shards", 1))
    shard_index = int(args.shard_index if args.shard_index is not None else eval_cfg.get("shard_index", 0))
    quiet = _as_bool(args.quiet, default=_as_bool(eval_cfg.get("quiet"), False))

    run_cfg = _load_run_config(checkpoint)
    model_type = _resolve_model_type(args.model_type or eval_cfg.get("model_type") or run_cfg.get("model_type", "energnn"))
    dh = int(args.dh if args.dh is not None else run_cfg.get("dh", 128))
    nlayers = int(args.nlayers if args.nlayers is not None else run_cfg.get("nlayers", 2))
    dropout = float(args.dropout if args.dropout is not None else run_cfg.get("dropout", 0.1))
    ca_head_variant = str(args.ca_head_variant or run_cfg.get("ca_head_variant", "mlp32"))
    fixing_ratio = args.fixing_ratio if args.fixing_ratio is not None else eval_cfg.get("fixing_ratio")
    fixing_threshold_high = (
        args.fixing_threshold_high
        if args.fixing_threshold_high is not None
        else eval_cfg.get("fixing_threshold_high")
    )
    fixing_threshold_low = (
        args.fixing_threshold_low
        if args.fixing_threshold_low is not None
        else eval_cfg.get("fixing_threshold_low")
    )
    n_dives = args.n_dives if args.n_dives is not None else eval_cfg.get("n_dives")
    if model_type in {"predict_and_search", "gnn_fixing_mip"}:
        fixing_defaults = resolve_fixing_inference_defaults(
            "cart",
            model_type,
            fixing_ratio=fixing_ratio,
            fixing_threshold_high=fixing_threshold_high,
            fixing_threshold_low=fixing_threshold_low,
            n_dives=n_dives,
        )
        fixing_ratio = fixing_defaults["fixing_ratio"]
        fixing_threshold_high = fixing_defaults["fixing_threshold_high"]
        fixing_threshold_low = fixing_defaults["fixing_threshold_low"]
        n_dives = fixing_defaults["n_dives"]
    parallel_dives = _as_bool(args.parallel_dives, default=_as_bool(eval_cfg.get("parallel_dives"), False))
    dive_workers = args.dive_workers if args.dive_workers is not None else eval_cfg.get("dive_workers")
    dive_workers = None if dive_workers is None else int(dive_workers)
    conpas_k0 = args.conpas_k0 if args.conpas_k0 is not None else eval_cfg.get("conpas_k0")
    conpas_k1 = args.conpas_k1 if args.conpas_k1 is not None else eval_cfg.get("conpas_k1")
    conpas_delta = args.conpas_delta if args.conpas_delta is not None else eval_cfg.get("conpas_delta")
    conpas_k0 = None if conpas_k0 is None else int(conpas_k0)
    conpas_k1 = None if conpas_k1 is None else int(conpas_k1)
    conpas_delta = None if conpas_delta is None else int(conpas_delta)
    conpas_benchmark_name = str(
        args.conpas_benchmark_name
        if args.conpas_benchmark_name is not None
        else eval_cfg.get("conpas_benchmark_name", f"cart_m{facility_count}")
    )

    metas, pools, _, tmfe = _load_eval_data(
        "cart",
        Path(test_dir).as_posix(),
        None if static_config is None else Path(static_config).as_posix(),
        Path(dat_path).as_posix(),
    )
    ids = _select_instance_ids(
        metas,
        instance_id=args.instance_id,
        limit=limit,
        num_shards=num_shards,
        shard_index=shard_index,
    )
    model = _load_cart_model(
        checkpoint,
        model_type=model_type,
        metas=metas,
        tmfe=tmfe,
        dh=dh,
        nlayers=nlayers,
        dropout=dropout,
        ca_head_variant=ca_head_variant,
    )
    ds = InstancePoolDataset(metas, {k: [] for k in metas.keys()}, TMFE=tmfe)

    resolved = {
        "facility_count": facility_count,
        "model_type": model_type,
        "checkpoint": checkpoint.as_posix(),
        "test_dir": Path(test_dir).as_posix(),
        "static_config": None if static_config is None else Path(static_config).as_posix(),
        "outage_sizes": outage_sizes,
        "force_off": force_off,
        "force_on": force_on,
        "max_facilities": max_facilities,
        "active_off_inactive_on": active_off_inactive_on,
        "max_pairs_per_instance": None if max_pairs is None else int(max_pairs),
        "max_triples_per_instance": None if max_triples is None else int(max_triples),
        "seed": seed,
        "time_limit": time_limit,
        "threads": threads,
        "mip_gap": mip_gap,
        "min_k": min_k,
        "max_k": max_k,
        "cart_symmetry_breaking": cart_symmetry_breaking,
        "reference_formulation": reference_formulation,
        "fixing_ratio": fixing_ratio,
        "fixing_threshold_high": fixing_threshold_high,
        "fixing_threshold_low": fixing_threshold_low,
        "n_dives": n_dives,
        "parallel_dives": parallel_dives,
        "dive_workers": dive_workers,
        "conpas_k0": conpas_k0,
        "conpas_k1": conpas_k1,
        "conpas_delta": conpas_delta,
        "conpas_benchmark_name": conpas_benchmark_name,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "config_resolved.json", resolved)

    results_path = output_dir / "scenario_results.jsonl"
    if save_instance_results:
        results_path.write_text("", encoding="utf-8")

    rows: List[dict] = []
    skipped: List[dict] = []
    started = time.perf_counter()

    for inst_pos, inst_id in enumerate(ids, start=1):
        meta = metas[int(inst_id)]
        if len(meta.M) != facility_count:
            raise ValueError(
                f"facility-count={facility_count} but instance {inst_id} has {len(meta.M)} facilities."
            )

        original = _solve_cart_details(
            meta,
            min_k=min_k,
            max_k=max_k,
            time_limit=time_limit,
            threads=threads,
            mip_gap=mip_gap,
            formulation=reference_formulation,
        )
        original_mask = original.get("configuration")
        if original_mask is None or not original.get("proven_optimal"):
            skipped.append(
                {
                    "instance_id": int(inst_id),
                    "reason": "original_reference_no_solution" if original_mask is None
                    else "original_reference_not_proven_optimal",
                    "original_status": original.get("status"),
                }
            )
            continue

        active_indices = [idx for idx, bit in enumerate(original_mask) if int(bit) == 1]
        scenarios = _generate_scenarios(
            original_mask,
            outage_sizes=outage_sizes,
            max_pairs_per_instance=None if max_pairs is None else int(max_pairs),
            max_triples_per_instance=None if max_triples is None else int(max_triples),
            active_off_inactive_on=active_off_inactive_on,
            seed=seed,
            instance_id=int(inst_id),
            force_off=force_off,
            force_on=force_on,
            max_facilities=max_facilities,
        )
        graph, _ = ds._build_graph(int(inst_id))

        if not quiet:
            print(
                f"[instance] {inst_pos}/{len(ids)} id={inst_id} "
                f"active={len(active_indices)} scenarios={len(scenarios)}"
            )

        for scenario_idx, scenario in enumerate(scenarios):
            scenario_max_k = _scenario_max_k(scenario, max_k)
            off_indices = tuple(int(v) for v in scenario["off_indices"])
            on_indices = tuple(int(v) for v in scenario["on_indices"])
            fixed = _fixed_assignments(off_indices, on_indices)

            reference = _solve_cart_details(
                meta,
                min_k=min_k,
                max_k=scenario_max_k,
                time_limit=time_limit,
                threads=threads,
                mip_gap=mip_gap,
                fixed_assignments=fixed,
                formulation=reference_formulation,
            )
            reference_mask = reference.get("configuration")
            reference_obj = reference.get("objective")
            if reference_mask is None or reference_obj is None:
                skipped.append(
                    {
                        "instance_id": int(inst_id),
                        "scenario_idx": int(scenario_idx),
                        "scenario_type": scenario["scenario_type"],
                        "outage_size": int(scenario["outage_size"]),
                        "min_facilities": min_k,
                        "max_facilities": scenario_max_k,
                        "deactivated_facilities": _facility_names(meta, off_indices),
                        "forced_active_facilities": _facility_names(meta, on_indices),
                        "reason": "constrained_reference_no_solution",
                        "reference_status": reference.get("status"),
                    }
                )
                continue

            inference_start = time.perf_counter()
            gnn_mask, gnn_mip_energy, gnn_true_energy, gnn_status, extra = solve_instance_simple(
                model,
                graph,
                meta,
                problem="cart",
                min_k=min_k,
                max_k=scenario_max_k,
                time_limit=time_limit,
                solver_threads=threads,
                mip_gap=mip_gap,
                enforce_problem_constraints=True,
                fixed_assignments=fixed,
                cart_symmetry_breaking=cart_symmetry_breaking,
                fixing_ratio=fixing_ratio,
                fixing_threshold_high=fixing_threshold_high,
                fixing_threshold_low=fixing_threshold_low,
                n_dives=n_dives,
                parallel_dives=parallel_dives,
                dive_workers=dive_workers,
                conpas_k0=conpas_k0,
                conpas_k1=conpas_k1,
                conpas_delta=conpas_delta,
                conpas_benchmark_name=conpas_benchmark_name,
            )
            total_inference_time = time.perf_counter() - inference_start
            fixed_ok = _validate_mask_against_fixed(gnn_mask, fixed)
            constraints_ok = _validate_mask_constraints(gnn_mask, fixed, min_k, scenario_max_k)

            gnn_eval = None
            gnn_obj = None
            if gnn_mask is not None and constraints_ok:
                gnn_eval = _solve_cart_details(
                    meta,
                    min_k=min_k,
                    max_k=scenario_max_k,
                    time_limit=time_limit,
                    threads=threads,
                    mip_gap=mip_gap,
                    fixed_mask=gnn_mask,
                    fixed_assignments=fixed,
                    formulation=reference_formulation,
                )
                gnn_obj = gnn_eval.get("objective")

            reference_proven = bool(reference.get("proven_optimal"))
            gap_exact = (
                _relative_opt_gap(gnn_obj, reference_obj, sense="min")
                if reference_proven and gnn_obj is not None
                else None
            )
            gap_best_known = (
                _relative_opt_gap(gnn_obj, reference_obj, sense="min")
                if gnn_obj is not None and reference_obj is not None
                else None
            )
            exact_match = _mask_key(gnn_mask) == _mask_key(reference_mask)
            d_shift = _hamming(original_mask, reference_mask)
            d_energnn = _hamming(original_mask, gnn_mask)
            d_error = _hamming(reference_mask, gnn_mask)
            rank, rank_total, rank_feasible_count = None, None, None

            row = {
                "instance_id": int(inst_id),
                "model_type": model_type,
                "num_facilities": int(len(meta.M)),
                "scenario_idx": int(scenario_idx),
                "scenario_type": scenario["scenario_type"],
                "outage_size": int(scenario["outage_size"]),
                "min_facilities": min_k,
                "max_facilities": scenario_max_k,
                "deactivated_facilities": _facility_names(meta, off_indices),
                "forced_active_facilities": _facility_names(meta, on_indices),
                "deactivated_indices": list(off_indices),
                "forced_active_indices": list(on_indices),
                "original_optimal_configuration": list(original_mask),
                "original_active_facilities": _facility_names(meta, active_indices),
                "original_optimal_objective": original.get("objective"),
                "original_status": original.get("status"),
                "original_proven_optimal": bool(original.get("proven_optimal")),
                "original_runtime_sec": original.get("runtime"),
                "reference_configuration": list(reference_mask),
                "reference_objective": reference_obj,
                "reference_status": reference.get("status"),
                "reference_runtime_sec": reference.get("runtime"),
                "reference_bound": reference.get("bound"),
                "reference_solver_gap": reference.get("solver_gap"),
                "reference_proven_optimal": reference_proven,
                "energnn_configuration": None if gnn_mask is None else list(gnn_mask),
                "energnn_status": gnn_status,
                "energnn_energy": gnn_true_energy if gnn_true_energy is not None else gnn_mip_energy,
                "energnn_mip_energy": gnn_mip_energy,
                "energnn_facility_location_objective": gnn_obj,
                "energnn_feasible": bool(gnn_obj is not None and fixed_ok),
                "energnn_fixed_constraints_satisfied": bool(fixed_ok),
                "energnn_constraints_satisfied": bool(constraints_ok),
                "method_configuration": None if gnn_mask is None else list(gnn_mask),
                "method_status": gnn_status,
                "method_objective": gnn_obj,
                "method_feasible": bool(gnn_obj is not None and fixed_ok),
                "method_fixed_constraints_satisfied": bool(fixed_ok),
                "encoder_time_sec": extra.get("encoder_time_sec"),
                "neural_mip_solve_time_sec": extra.get("solver_runtime_sec"),
                "energnn_total_inference_time_sec": float(total_inference_time),
                "method_total_inference_time_sec": float(total_inference_time),
                "num_fixed_variables": extra.get("num_fixed_variables"),
                "effective_fixed_variables": extra.get("effective_fixed_variables"),
                "fixing_ratio_used": extra.get("fixing_ratio_used"),
                "fixing_threshold_high": extra.get("fixing_threshold_high"),
                "fixing_threshold_low": extra.get("fixing_threshold_low"),
                "n_dives": extra.get("n_dives"),
                "conpas_k0": extra.get("conpas_k0"),
                "conpas_k1": extra.get("conpas_k1"),
                "conpas_delta": extra.get("conpas_delta"),
                "allowed_disagreements": extra.get("allowed_disagreements"),
                "gap_to_exact_constrained_optimum": gap_exact,
                "gap_to_reference_best_known": gap_best_known,
                "gap_to_exact_constrained_optimum_pct": None if gap_exact is None else 100.0 * float(gap_exact),
                "exact_configuration_match": bool(exact_match),
                "within_1pct": bool(gap_exact is not None and gap_exact <= 0.01 + 1e-12),
                "within_5pct": bool(gap_exact is not None and gap_exact <= 0.05 + 1e-12),
                "within_10pct": bool(gap_exact is not None and gap_exact <= 0.10 + 1e-12),
                "d_shift": d_shift,
                "d_energnn": d_energnn,
                "d_error": d_error,
                "constrained_optimum_original_objective_rank": rank,
                "constrained_optimum_original_objective_rank_total": rank_total,
                "constrained_optimum_original_objective_rank_feasible_count": rank_feasible_count,
            }
            rows.append(row)
            if save_instance_results:
                _append_jsonl(results_path, row)

            if not quiet:
                gap_text = "N/A" if gap_exact is None else f"{100.0 * gap_exact:.3f}%"
                print(
                    f"  [scenario] idx={scenario_idx} type={scenario['scenario_type']} "
                    f"off={row['deactivated_facilities']} on={row['forced_active_facilities']} "
                    f"max_facilities={scenario_max_k or 'unlimited'} "
                    f"ref={reference.get('status')} gnn={gnn_status} gap={gap_text}"
                )

    summary = _build_summary(rows, skipped, resolved)
    summary["wallclock_sec"] = float(time.perf_counter() - started)
    _write_json(output_dir / "summary.json", summary)
    if save_instance_results:
        _write_csv(output_dir / "scenario_results.csv", rows)
    if save_summary_tables:
        _write_csv(output_dir / "summary_by_constraint.csv", summary["summary_by_constraint"])
        _write_csv(output_dir / "summary_by_outage_size.csv", summary["summary_by_outage_size"])
        _write_csv(output_dir / "summary_by_shift_distance.csv", summary["summary_by_shift_distance"])

    if not quiet:
        counts = summary["counts"]
        print(
            f"[summary] scenarios={counts['evaluated_scenarios']} "
            f"skipped={counts['skipped_scenarios']} "
            f"reference_optimal={counts['reference_optimal']} "
            f"{model_type}_feasible={counts['method_feasible']}"
        )
        print(f"[output] {output_dir.as_posix()}")


if __name__ == "__main__":
    main()
