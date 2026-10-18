"""Evaluate CA accepted-bid outage robustness for trained EnerGNN checkpoints.

For each held-out combinatorial-auction instance, this script solves the
original winner-determination MIP, removes bids accepted in that optimum,
re-solves the constrained MIP exactly, and runs EnerGNN with the same fixed
bid constraints in its learned-energy MIP.
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
    _eval_ca_mask,
    _load_eval_data,
    _relative_opt_gap,
    _resolve_model_type,
    _solve_true_ca_milp,
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
    if value is None or str(value).strip() == "" or str(value).strip().lower() == "null":
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


def _load_run_config(checkpoint: Path) -> dict:
    path = checkpoint.parent / "run_config.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _load_ca_model(
    checkpoint: Path,
    *,
    model_type: str,
    metas: Dict[int, object],
    tmfe: int,
    dh: int,
    nlayers: int,
    dropout: float,
    ca_readout_variant: str,
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
        "problem": "ca",
        "ca_readout_variant": str(ca_readout_variant),
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
    if a is None or b is None or len(a) != len(b):
        return None
    return sum(1 for x, y in zip(a, b) if int(x) != int(y))


def _replacement_count(original: Optional[Sequence[int]], solution: Optional[Sequence[int]]) -> Optional[int]:
    if original is None or solution is None or len(original) != len(solution):
        return None
    return sum(1 for orig, sol in zip(original, solution) if int(orig) == 0 and int(sol) == 1)


def _bid_names(meta, indices: Sequence[int]) -> List[str]:
    return [str(meta.M[int(idx)]) for idx in indices]


def _bid_values(meta, indices: Sequence[int]) -> List[float]:
    costs = np.asarray(meta.globals["costs"], dtype=float).ravel()
    return [float(costs[int(idx)]) for idx in indices]


def _fixed_assignments(off_indices: Sequence[int], on_indices: Sequence[int]) -> Dict[int, int]:
    forced: Dict[int, int] = {int(idx): 0 for idx in off_indices}
    for idx in on_indices:
        idx = int(idx)
        if idx in forced and forced[idx] != 1:
            raise ValueError(f"Conflicting outage scenario for bid index {idx}.")
        forced[idx] = 1
    return forced


def _validate_mask_against_fixed(mask: Optional[Sequence[int]], fixed: Dict[int, int]) -> bool:
    if mask is None:
        return False
    for idx, val in fixed.items():
        if int(mask[int(idx)]) != int(val):
            return False
    return True


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
    rng = np.random.default_rng(int(seed) + 4099 * int(instance_id) + 100003 * int(outage_size))
    picked = sorted(int(i) for i in rng.choice(len(combos), size=int(max_count), replace=False).tolist())
    return [combos[i] for i in picked]


def _random_winner_subsets(
    accepted: Sequence[int],
    *,
    k: int,
    n_samples: int,
    seed: int,
    instance_id: int,
) -> List[Tuple[int, ...]]:
    accepted = sorted(int(v) for v in accepted)
    if k <= 0 or k > len(accepted) or n_samples <= 0:
        return []
    total_possible = math.comb(len(accepted), int(k))
    target = min(int(n_samples), int(total_possible))
    rng = np.random.default_rng(int(seed) + 7919 * int(instance_id) + 104729 * int(k))
    subsets: set[Tuple[int, ...]] = set()
    attempts = 0
    max_attempts = max(100, 50 * target)
    while len(subsets) < target and attempts < max_attempts:
        attempts += 1
        subset = tuple(sorted(int(v) for v in rng.choice(accepted, size=int(k), replace=False).tolist()))
        subsets.add(subset)
    if len(subsets) < target and total_possible <= 100000:
        for subset in itertools.combinations(accepted, int(k)):
            subsets.add(tuple(int(v) for v in subset))
            if len(subsets) >= target:
                break
    return sorted(subsets)


def _top_value_indices(meta, indices: Sequence[int], k: int) -> Optional[Tuple[int, ...]]:
    if k <= 0 or k > len(indices):
        return None
    costs = np.asarray(meta.globals["costs"], dtype=float).ravel()
    ordered = sorted((int(idx) for idx in indices), key=lambda idx: (-float(costs[idx]), int(idx)))
    return tuple(ordered[: int(k)])


def _top_value_compatible_indices(meta, indices: Sequence[int], k: int) -> Optional[Tuple[int, ...]]:
    if k <= 0 or k > len(indices):
        return None
    costs = np.asarray(meta.globals["costs"], dtype=float).ravel()
    A_csr = meta.globals["A_csr"]
    load = np.zeros(int(A_csr.shape[0]), dtype=np.int16)
    selected: List[int] = []
    for idx in sorted((int(v) for v in indices), key=lambda j: (-float(costs[j]), int(j))):
        col = A_csr.getcol(int(idx))
        used_rows = col.indices
        if used_rows.size and np.any(load[used_rows] > 0):
            continue
        selected.append(int(idx))
        if used_rows.size:
            load[used_rows] += 1
        if len(selected) >= int(k):
            return tuple(selected)
    return None


def _generate_scenarios(
    meta,
    original_mask: Sequence[int],
    *,
    removal_sizes: Sequence[int],
    random_scenarios_per_k: int,
    force_rejected_sizes: Sequence[int],
    enable_random_winner_removal: bool,
    enable_high_value_winner_removal: bool,
    enable_high_value_rejected_force: bool,
    seed: int,
    instance_id: int,
) -> Tuple[List[dict], List[dict]]:
    accepted = [idx for idx, bit in enumerate(original_mask) if int(bit) == 1]
    rejected = [idx for idx, bit in enumerate(original_mask) if int(bit) == 0]
    scenarios: List[dict] = []
    skipped: List[dict] = []

    for k in sorted({int(v) for v in removal_sizes}):
        if k <= 0:
            continue
        if k > len(accepted):
            skipped.append(
                {
                    "instance_id": int(instance_id),
                    "scenario_type": "winner_removal",
                    "scenario_k": int(k),
                    "reason": "not_enough_original_winners",
                    "original_winner_count": int(len(accepted)),
                }
            )
            continue

        if enable_random_winner_removal:
            subsets = _random_winner_subsets(
                accepted,
                k=int(k),
                n_samples=int(random_scenarios_per_k),
                seed=seed,
                instance_id=instance_id,
            )
            for sample_idx, subset in enumerate(subsets):
                scenarios.append(
                    {
                        "scenario_type": "random_winner_removal",
                        "scenario_family": "random_winner_removal",
                        "scenario_k": int(k),
                        "random_sample_index": int(sample_idx),
                        "scenario_seed": int(seed) + 7919 * int(instance_id) + 104729 * int(k),
                        "outage_size": int(k),
                        "off_indices": tuple(int(i) for i in subset),
                        "on_indices": tuple(),
                    }
                )

        if enable_high_value_winner_removal:
            top_winners = _top_value_indices(meta, accepted, int(k))
            if top_winners is not None:
                scenarios.append(
                    {
                        "scenario_type": "high_value_winner_removal",
                        "scenario_family": "high_value_winner_removal",
                        "scenario_k": int(k),
                        "random_sample_index": None,
                        "scenario_seed": None,
                        "outage_size": int(k),
                        "off_indices": top_winners,
                        "on_indices": tuple(),
                    }
                )

    for k in sorted({int(v) for v in force_rejected_sizes}):
        if k <= 0 or not enable_high_value_rejected_force:
            continue
        forced = _top_value_compatible_indices(meta, rejected, int(k))
        if forced is None:
            skipped.append(
                {
                    "instance_id": int(instance_id),
                    "scenario_type": "high_value_rejected_force_on",
                    "scenario_k": int(k),
                    "reason": "not_enough_compatible_rejected_bids",
                    "original_rejected_count": int(len(rejected)),
                }
            )
            continue
        scenarios.append(
            {
                "scenario_type": "high_value_rejected_force_on",
                "scenario_family": "high_value_rejected_force_on",
                "scenario_k": int(k),
                "random_sample_index": None,
                "scenario_seed": None,
                "outage_size": 0,
                "off_indices": tuple(),
                "on_indices": forced,
            }
        )

    return scenarios, skipped


def _generate_legacy_combination_scenarios(
    original_mask: Sequence[int],
    *,
    outage_sizes: Sequence[int],
    max_singles_per_instance: Optional[int],
    max_pairs_per_instance: Optional[int],
    max_triples_per_instance: Optional[int],
    accepted_off_rejected_on: bool,
    seed: int,
    instance_id: int,
) -> List[dict]:
    accepted = [idx for idx, bit in enumerate(original_mask) if int(bit) == 1]
    rejected = [idx for idx, bit in enumerate(original_mask) if int(bit) == 0]
    scenarios: List[dict] = []

    for size in sorted({int(v) for v in outage_sizes}):
        if size <= 0 or size > len(accepted):
            continue
        combos = list(itertools.combinations(accepted, size))
        if size == 1:
            combos = _sample_combinations(
                combos,
                max_count=max_singles_per_instance,
                seed=seed,
                instance_id=instance_id,
                outage_size=size,
            )
        elif size == 2:
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
                    "scenario_type": "accepted_off",
                    "scenario_family": "legacy_accepted_off",
                    "scenario_k": int(size),
                    "random_sample_index": None,
                    "scenario_seed": None,
                    "outage_size": int(size),
                    "off_indices": tuple(int(i) for i in combo),
                    "on_indices": tuple(),
                }
            )

    if accepted_off_rejected_on:
        for off_idx in accepted:
            for on_idx in rejected:
                scenarios.append(
                    {
                        "scenario_type": "accepted_off_rejected_on",
                        "scenario_family": "legacy_accepted_off_rejected_on",
                        "scenario_k": 1,
                        "random_sample_index": None,
                        "scenario_seed": None,
                        "outage_size": 1,
                        "off_indices": (int(off_idx),),
                        "on_indices": (int(on_idx),),
                    }
                )
    return scenarios


def _solve_ca_details(
    meta,
    *,
    time_limit: float,
    threads: Optional[int],
    mip_gap: Optional[float],
    ca_k_max: Optional[int],
    fixed_assignments: Optional[Dict[int, int]] = None,
) -> dict:
    return _solve_true_ca_milp(
        meta,
        time_limit=float(time_limit),
        fixed_assignments=fixed_assignments,
        k_max=ca_k_max,
        solver_threads=threads,
        mip_gap=mip_gap,
        return_details=True,
    )


def _evaluate_ca_solution(
    meta,
    mask: Optional[Sequence[int]],
    *,
    fixed: Dict[int, int],
    ca_k_max: Optional[int],
) -> dict:
    fixed_ok = _validate_mask_against_fixed(mask, fixed)
    if mask is None:
        return {
            "objective": None,
            "feasible": False,
            "fixed_ok": False,
            "num_conflicts": None,
            "num_conflicted_items": None,
            "selected_count": None,
            "cardinality_ok": None,
        }
    eval_row = _eval_ca_mask(meta, mask, k_max=ca_k_max)
    feasible = bool(eval_row["feasible"] and fixed_ok)
    return {
        "objective": float(eval_row["objective"]) if feasible else None,
        "raw_objective": float(eval_row["objective"]),
        "feasible": feasible,
        "fixed_ok": bool(fixed_ok),
        "num_conflicts": int(eval_row["num_conflicts"]),
        "num_conflicted_items": int(eval_row["num_conflicted_items"]),
        "selected_count": int(eval_row["selected_count"]),
        "cardinality_ok": bool(eval_row["cardinality_ok"]),
    }


def _summarise_group(rows: List[dict]) -> dict:
    n = len(rows)
    gaps = [float(r["gap_to_exact_constrained_optimum"]) for r in rows if r.get("gap_to_exact_constrained_optimum") is not None]
    d_shift = [int(r["d_shift"]) for r in rows if r.get("d_shift") is not None]
    d_error = [int(r["d_error"]) for r in rows if r.get("d_error") is not None]
    reference_replacements = [
        int(r["reference_replacement_bid_count"])
        for r in rows
        if r.get("reference_replacement_bid_count") is not None
    ]
    energnn_replacements = [
        int(r["energnn_replacement_bid_count"])
        for r in rows
        if r.get("energnn_replacement_bid_count") is not None
    ]
    method_replacements = [
        int(r["method_replacement_bid_count"])
        for r in rows
        if r.get("method_replacement_bid_count") is not None
    ]
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
        "mean_reference_replacement_bid_count": None if not reference_replacements else float(sum(reference_replacements) / len(reference_replacements)),
        "mean_energnn_replacement_bid_count": None if not energnn_replacements else float(sum(energnn_replacements) / len(energnn_replacements)),
        "mean_method_replacement_bid_count": None if not method_replacements else float(sum(method_replacements) / len(method_replacements)),
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
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
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
    by_size = []
    for size in sorted({int(row["outage_size"]) for row in rows}):
        group = [row for row in rows if int(row["outage_size"]) == size]
        by_size.append({"outage_size": size, **_summarise_group(group)})

    by_scenario_type = []
    for scenario_type in sorted({str(row.get("scenario_type", "")) for row in rows}):
        group = [row for row in rows if str(row.get("scenario_type", "")) == scenario_type]
        by_scenario_type.append({"scenario_type": scenario_type, **_summarise_group(group)})

    by_scenario_k = []
    scenario_keys = sorted(
        {
            (str(row.get("scenario_type", "")), int(row.get("scenario_k", row.get("outage_size", 0)) or 0))
            for row in rows
        },
        key=lambda item: (item[0], item[1]),
    )
    for scenario_type, scenario_k in scenario_keys:
        group = [
            row
            for row in rows
            if str(row.get("scenario_type", "")) == scenario_type
            and int(row.get("scenario_k", row.get("outage_size", 0)) or 0) == scenario_k
        ]
        by_scenario_k.append(
            {
                "scenario_type": scenario_type,
                "scenario_k": int(scenario_k),
                **_summarise_group(group),
            }
        )

    by_shift = []
    bucket_order = {"0": 0, "1": 1, "2": 2, "3": 3, ">=4": 4, "unknown": 5}
    for bucket in sorted({_shift_bucket(row.get("d_shift")) for row in rows}, key=lambda x: bucket_order.get(x, 99)):
        group = [row for row in rows if _shift_bucket(row.get("d_shift")) == bucket]
        by_shift.append({"d_shift_bucket": bucket, **_summarise_group(group)})

    return {
        "evaluation": {
            "name": "ca_bid_outage_ood",
            "model_type": resolved["model_type"],
            "checkpoint": resolved["checkpoint"],
            "data_dir": resolved["data_dir"],
            "removal_sizes": resolved["removal_sizes"],
            "random_scenarios_per_k": resolved["random_scenarios_per_k"],
            "force_rejected_sizes": resolved["force_rejected_sizes"],
            "enable_random_winner_removal": resolved["enable_random_winner_removal"],
            "enable_high_value_winner_removal": resolved["enable_high_value_winner_removal"],
            "enable_high_value_rejected_force": resolved["enable_high_value_rejected_force"],
            "legacy_outage_sizes": resolved["legacy_outage_sizes"],
            "accepted_off_rejected_on": resolved["accepted_off_rejected_on"],
            "max_singles_per_instance": resolved["max_singles_per_instance"],
            "max_pairs_per_instance": resolved["max_pairs_per_instance"],
            "max_triples_per_instance": resolved["max_triples_per_instance"],
            "seed": resolved["seed"],
        },
        "solver": {
            "time_limit": resolved["time_limit"],
            "threads": resolved["threads"],
            "mip_gap": resolved["mip_gap"],
            "ca_k_max": resolved["ca_k_max"],
        },
        "counts": {
            "evaluated_scenarios": len(rows),
            "skipped_scenarios": len(skipped),
            "reference_optimal": sum(1 for row in rows if bool(row.get("reference_proven_optimal"))),
            "energnn_feasible": sum(1 for row in rows if bool(row.get("energnn_feasible"))),
            "method_feasible": sum(1 for row in rows if bool(row.get("method_feasible"))),
        },
        "summary_by_outage_size": by_size,
        "summary_by_scenario_type": by_scenario_type,
        "summary_by_scenario_k": by_scenario_k,
        "summary_by_shift_distance": by_shift,
        "skipped": skipped,
    }


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CA accepted-bid outage OOD robustness.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--model-type", "--model_type", dest="model_type", type=str, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--data-dir", "--data_dir", dest="data_dir", type=Path, default=None)
    parser.add_argument("--outage-sizes", "--outage_sizes", dest="outage_sizes", type=int, nargs="*", default=None)
    parser.add_argument("--removal-sizes", "--removal_sizes", dest="removal_sizes", type=int, nargs="*", default=None)
    parser.add_argument("--random-scenarios-per-k", "--random_scenarios_per_k", dest="random_scenarios_per_k", type=int, default=None)
    parser.add_argument("--force-rejected-sizes", "--force_rejected_sizes", dest="force_rejected_sizes", type=int, nargs="*", default=None)
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", type=Path, default=None)
    parser.add_argument("--accepted-off-rejected-on", action="store_true", default=None)
    parser.add_argument("--disable-random-winner-removal", action="store_true", default=None)
    parser.add_argument("--disable-high-value-winner-removal", action="store_true", default=None)
    parser.add_argument("--disable-high-value-rejected-force", action="store_true", default=None)
    parser.add_argument("--max-singles-per-instance", type=int, default=None)
    parser.add_argument("--max-pairs-per-instance", type=int, default=None)
    parser.add_argument("--max-triples-per-instance", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--time-limit", "--time_limit", dest="time_limit", type=float, default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--mip-gap", "--mip_gap", dest="mip_gap", type=float, default=None)
    parser.add_argument("--ca-k-max", "--ca_k_max", dest="ca_k_max", type=int, default=None)
    parser.add_argument("--dh", type=int, default=None)
    parser.add_argument("--nlayers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--ca-readout-variant", "--ca_readout_variant", dest="ca_readout_variant", type=str, default=None)
    parser.add_argument("--ca-head-variant", "--ca_head_variant", dest="ca_head_variant", type=str, default=None)
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
    parser.add_argument("--instance-id", "--instance_id", dest="instance_id", type=int, default=None)
    parser.add_argument("--num-shards", "--num_shards", dest="num_shards", type=int, default=None)
    parser.add_argument("--shard-index", "--shard_index", dest="shard_index", type=int, default=None)
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

    checkpoint = args.checkpoint or _as_path(eval_cfg.get("checkpoint"))
    if checkpoint is None:
        raise ValueError("Pass --checkpoint or set evaluation.checkpoint in the config.")
    checkpoint = Path(checkpoint)

    data_dir = (
        args.data_dir
        or _as_path(eval_cfg.get("data_dir"))
        or _as_path(eval_cfg.get("test_dir"))
        or Path("data/CA/175/test")
    )
    output_dir = args.output_dir or _as_path(output_cfg.get("output_dir")) or Path("results/ca_bid_outage_ood")
    legacy_outage_sizes = args.outage_sizes if args.outage_sizes is not None else eval_cfg.get("outage_sizes")
    legacy_outage_sizes = None if legacy_outage_sizes is None else [int(v) for v in legacy_outage_sizes]
    removal_sizes = args.removal_sizes if args.removal_sizes is not None else eval_cfg.get("removal_sizes")
    if removal_sizes is None and legacy_outage_sizes is None:
        removal_sizes = [5, 10, 15]
    removal_sizes = [] if removal_sizes is None else [int(v) for v in removal_sizes]
    random_scenarios_per_k = int(
        args.random_scenarios_per_k
        if args.random_scenarios_per_k is not None
        else eval_cfg.get("random_scenarios_per_k", 10)
    )
    force_rejected_sizes = (
        args.force_rejected_sizes
        if args.force_rejected_sizes is not None
        else eval_cfg.get("force_rejected_sizes", [1, 3, 5])
    )
    force_rejected_sizes = [] if force_rejected_sizes is None else [int(v) for v in force_rejected_sizes]
    enable_random_winner_removal = _as_bool(eval_cfg.get("enable_random_winner_removal"), True)
    if bool(args.disable_random_winner_removal):
        enable_random_winner_removal = False
    enable_high_value_winner_removal = _as_bool(eval_cfg.get("enable_high_value_winner_removal"), True)
    if bool(args.disable_high_value_winner_removal):
        enable_high_value_winner_removal = False
    enable_high_value_rejected_force = _as_bool(eval_cfg.get("enable_high_value_rejected_force"), True)
    if bool(args.disable_high_value_rejected_force):
        enable_high_value_rejected_force = False

    accepted_off_rejected_on = _as_bool(
        args.accepted_off_rejected_on,
        default=_as_bool(eval_cfg.get("accepted_off_rejected_on"), False),
    )
    save_instance_results = not _as_bool(
        args.no_instance_results,
        default=not _as_bool(output_cfg.get("save_instance_results"), True),
    )
    save_summary_tables = not _as_bool(
        args.no_summary_tables,
        default=not _as_bool(output_cfg.get("save_summary_tables"), True),
    )

    max_singles = args.max_singles_per_instance
    if max_singles is None:
        max_singles = eval_cfg.get("max_singles_per_instance")
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
    ca_k_max = args.ca_k_max if args.ca_k_max is not None else solver_cfg.get("ca_k_max")
    ca_k_max = None if ca_k_max is None else int(ca_k_max)
    limit = int(args.limit if args.limit is not None else eval_cfg.get("limit", 0))
    num_shards = int(args.num_shards if args.num_shards is not None else eval_cfg.get("num_shards", 1))
    shard_index = int(args.shard_index if args.shard_index is not None else eval_cfg.get("shard_index", 0))
    quiet = _as_bool(args.quiet, default=_as_bool(eval_cfg.get("quiet"), False))

    run_cfg = _load_run_config(checkpoint)
    model_type = _resolve_model_type(args.model_type or eval_cfg.get("model_type") or run_cfg.get("model_type", "energnn"))
    dh = int(args.dh if args.dh is not None else run_cfg.get("dh", 128))
    nlayers = int(args.nlayers if args.nlayers is not None else run_cfg.get("nlayers", 2))
    dropout = float(args.dropout if args.dropout is not None else run_cfg.get("dropout", 0.1))
    ca_readout_variant = str(args.ca_readout_variant or run_cfg.get("ca_readout_variant", "sum_only_plus"))
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
            "ca",
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
        else eval_cfg.get("conpas_benchmark_name", "CA")
    )

    metas, _, _, tmfe = _load_eval_data("ca", Path(data_dir).as_posix(), None, None)
    ids = _select_instance_ids(
        metas,
        instance_id=args.instance_id,
        limit=limit,
        num_shards=num_shards,
        shard_index=shard_index,
    )
    model = _load_ca_model(
        checkpoint,
        model_type=model_type,
        metas=metas,
        tmfe=tmfe,
        dh=dh,
        nlayers=nlayers,
        dropout=dropout,
        ca_readout_variant=ca_readout_variant,
        ca_head_variant=ca_head_variant,
    )
    ds = InstancePoolDataset(metas, {k: [] for k in metas.keys()}, TMFE=tmfe)

    resolved = {
        "model_type": model_type,
        "checkpoint": checkpoint.as_posix(),
        "data_dir": Path(data_dir).as_posix(),
        "removal_sizes": removal_sizes,
        "random_scenarios_per_k": random_scenarios_per_k,
        "force_rejected_sizes": force_rejected_sizes,
        "enable_random_winner_removal": enable_random_winner_removal,
        "enable_high_value_winner_removal": enable_high_value_winner_removal,
        "enable_high_value_rejected_force": enable_high_value_rejected_force,
        "legacy_outage_sizes": legacy_outage_sizes,
        "accepted_off_rejected_on": accepted_off_rejected_on,
        "max_singles_per_instance": None if max_singles is None else int(max_singles),
        "max_pairs_per_instance": None if max_pairs is None else int(max_pairs),
        "max_triples_per_instance": None if max_triples is None else int(max_triples),
        "seed": seed,
        "time_limit": time_limit,
        "threads": threads,
        "mip_gap": mip_gap,
        "ca_k_max": ca_k_max,
        "ca_readout_variant": ca_readout_variant,
        "ca_head_variant": ca_head_variant,
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
        original = _solve_ca_details(
            meta,
            time_limit=time_limit,
            threads=threads,
            mip_gap=mip_gap,
            ca_k_max=ca_k_max,
        )
        original_mask = original.get("configuration")
        if original_mask is None:
            skipped.append(
                {
                    "instance_id": int(inst_id),
                    "reason": "original_reference_no_solution",
                    "original_status": original.get("status"),
                }
            )
            continue

        accepted_indices = [idx for idx, bit in enumerate(original_mask) if int(bit) == 1]
        if removal_sizes:
            scenarios, scenario_skips = _generate_scenarios(
                meta,
                original_mask,
                removal_sizes=removal_sizes,
                random_scenarios_per_k=random_scenarios_per_k,
                force_rejected_sizes=force_rejected_sizes,
                enable_random_winner_removal=enable_random_winner_removal,
                enable_high_value_winner_removal=enable_high_value_winner_removal,
                enable_high_value_rejected_force=enable_high_value_rejected_force,
                seed=seed,
                instance_id=int(inst_id),
            )
            skipped.extend(scenario_skips)
        else:
            scenarios = _generate_legacy_combination_scenarios(
                original_mask,
                outage_sizes=[] if legacy_outage_sizes is None else legacy_outage_sizes,
                max_singles_per_instance=None if max_singles is None else int(max_singles),
                max_pairs_per_instance=None if max_pairs is None else int(max_pairs),
                max_triples_per_instance=None if max_triples is None else int(max_triples),
                accepted_off_rejected_on=accepted_off_rejected_on,
                seed=seed,
                instance_id=int(inst_id),
            )
        graph, _ = ds._build_graph(int(inst_id))

        if not quiet:
            print(
                f"[instance] {inst_pos}/{len(ids)} id={inst_id} "
                f"accepted={len(accepted_indices)} scenarios={len(scenarios)}"
            )

        for scenario_idx, scenario in enumerate(scenarios):
            off_indices = tuple(int(v) for v in scenario["off_indices"])
            on_indices = tuple(int(v) for v in scenario["on_indices"])
            fixed = _fixed_assignments(off_indices, on_indices)

            reference = _solve_ca_details(
                meta,
                time_limit=time_limit,
                threads=threads,
                mip_gap=mip_gap,
                ca_k_max=ca_k_max,
                fixed_assignments=fixed,
            )
            reference_mask = reference.get("configuration")
            reference_obj = reference.get("objective")
            if reference_mask is None or reference_obj is None:
                skipped.append(
                    {
                        "instance_id": int(inst_id),
                        "scenario_idx": int(scenario_idx),
                        "scenario_type": scenario["scenario_type"],
                        "scenario_family": scenario.get("scenario_family"),
                        "scenario_k": scenario.get("scenario_k"),
                        "outage_size": int(scenario["outage_size"]),
                        "deactivated_bids": _bid_names(meta, off_indices),
                        "forced_accepted_bids": _bid_names(meta, on_indices),
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
                problem="ca",
                time_limit=time_limit,
                solver_threads=threads,
                mip_gap=mip_gap,
                enforce_problem_constraints=True,
                ca_k_max=ca_k_max,
                fixed_assignments=fixed,
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
            gnn_eval = _evaluate_ca_solution(meta, gnn_mask, fixed=fixed, ca_k_max=ca_k_max)
            gnn_obj = gnn_eval.get("objective")

            reference_proven = bool(reference.get("proven_optimal"))
            gap_exact = (
                _relative_opt_gap(gnn_obj, reference_obj, sense="max")
                if reference_proven and gnn_obj is not None
                else None
            )
            gap_best_known = (
                _relative_opt_gap(gnn_obj, reference_obj, sense="max")
                if gnn_obj is not None and reference_obj is not None
                else None
            )
            exact_match = _mask_key(gnn_mask) == _mask_key(reference_mask)
            d_shift = _hamming(original_mask, reference_mask)
            d_energnn = _hamming(original_mask, gnn_mask)
            d_error = _hamming(reference_mask, gnn_mask)
            reference_replacements = _replacement_count(original_mask, reference_mask)
            energnn_replacements = _replacement_count(original_mask, gnn_mask)

            row = {
                "instance_id": int(inst_id),
                "model_type": model_type,
                "num_bids": int(len(meta.M)),
                "num_items": int(meta.globals.get("num_items", len(meta.C))),
                "num_constraints": int(meta.globals.get("num_constraints", len(meta.C))),
                "num_dummy_items": int(meta.globals.get("num_dummy_items", 0)),
                "scenario_idx": int(scenario_idx),
                "scenario_type": scenario["scenario_type"],
                "scenario_family": scenario.get("scenario_family"),
                "scenario_k": scenario.get("scenario_k"),
                "random_sample_index": scenario.get("random_sample_index"),
                "scenario_seed": scenario.get("scenario_seed"),
                "outage_size": int(scenario["outage_size"]),
                "removed_winner_count": int(len(off_indices)),
                "forced_rejected_count": int(len(on_indices)),
                "deactivated_bids": _bid_names(meta, off_indices),
                "forced_accepted_bids": _bid_names(meta, on_indices),
                "deactivated_indices": list(off_indices),
                "forced_active_indices": list(on_indices),
                "forced_accepted_indices": list(on_indices),
                "deactivated_bid_values": _bid_values(meta, off_indices),
                "forced_accepted_bid_values": _bid_values(meta, on_indices),
                "deactivated_bid_value_sum": float(sum(_bid_values(meta, off_indices))),
                "forced_accepted_bid_value_sum": float(sum(_bid_values(meta, on_indices))),
                "original_optimal_configuration": list(original_mask),
                "original_accepted_bids": _bid_names(meta, accepted_indices),
                "original_accepted_indices": list(accepted_indices),
                "original_accepted_count": int(len(accepted_indices)),
                "original_optimal_objective": original.get("objective"),
                "original_status": original.get("status"),
                "original_proven_optimal": bool(original.get("proven_optimal")),
                "original_runtime_sec": original.get("runtime"),
                "reference_configuration": list(reference_mask),
                "reference_objective": reference_obj,
                "reference_selected_count": int(sum(int(v) for v in reference_mask)),
                "reference_replacement_bid_count": reference_replacements,
                "reference_status": reference.get("status"),
                "reference_runtime_sec": reference.get("runtime"),
                "reference_bound": reference.get("bound"),
                "reference_solver_gap": reference.get("solver_gap"),
                "reference_proven_optimal": reference_proven,
                "energnn_configuration": None if gnn_mask is None else list(gnn_mask),
                "energnn_status": gnn_status,
                "energnn_energy": gnn_true_energy if gnn_true_energy is not None else gnn_mip_energy,
                "energnn_mip_energy": gnn_mip_energy,
                "energnn_ca_objective": gnn_obj,
                "energnn_raw_ca_objective": gnn_eval.get("raw_objective"),
                "energnn_feasible": bool(gnn_eval.get("feasible")),
                "energnn_fixed_constraints_satisfied": bool(gnn_eval.get("fixed_ok")),
                "energnn_num_conflicts": gnn_eval.get("num_conflicts"),
                "energnn_num_conflicted_items": gnn_eval.get("num_conflicted_items"),
                "energnn_selected_count": gnn_eval.get("selected_count"),
                "energnn_replacement_bid_count": energnn_replacements,
                "method_configuration": None if gnn_mask is None else list(gnn_mask),
                "method_status": gnn_status,
                "method_objective": gnn_obj,
                "method_raw_objective": gnn_eval.get("raw_objective"),
                "method_feasible": bool(gnn_eval.get("feasible")),
                "method_fixed_constraints_satisfied": bool(gnn_eval.get("fixed_ok")),
                "method_num_conflicts": gnn_eval.get("num_conflicts"),
                "method_num_conflicted_items": gnn_eval.get("num_conflicted_items"),
                "method_selected_count": gnn_eval.get("selected_count"),
                "method_replacement_bid_count": energnn_replacements,
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
            }
            rows.append(row)
            if save_instance_results:
                _append_jsonl(results_path, row)

            if not quiet:
                gap_text = "N/A" if gap_exact is None else f"{100.0 * gap_exact:.3f}%"
                print(
                    f"  [scenario] idx={scenario_idx} type={scenario['scenario_type']} "
                    f"off={row['deactivated_bids']} on={row['forced_accepted_bids']} "
                    f"ref={reference.get('status')} gnn={gnn_status} gap={gap_text}"
                )

    summary = _build_summary(rows, skipped, resolved)
    summary["wallclock_sec"] = float(time.perf_counter() - started)
    _write_json(output_dir / "summary.json", summary)
    if save_instance_results:
        _write_csv(output_dir / "scenario_results.csv", rows)
    if save_summary_tables:
        _write_csv(output_dir / "summary_by_outage_size.csv", summary["summary_by_outage_size"])
        _write_csv(output_dir / "summary_by_scenario_type.csv", summary["summary_by_scenario_type"])
        _write_csv(output_dir / "summary_by_scenario_k.csv", summary["summary_by_scenario_k"])
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
