"""Generate quadratic-knapsack instances and labeled candidate pools."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    gp = GRB = None
import numpy as np


EPS = 1e-9
DEFAULT_FEASIBLE_NEGS = 128
DEFAULT_INFEASIBLE_NEGS = 64
DEFAULT_POS_THRESHOLD = 0.01


@dataclass
class IncumbentSolution:
    mask: np.ndarray
    objective: float


def qkp_objective(mask, p, Q, w, W):
    z = np.asarray(mask, dtype=float).reshape(-1)
    linear = float(np.dot(p, z))
    quadratic = float(0.5 * z @ Q @ z)
    total_weight = float(np.dot(w, z))
    feasible = bool(total_weight <= float(W) + 1e-6)
    return linear + quadratic, feasible, total_weight


def _mask_key(mask: np.ndarray) -> tuple[int, ...]:
    return tuple(int(v) for v in np.asarray(mask, dtype=np.int8).reshape(-1).tolist())


def _mask_to_dict(mask: Sequence[int]) -> Dict[str, int]:
    return {str(i): int(mask[i]) for i in range(len(mask))}


class IncumbentCollector:
    def __init__(self, vars_: Sequence[gp.Var]) -> None:
        self.vars = list(vars_)
        self.incumbents: List[IncumbentSolution] = []
        self._seen: set[tuple[int, ...]] = set()

    def __call__(self, model: gp.Model, where: int) -> None:
        if where != GRB.Callback.MIPSOL:
            return
        sol_values = model.cbGetSolution(self.vars)
        mask = np.array([1 if float(v) > 0.5 else 0 for v in sol_values], dtype=np.int8)
        key = _mask_key(mask)
        if key in self._seen:
            return
        self._seen.add(key)
        self.incumbents.append(
            IncumbentSolution(mask=mask, objective=float(model.cbGet(GRB.Callback.MIPSOL_OBJ)))
        )


def _generate_qkp_params(
    rng: np.random.Generator,
    n_items: int,
    sparsity: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    p = rng.uniform(0.0, 10.0, n_items).astype(float)
    Q = rng.uniform(-2.0, 8.0, size=(n_items, n_items)).astype(float)
    Q = 0.5 * (Q + Q.T)
    np.fill_diagonal(Q, 0.0)

    if n_items >= 2:
        upper = np.triu_indices(n_items, k=1)
        abs_upper = np.abs(Q[upper])
        n_pairs = int(abs_upper.size)
        keep_pairs = int(round(float(sparsity) * n_pairs))
        keep_pairs = max(1, min(n_pairs, keep_pairs)) if n_pairs > 0 and float(sparsity) > 0.0 else 0
        keep_mask = np.zeros(n_pairs, dtype=bool)
        if keep_pairs > 0:
            order = np.argsort(-abs_upper)
            keep_mask[order[:keep_pairs]] = True
        Q_sparse = np.zeros_like(Q)
        rows = upper[0][keep_mask]
        cols = upper[1][keep_mask]
        Q_sparse[rows, cols] = Q[rows, cols]
        Q_sparse[cols, rows] = Q[rows, cols]
        Q = Q_sparse

    w = rng.uniform(1.0, 10.0, n_items).astype(float)
    W = float(0.5 * w.sum())
    return p, Q, w, W


def solve_qkp_gurobi(
    p: np.ndarray,
    Q: np.ndarray,
    w: np.ndarray,
    W: float,
    seed: int,
    threads: int = 1,
    time_limit: float = 0.0,
) -> tuple[np.ndarray, float, list[IncumbentSolution], bool, float]:
    if gp is None:
        raise RuntimeError("Dataset solving requires gurobipy; install mip-energnn[solver] and configure a license.")
    n_items = int(p.size)
    model = gp.Model("QKP")
    model.Params.OutputFlag = 0
    model.Params.Threads = max(int(threads), 1)
    model.Params.Seed = int(seed)
    model.Params.MIPFocus = 1
    if float(time_limit) > 0.0:
        model.Params.TimeLimit = float(time_limit)

    z = model.addVars(n_items, vtype=GRB.BINARY, name="z")
    var_list = [z[j] for j in range(n_items)]

    objective = gp.quicksum(float(p[j]) * z[j] for j in range(n_items))
    pair_rows, pair_cols = np.triu_indices(n_items, k=1)
    for i, j in zip(pair_rows.tolist(), pair_cols.tolist()):
        q_ij = float(Q[i, j])
        if abs(q_ij) <= 1e-12:
            continue
        x_ij = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
        model.addConstr(x_ij <= z[i], name=f"pair_ub_i_{i}_{j}")
        model.addConstr(x_ij <= z[j], name=f"pair_ub_j_{i}_{j}")
        model.addConstr(x_ij >= z[i] + z[j] - 1, name=f"pair_lb_{i}_{j}")
        objective += q_ij * x_ij

    model.addConstr(
        gp.quicksum(float(w[j]) * z[j] for j in range(n_items)) <= float(W),
        name="capacity",
    )
    model.setObjective(objective, GRB.MAXIMIZE)

    collector = IncumbentCollector(var_list)
    model.optimize(collector)
    if int(model.SolCount) < 1:
        raise RuntimeError(f"Gurobi did not find an incumbent for QKP instance (n={n_items}).")

    best_mask = np.array([int(round(var.X)) for var in var_list], dtype=np.int8)
    best_obj = float(model.ObjVal)
    best_key = _mask_key(best_mask)
    if best_key not in collector._seen:
        collector.incumbents.append(IncumbentSolution(mask=best_mask, objective=best_obj))
        collector._seen.add(best_key)

    collector.incumbents.sort(key=lambda row: (-float(row.objective), _mask_key(row.mask)))
    optimal = int(model.Status) == GRB.OPTIMAL
    return best_mask, best_obj, collector.incumbents, optimal, float(model.Runtime)


def _sample_row(
    instance_id: int,
    label_type: str,
    feasible: bool,
    objective: Optional[float],
    regret_rel: Optional[float],
    regret_abs: Optional[float],
    mask: np.ndarray,
    violation: float,
    hamming_r: Optional[int],
    optimal: bool,
    from_incumbent: bool,
    source: str,
) -> Dict[str, object]:
    is_optimal = bool(feasible) and bool(optimal) and regret_abs is not None and abs(float(regret_abs)) <= EPS
    return {
        "instance_id": int(instance_id),
        "label_type": str(label_type),
        "feasible": bool(feasible),
        "objective": None if objective is None else float(objective),
        "status": "OPTIMAL" if is_optimal else ("FEASIBLE" if feasible else "INFEASIBLE"),
        "e1": _mask_to_dict(mask.tolist()),
        "hamming_r": None if hamming_r is None else int(hamming_r),
        "regret_rel": None if regret_rel is None else float(regret_rel),
        "regret_abs": None if regret_abs is None else float(regret_abs),
        "seed_demand": 0,
        "seed_solver": None,
        "source": str(source),
        "extras": {
            "violation": float(violation),
            "from_incumbent": bool(from_incumbent),
            "source": str(source),
            "hamming_dist": None if hamming_r is None else int(hamming_r),
        },
    }


def _collect_incumbent_rows(
    instance_id: int,
    best_mask: np.ndarray,
    best_obj: float,
    incumbents: Sequence[IncumbentSolution],
    pos_threshold: float,
) -> tuple[list[Dict[str, object]], set[tuple[int, ...]]]:
    rows: List[Dict[str, object]] = []
    seen: set[tuple[int, ...]] = set()

    primary = [IncumbentSolution(mask=best_mask.copy(), objective=float(best_obj))]
    all_rows = primary + list(incumbents)
    for incumbent in all_rows:
        key = _mask_key(incumbent.mask)
        if key in seen:
            continue
        seen.add(key)

        regret_abs = float(best_obj) - float(incumbent.objective)
        regret_rel = regret_abs / max(abs(float(best_obj)), 1e-9)
        label_type = "positive" if regret_rel <= float(pos_threshold) + EPS else "ham_neg_feas"
        source = "optimal" if abs(regret_abs) <= EPS else "incumbent"
        rows.append(
            _sample_row(
                instance_id=instance_id,
                label_type=label_type,
                feasible=True,
                objective=float(incumbent.objective),
                regret_rel=float(regret_rel),
                regret_abs=float(regret_abs),
                mask=incumbent.mask,
                violation=0.0,
                hamming_r=int(np.count_nonzero(incumbent.mask != best_mask)),
                optimal=abs(regret_abs) <= EPS,
                from_incumbent=True,
                source=source,
            )
        )
    return rows, seen


def _random_candidate_mask(rng: np.random.Generator, n_items: int, best_mask: np.ndarray) -> tuple[np.ndarray, str]:
    if rng.random() < 0.5:
        density = float(rng.uniform(0.1, 0.9))
        mask = (rng.random(n_items) < density).astype(np.int8)
        return mask, "random"

    mask = best_mask.copy()
    n_flip = int(rng.integers(1, max(2, min(n_items, (n_items // 3) + 1))))
    flip_idx = rng.choice(n_items, size=n_flip, replace=False)
    mask[flip_idx] = 1 - mask[flip_idx]
    return mask, "mutation"


def _collect_random_rows(
    instance_id: int,
    p: np.ndarray,
    Q: np.ndarray,
    w: np.ndarray,
    W: float,
    best_mask: np.ndarray,
    best_obj: float,
    seen: set[tuple[int, ...]],
    rng: np.random.Generator,
    n_feasible_negs: int,
    n_infeasible_negs: int,
) -> tuple[list[Dict[str, object]], list[Dict[str, object]]]:
    feasible_rows: List[Dict[str, object]] = []
    infeasible_rows: List[Dict[str, object]] = []

    max_attempts = max(2000, 80 * (int(n_feasible_negs) + int(n_infeasible_negs)))
    attempts = 0
    n_items = int(best_mask.size)
    while attempts < max_attempts and (
        len(feasible_rows) < int(n_feasible_negs) or len(infeasible_rows) < int(n_infeasible_negs)
    ):
        attempts += 1
        mask, source = _random_candidate_mask(rng, n_items, best_mask)
        key = _mask_key(mask)
        if key in seen:
            continue

        obj, feasible, total_weight = qkp_objective(mask, p, Q, w, W)
        seen.add(key)
        hamming_r = int(np.count_nonzero(mask != best_mask))

        if feasible and len(feasible_rows) < int(n_feasible_negs):
            regret_abs = float(best_obj) - float(obj)
            regret_rel = regret_abs / max(abs(float(best_obj)), 1e-9)
            feasible_rows.append(
                _sample_row(
                    instance_id=instance_id,
                    label_type="ham_neg_feas",
                    feasible=True,
                    objective=float(obj),
                    regret_rel=float(regret_rel),
                    regret_abs=float(regret_abs),
                    mask=mask,
                    violation=0.0,
                    hamming_r=hamming_r,
                    optimal=False,
                    from_incumbent=False,
                    source=source,
                )
            )
        elif (not feasible) and len(infeasible_rows) < int(n_infeasible_negs):
            infeasible_rows.append(
                _sample_row(
                    instance_id=instance_id,
                    label_type="ham_neg_infeas",
                    feasible=False,
                    objective=None,
                    regret_rel=None,
                    regret_abs=None,
                    mask=mask,
                    violation=max(0.0, float(total_weight) - float(W)),
                    hamming_r=hamming_r,
                    optimal=False,
                    from_incumbent=False,
                    source=source,
                )
            )

    if len(infeasible_rows) < int(n_infeasible_negs):
        all_ones = np.ones(n_items, dtype=np.int8)
        key = _mask_key(all_ones)
        if key not in seen:
            _, feasible, total_weight = qkp_objective(all_ones, p, Q, w, W)
            if not feasible:
                seen.add(key)
                infeasible_rows.append(
                    _sample_row(
                        instance_id=instance_id,
                        label_type="ham_neg_infeas",
                        feasible=False,
                        objective=None,
                        regret_rel=None,
                        regret_abs=None,
                        mask=all_ones,
                        violation=max(0.0, float(total_weight) - float(W)),
                        hamming_r=int(np.count_nonzero(all_ones != best_mask)),
                        optimal=False,
                        from_incumbent=False,
                        source="all_ones",
                    )
                )

    return feasible_rows[: int(n_feasible_negs)], infeasible_rows[: int(n_infeasible_negs)]


def _append_jsonl_records(fh, records: Sequence[Dict[str, object]]) -> None:
    for record in records:
        fh.write(json.dumps(record))
        fh.write("\n")


def _flush_jsonl_handle(fh) -> None:
    fh.flush()
    os.fsync(fh.fileno())


def _instance_npz_path(output_dir: Path, instance_id: int) -> Path:
    return output_dir / f"instance_{instance_id:04d}.npz"


def _generate_single_instance(task: Dict[str, object]) -> Dict[str, object]:
    instance_id = int(task["instance_id"])
    output_dir = Path(str(task["output_dir"]))
    base_seed = int(task["seed"])
    n_items = int(task["n_items"])
    sparsity = float(task["sparsity"])
    pos_threshold = float(task["pos_threshold"])
    n_feasible_negs = int(task["n_feasible_negs"])
    n_infeasible_negs = int(task["n_infeasible_negs"])
    threads = int(task["threads"])
    time_limit = float(task["time_limit"])

    rng = np.random.default_rng(base_seed + instance_id)
    p, Q, w, W = _generate_qkp_params(rng, n_items, sparsity)
    best_mask, best_obj, incumbents, optimal, solve_time = solve_qkp_gurobi(
        p,
        Q,
        w,
        W,
        seed=base_seed + 10_000 + instance_id,
        threads=threads,
        time_limit=time_limit,
    )

    incumbent_rows, seen = _collect_incumbent_rows(
        instance_id=instance_id,
        best_mask=best_mask,
        best_obj=float(best_obj),
        incumbents=incumbents,
        pos_threshold=pos_threshold,
    )
    feasible_rows, infeasible_rows = _collect_random_rows(
        instance_id=instance_id,
        p=p,
        Q=Q,
        w=w,
        W=W,
        best_mask=best_mask,
        best_obj=float(best_obj),
        seen=seen,
        rng=rng,
        n_feasible_negs=n_feasible_negs,
        n_infeasible_negs=n_infeasible_negs,
    )

    npz_path = _instance_npz_path(output_dir, instance_id)
    np.savez_compressed(
        npz_path,
        p=np.asarray(p, dtype=float),
        Q=np.asarray(Q, dtype=float),
        w=np.asarray(w, dtype=float),
        W=np.asarray([float(W)], dtype=float),
        n_items=np.asarray([n_items], dtype=np.int64),
        sparsity=np.asarray([sparsity], dtype=float),
        opt_obj=np.asarray([float(best_obj)], dtype=float),
        optimal=np.asarray([bool(optimal)], dtype=bool),
        solve_time=np.asarray([float(solve_time)], dtype=float),
        optimal_mask=np.asarray(best_mask, dtype=np.int8),
    )

    instance_row = {
        "instance_id": instance_id,
        "problem": "qkp",
        "n_items": n_items,
        "sparsity": sparsity,
        "opt_obj": float(best_obj),
        "optimal_solution": [int(v) for v in best_mask.tolist()],
        "optimal": bool(optimal),
        "solve_time": float(solve_time),
        "npz_path": npz_path.name,
    }
    sample_rows = list(incumbent_rows) + list(feasible_rows) + list(infeasible_rows)
    return {
        "instance_id": instance_id,
        "instance_row": instance_row,
        "sample_rows": sample_rows,
        "best_obj": float(best_obj),
        "optimal": bool(optimal),
        "solve_time": float(solve_time),
        "n_incumbents": len(incumbent_rows),
        "n_feasible_negs": len(feasible_rows),
        "n_infeasible_negs": len(infeasible_rows),
    }


def _persist_instance_result(result: Dict[str, object], instances_fh, samples_fh) -> None:
    _append_jsonl_records(instances_fh, [result["instance_row"]])
    _append_jsonl_records(samples_fh, result["sample_rows"])
    _flush_jsonl_handle(instances_fh)
    _flush_jsonl_handle(samples_fh)
    print(
        f"[done] instance={int(result['instance_id']):04d} "
        f"opt_obj={float(result['best_obj']):.4f} "
        f"incumbents={int(result['n_incumbents'])} "
        f"feasible_negs={int(result['n_feasible_negs'])} "
        f"infeasible_negs={int(result['n_infeasible_negs'])} "
        f"optimal={bool(result['optimal'])} "
        f"solve_time={float(result['solve_time']):.2f}s",
        flush=True,
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Generate Quadratic Knapsack datasets for EnerGNN.")
    parser.add_argument("--output-dir", "--output_dir", type=Path, required=True)
    parser.add_argument("--n-instances", "--n_instances", type=int, default=100)
    parser.add_argument("--n-items", "--n_items", type=int, default=20)
    parser.add_argument("--sparsity", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--n-feasible-negs", "--n_feasible_negs", type=int, default=DEFAULT_FEASIBLE_NEGS)
    parser.add_argument("--n-infeasible-negs", "--n_infeasible_negs", type=int, default=DEFAULT_INFEASIBLE_NEGS)
    parser.add_argument("--pos-threshold", "--pos_threshold", type=float, default=DEFAULT_POS_THRESHOLD)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--n-jobs", "--n_jobs", type=int, default=1)
    parser.add_argument("--time-limit", "--time_limit", type=float, default=3600.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.n_instances <= 0 or args.n_items <= 0 or not 0 <= args.sparsity <= 1:
        parser.error("Instance/item counts must be positive and sparsity between zero and one.")
    if args.threads <= 0 or args.n_jobs <= 0 or args.time_limit < 0:
        parser.error("Threads/jobs must be positive and time limit nonnegative (zero is unlimited).")
    if min(args.n_feasible_negs, args.n_infeasible_negs, args.pos_threshold) < 0:
        parser.error("Candidate counts and positive threshold must be nonnegative.")

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for path in out_dir.glob("instance_*.npz"):
            path.unlink()
        for name in ("instances.jsonl", "samples.jsonl"):
            target = out_dir / name
            if target.exists():
                target.unlink()

    generation_config = {
        "problem": "qkp", "seed": int(args.seed), "n_items": int(args.n_items),
        "sparsity": float(args.sparsity), "time_limit": float(args.time_limit),
        "threads": int(args.threads), "n_feasible_negs": int(args.n_feasible_negs),
        "n_infeasible_negs": int(args.n_infeasible_negs),
        "pos_threshold": float(args.pos_threshold),
        "instance_seed": "seed + instance_id",
        "solver_seed": "seed + 10000 + instance_id",
    }
    config_path = out_dir / "generation_config.json"
    if config_path.exists() and not args.overwrite:
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != generation_config:
            raise ValueError("Resume settings differ from generation_config.json; choose a new output directory.")
    elif not args.overwrite and any(out_dir.glob("instance_*.npz")):
        raise ValueError("Legacy output has no generation provenance; choose a new output directory.")
    config_path.write_text(json.dumps(generation_config, indent=2) + "\n", encoding="utf-8")

    instances_path = out_dir / "instances.jsonl"
    samples_path = out_dir / "samples.jsonl"
    existing_ids = set()
    if not args.overwrite:
        for path in out_dir.glob("instance_*.npz"):
            try:
                existing_ids.add(int(path.stem.split("_")[1]))
            except (IndexError, ValueError):
                continue

    pending_ids = [i for i in range(int(args.n_instances)) if i not in existing_ids]
    if existing_ids:
        print(
            f"[resume] found {len(existing_ids)} existing instances; "
            f"generating {len(pending_ids)} pending instances",
            flush=True,
        )
    if int(args.n_jobs) > 1 and int(args.threads) > 1:
        print(
            f"[warn] n_jobs={int(args.n_jobs)} with threads={int(args.threads)} per job "
            f"may oversubscribe the machine; consider --threads 1",
            flush=True,
        )

    tasks = [
        {
            "instance_id": int(instance_id),
            "output_dir": out_dir.as_posix(),
            "seed": int(args.seed),
            "n_items": int(args.n_items),
            "sparsity": float(args.sparsity),
            "pos_threshold": float(args.pos_threshold),
            "n_feasible_negs": int(args.n_feasible_negs),
            "n_infeasible_negs": int(args.n_infeasible_negs),
            "threads": int(args.threads),
            "time_limit": float(args.time_limit),
        }
        for instance_id in pending_ids
    ]

    with instances_path.open("a", encoding="utf-8", buffering=1) as instances_fh, \
         samples_path.open("a", encoding="utf-8", buffering=1) as samples_fh:
        if int(args.n_jobs) <= 1:
            for task in tasks:
                result = _generate_single_instance(task)
                _persist_instance_result(result, instances_fh, samples_fh)
        else:
            with ProcessPoolExecutor(max_workers=int(args.n_jobs)) as executor:
                futures = [executor.submit(_generate_single_instance, task) for task in tasks]
                for future in as_completed(futures):
                    result = future.result()
                    _persist_instance_result(result, instances_fh, samples_fh)


if __name__ == "__main__":
    main()
