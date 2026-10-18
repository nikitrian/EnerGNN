"""Generate combinatorial-auction instances and labeled candidate pools."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing as mp
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    gp = GRB = None
import numpy as np
import scipy.sparse as sp


EPS = 1e-9
HAMMING_BUCKETS: tuple[tuple[int, int], ...] = ((1, 1), (2, 3), (4, 6), (7, 12))
FEASIBLE_BUCKET_WEIGHTS: tuple[float, ...] = (0.15, 0.30, 0.35, 0.20)
FEASIBLE_SOURCE_WEIGHTS: tuple[float, ...] = (0.20, 0.40, 0.40)
FEASIBLE_REGRET_THRESHOLDS: tuple[float, ...] = (0.01, 0.02, 0.05)
INFEASIBLE_VIOLATION_WEIGHTS: tuple[float, ...] = (0.40, 0.30, 0.30)
DEFAULT_FEASIBLE_NEGS = 64
DEFAULT_INFEASIBLE_NEGS = 50
DEFAULT_POS_THRESHOLD = 0.01
DEFAULT_MAX_POSITIVES = 50

if len(HAMMING_BUCKETS) != len(FEASIBLE_BUCKET_WEIGHTS):
    raise ValueError(
        "HAMMING_BUCKETS and FEASIBLE_BUCKET_WEIGHTS must have the same length."
    )


@dataclass(frozen=True)
class ScaleSpec:
    name: str
    n_items: int
    n_bids: int


@dataclass(frozen=True)
class TaskSpec:
    scale: str
    split: str
    instance_id: int
    out_dir: str
    n_items: int
    n_bids: int
    seed: int
    n_feasible_negs: int
    n_infeasible_negs: int
    solve_time_limit: float
    max_positives: int
    pos_threshold: float
    min_value: int
    max_value: int
    value_deviation: float
    add_item_prob: float
    max_n_sub_bids: int
    additivity: float
    budget_factor: float
    resale_factor: float
    integers: bool
    warnings: bool
    solver_threads: int


@dataclass
class CauctionInstance:
    real_n_items: int
    n_dummy_items: int
    A_csr: sp.csr_matrix
    costs: np.ndarray

    @property
    def n_constraints(self) -> int:
        return int(self.A_csr.shape[0])

    @property
    def n_bids(self) -> int:
        return int(self.A_csr.shape[1])


@dataclass
class IncumbentSolution:
    mask: np.ndarray
    objective: float


SCALES: Dict[str, ScaleSpec] = {
    "175": ScaleSpec("175", n_items=100, n_bids=175),
    "250": ScaleSpec("250", n_items=100, n_bids=250),
    "small": ScaleSpec("small", n_items=25, n_bids=50),
    "medium": ScaleSpec("medium", n_items=50, n_bids=100),
    "large": ScaleSpec("large", n_items=100, n_bids=200),
    "ca100_b250": ScaleSpec("ca100_b250", n_items=100, n_bids=250),
    "ca100_b300": ScaleSpec("ca100_b300", n_items=100, n_bids=300),
    "ca100_b400": ScaleSpec("ca100_b400", n_items=100, n_bids=400),
    "ca100_b600": ScaleSpec("ca100_b600", n_items=100, n_bids=600),
}

DEFAULT_SPLIT_COUNTS = {
    "train": 1000,
    "val": 300,
    "test": 300,
}

SCALE_DEFAULTS = {
    "175": {"solve_time_limit": 3600.0, "max_positives": 50, "train_count": 1000},
    "250": {"solve_time_limit": 3600.0, "max_positives": 50, "train_count": 1000},
    "small": {"solve_time_limit": 60.0, "max_positives": 50, "train_count": 1000},
    "medium": {"solve_time_limit": 3600.0, "max_positives": 50, "train_count": 1000},
    "large": {"solve_time_limit": 3600.0, "max_positives": 50, "train_count": 1000},
    "ca100_b250": {"solve_time_limit": 3600.0, "max_positives": 50, "train_count": 1000},
    "ca100_b300": {"solve_time_limit": 3600.0, "max_positives": 50, "train_count": 1000},
    "ca100_b400": {"solve_time_limit": 3600.0, "max_positives": 50, "train_count": 1000},
    "ca100_b600": {"solve_time_limit": 3600.0, "max_positives": 50, "train_count": 1000},
}

SCALE_CODES = {
    "175": 3,
    "250": 4,
    "small": 1,
    "medium": 2,
    "large": 3,
    "ca100_b250": 4,
    "ca100_b300": 5,
    "ca100_b400": 6,
    "ca100_b600": 7,
}
SPLIT_CODES = {"train": 1, "val": 2, "test": 3}


def _column_rows(A_csc: sp.csc_matrix, j: int) -> np.ndarray:
    start, end = A_csc.indptr[j], A_csc.indptr[j + 1]
    return A_csc.indices[start:end]


def _mask_key(mask: np.ndarray) -> tuple[int, ...]:
    return tuple(int(v) for v in mask.tolist())


def _mask_to_dict(mask: Sequence[int]) -> Dict[str, int]:
    return {str(i): int(mask[i]) for i in range(len(mask))}


def _mask_from_sample_row(row: Dict[str, object]) -> np.ndarray:
    e1 = row.get("e1")
    if not isinstance(e1, dict):
        raise ValueError("Sample row missing e1 mask dict.")
    size = len(e1)
    return np.asarray([int(e1[str(i)]) for i in range(size)], dtype=np.int8)


def _mask_key_from_sample_row(row: Dict[str, object]) -> tuple[int, ...]:
    return _mask_key(_mask_from_sample_row(row))


def _row_regret(row: Dict[str, object]) -> float:
    value = row.get("regret_rel")
    if value is None:
        return float("inf")
    return float(value)


def _row_hamming(row: Dict[str, object]) -> int:
    value = row.get("hamming_r")
    if value is None:
        extras = row.get("extras")
        if isinstance(extras, dict):
            value = extras.get("hamming_dist")
    if value is None:
        return 10**9
    return int(value)


def _row_source(row: Dict[str, object]) -> str:
    value = row.get("source")
    if value is None:
        extras = row.get("extras")
        if isinstance(extras, dict):
            value = extras.get("source")
    return str(value) if value is not None else ""


def _row_violation(row: Dict[str, object]) -> float:
    value = row.get("violation")
    if value is None:
        extras = row.get("extras")
        if isinstance(extras, dict):
            value = extras.get("violation")
    return float(value or 0.0)


def _feasible_source_group(source: str) -> str:
    if source == "incumbent":
        return "incumbent"
    if source == "local_hamming":
        return "local_hamming"
    if source == "far_hamming":
        return "far_hamming"
    return "other"


def _feasible_regret_targets(n_target: int) -> list[int]:
    if n_target <= 0:
        return [0 for _ in FEASIBLE_REGRET_THRESHOLDS]
    q1 = min(int(n_target), max(2, int(round(float(n_target) * 0.05))))
    q2 = min(int(n_target), max(q1, int(round(float(n_target) * 0.10))))
    q5 = min(int(n_target), max(q2, int(round(float(n_target) * 0.20))))
    return [q1, q2, q5]


def _infeasible_violation_bucket(violation: float) -> int:
    if abs(float(violation) - 1.0) <= EPS:
        return 0
    if float(violation) <= 3.0 + EPS:
        return 1
    return 2


def _infeasible_violation_label(bucket_idx: int) -> str:
    if bucket_idx == 0:
        return "violation_1"
    if bucket_idx == 1:
        return "violation_2_3"
    return "violation_gt_3"


def _sparse_to_json_dict(A_csr: sp.csr_matrix) -> Dict[str, object]:
    return {
        "data": [int(v) for v in A_csr.data.tolist()],
        "indices": [int(v) for v in A_csr.indices.tolist()],
        "indptr": [int(v) for v in A_csr.indptr.tolist()],
        "shape": [int(A_csr.shape[0]), int(A_csr.shape[1])],
    }


def _write_jsonl(path: Path, records: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record))
            fh.write("\n")


def _build_seed(base_seed: int, scale: str, split: str, instance_id: int) -> int:
    return int(
        base_seed
        + SCALE_CODES[scale] * 1_000_000
        + SPLIT_CODES[split] * 100_000
        + instance_id
    )


def _choose_next_item(
    bundle_mask: np.ndarray,
    interests: np.ndarray,
    compats: np.ndarray,
    random_state: np.random.RandomState,
) -> int:
    n_items = len(interests)
    available_mask = 1 - bundle_mask
    compat_mean = compats[bundle_mask.astype(bool), :].mean(axis=0)
    prob = available_mask * interests * compat_mean
    prob_sum = float(prob.sum())
    if prob_sum <= 0.0:
        choices = np.flatnonzero(available_mask)
        return int(random_state.choice(choices))
    prob = prob / prob_sum
    return int(random_state.choice(n_items, p=prob))


def _build_cauction_core(
    random_state: np.random.RandomState,
    n_items: int,
    n_bids: int,
    min_value: int,
    max_value: int,
    value_deviation: float,
    add_item_prob: float,
    max_n_sub_bids: int,
    additivity: float,
    budget_factor: float,
    resale_factor: float,
    integers: bool,
    warnings: bool,
) -> tuple[list[tuple[list[int], float]], int]:
    if not (0 <= add_item_prob <= 1):
        raise ValueError("add_item_prob must be in [0, 1].")
    if max_value < min_value or min_value < 0:
        raise ValueError("Expected 0 <= min_value <= max_value.")

    values = min_value + (max_value - min_value) * random_state.rand(n_items)

    compats = np.triu(random_state.rand(n_items, n_items), k=1)
    compats = compats + compats.transpose()
    compats = compats / np.maximum(compats.sum(axis=1, keepdims=True), 1e-12)

    bids: list[tuple[list[int], float]] = []
    n_dummy_items = 0

    while len(bids) < n_bids:
        private_interests = random_state.rand(n_items)
        private_values = values + max_value * value_deviation * (2 * private_interests - 1)

        bidder_bids: Dict[frozenset[int], float] = {}

        first_prob = private_interests / private_interests.sum()
        item = int(random_state.choice(n_items, p=first_prob))
        bundle_mask = np.zeros(n_items, dtype=int)
        bundle_mask[item] = 1

        while random_state.rand() < add_item_prob:
            if int(bundle_mask.sum()) == n_items:
                break
            item = _choose_next_item(bundle_mask, private_interests, compats, random_state)
            bundle_mask[item] = 1

        bundle = np.flatnonzero(bundle_mask)
        price = float(private_values[bundle].sum() + np.power(len(bundle), 1 + additivity))
        if integers:
            price = int(price)
        if price < 0:
            if warnings:
                print("warning: negatively priced bundle avoided")
            continue

        bidder_bids[frozenset(int(i) for i in bundle.tolist())] = float(price)

        sub_candidates: list[tuple[np.ndarray, float]] = []
        for shared_item in bundle.tolist():
            sub_mask = np.zeros(n_items, dtype=int)
            sub_mask[int(shared_item)] = 1

            while int(sub_mask.sum()) < len(bundle):
                item = _choose_next_item(sub_mask, private_interests, compats, random_state)
                sub_mask[item] = 1

            sub_bundle = np.flatnonzero(sub_mask)
            sub_price = float(
                private_values[sub_bundle].sum() + np.power(len(sub_bundle), 1 + additivity)
            )
            if integers:
                sub_price = int(sub_price)
            sub_candidates.append((sub_bundle, float(sub_price)))

        budget = float(budget_factor * price)
        min_resale_value = float(resale_factor * values[bundle].sum())
        order = np.argsort([-cand_price for _, cand_price in sub_candidates])

        for idx in order.tolist():
            sub_bundle, sub_price = sub_candidates[int(idx)]
            if len(bidder_bids) >= max_n_sub_bids + 1:
                break
            if len(bids) + len(bidder_bids) >= n_bids:
                break
            if sub_price < 0:
                if warnings:
                    print("warning: negatively priced substitutable bundle avoided")
                continue
            if sub_price > budget:
                if warnings:
                    print("warning: over priced substitutable bundle avoided")
                continue
            if float(values[sub_bundle].sum()) < min_resale_value:
                if warnings:
                    print("warning: substitutable bundle below min resale value avoided")
                continue
            key = frozenset(int(i) for i in sub_bundle.tolist())
            if key in bidder_bids:
                if warnings:
                    print("warning: duplicated substitutable bundle avoided")
                continue
            bidder_bids[key] = float(sub_price)

        if len(bidder_bids) > 2:
            dummy_item = [n_items + n_dummy_items]
            n_dummy_items += 1
        else:
            dummy_item = []

        for bundle_key, bid_price in bidder_bids.items():
            bids.append((list(bundle_key) + dummy_item, float(bid_price)))
            if len(bids) >= n_bids:
                break

    return bids[:n_bids], int(n_dummy_items)


def _write_cauction_lp(filename: str, bids: Sequence[tuple[list[int], float]], n_constraints: int) -> None:
    with open(filename, "w", encoding="utf-8") as fh:
        bids_per_item = [[] for _ in range(n_constraints)]
        fh.write("maximize\nOBJ:")
        for bid_idx, (bundle, price) in enumerate(bids):
            fh.write(f" +{price} x{bid_idx + 1}")
            for item in bundle:
                bids_per_item[int(item)].append(bid_idx)
        fh.write("\n\nsubject to\n")
        for item_bids in bids_per_item:
            if not item_bids:
                continue
            for bid_idx in item_bids:
                fh.write(f" +1 x{bid_idx + 1}")
            fh.write(" <= 1\n")
        fh.write("\nbinary\n")
        for bid_idx in range(len(bids)):
            fh.write(f" x{bid_idx + 1}")
        fh.write("\n")


def generate_cauctions(
    random: np.random.RandomState,
    filename: str,
    n_items: int = 100,
    n_bids: int = 500,
    min_value: int = 1,
    max_value: int = 100,
    value_deviation: float = 0.5,
    add_item_prob: float = 0.7,
    max_n_sub_bids: int = 5,
    additivity: float = 0.2,
    budget_factor: float = 1.5,
    resale_factor: float = 0.5,
    integers: bool = False,
    warnings: bool = False,
) -> CauctionInstance:
    """
    Adapted from the Learn2Branch implementation of `generate_cauctions`.

    The upstream helper writes an LP file only. This local copy also returns the
    sparse bid-item matrix so the EnerGNN pipeline can save it as `.npz`.
    """
    bids, n_dummy_items = _build_cauction_core(
        random_state=random,
        n_items=n_items,
        n_bids=n_bids,
        min_value=min_value,
        max_value=max_value,
        value_deviation=value_deviation,
        add_item_prob=add_item_prob,
        max_n_sub_bids=max_n_sub_bids,
        additivity=additivity,
        budget_factor=budget_factor,
        resale_factor=resale_factor,
        integers=integers,
        warnings=warnings,
    )

    n_constraints = int(n_items + n_dummy_items)
    rows: list[int] = []
    cols: list[int] = []
    costs = np.zeros(len(bids), dtype=float)
    for bid_idx, (bundle, price) in enumerate(bids):
        costs[bid_idx] = float(price)
        rows.extend(int(item) for item in bundle)
        cols.extend([int(bid_idx)] * len(bundle))

    data = np.ones(len(rows), dtype=np.int8)
    A_csr = sp.coo_matrix(
        (data, (rows, cols)),
        shape=(n_constraints, len(bids)),
        dtype=np.int8,
    ).tocsr()
    A_csr.eliminate_zeros()

    _write_cauction_lp(filename, bids, n_constraints)
    return CauctionInstance(
        real_n_items=int(n_items),
        n_dummy_items=int(n_dummy_items),
        A_csr=A_csr,
        costs=costs,
    )


class IncumbentCollector:
    def __init__(self, vars_: Sequence[gp.Var]) -> None:
        self.vars = list(vars_)
        self.incumbents: list[IncumbentSolution] = []
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
            IncumbentSolution(
                mask=mask,
                objective=float(model.cbGet(GRB.Callback.MIPSOL_OBJ)),
            )
        )


def solve_cauction_gurobi(
    A_csr: sp.csr_matrix,
    costs: np.ndarray,
    seed: int,
    threads: int = 1,
    solve_time_limit: float = 0.0,
) -> tuple[np.ndarray, float, list[IncumbentSolution], bool, float]:
    if gp is None:
        raise RuntimeError("Dataset solving requires gurobipy; install mip-energnn[solver] and configure a license.")
    n_constraints, n_bids = A_csr.shape
    model = gp.Model("CAUCTIONS")
    model.Params.OutputFlag = 0
    model.Params.Threads = max(int(threads), 1)
    model.Params.Seed = int(seed)
    model.Params.MIPFocus = 1
    if solve_time_limit and float(solve_time_limit) > 0.0:
        model.Params.TimeLimit = float(solve_time_limit)

    y = model.addVars(n_bids, vtype=GRB.BINARY, name="y")
    var_list = [y[j] for j in range(n_bids)]
    model.setObjective(
        gp.quicksum(float(costs[j]) * y[j] for j in range(n_bids)),
        GRB.MAXIMIZE,
    )

    for row_idx in range(n_constraints):
        row = A_csr.getrow(row_idx)
        model.addConstr(
            gp.quicksum(y[j] for j in row.indices.tolist()) <= 1,
            name=f"pack_{row_idx}",
        )

    collector = IncumbentCollector(var_list)
    model.optimize(collector)
    if int(model.SolCount) < 1:
        raise RuntimeError(
            f"Gurobi did not find any incumbent. status={int(model.Status)} instance_shape={A_csr.shape}"
        )

    best_mask = np.array([int(round(var.X)) for var in var_list], dtype=np.int8)
    best_obj = float(model.ObjVal)
    best_key = _mask_key(best_mask)
    if best_key not in collector._seen:
        collector._seen.add(best_key)
        collector.incumbents.append(IncumbentSolution(mask=best_mask, objective=best_obj))

    collector.incumbents.sort(key=lambda row: (-float(row.objective), _mask_key(row.mask)))
    optimal = int(model.Status) == GRB.OPTIMAL
    solve_time = float(model.Runtime)
    return best_mask, best_obj, collector.incumbents, optimal, solve_time


def evaluate_cauction_mask(
    A_csr: sp.csr_matrix,
    costs: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, bool, float]:
    mask_arr = np.asarray(mask, dtype=np.int8).reshape(-1)
    allocation = np.asarray(A_csr.dot(mask_arr.astype(np.int16))).ravel().astype(np.int32)
    violation = float(np.maximum(allocation - 1, 0).sum())
    obj = float(np.dot(costs, mask_arr.astype(float)))
    return obj, violation <= EPS, violation


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
    seed_solver: Optional[int],
    source: Optional[str] = None,
) -> Dict[str, object]:
    is_optimal_row = (
        bool(feasible)
        and bool(optimal)
        and regret_abs is not None
        and abs(float(regret_abs)) <= EPS
    )
    return {
        "instance_id": int(instance_id),
        "label_type": str(label_type),
        "feasible": bool(feasible),
        "objective": None if objective is None else float(objective),
        "status": "OPTIMAL" if is_optimal_row else ("FEASIBLE" if feasible else "INFEASIBLE"),
        "e1": _mask_to_dict(mask.tolist()),
        "hamming_r": None if hamming_r is None else int(hamming_r),
        "regret_rel": None if regret_rel is None else float(regret_rel),
        "regret_abs": None if regret_abs is None else float(regret_abs),
        "optimal": bool(optimal),
        "seed_demand": 0,
        "seed_solver": None if seed_solver is None else int(seed_solver),
        "violation": float(violation),
        "source": None if source is None else str(source),
        "extras": {
            "violation": float(violation),
            "violation_bucket": (
                None if feasible else _infeasible_violation_label(_infeasible_violation_bucket(violation))
            ),
            "hamming_dist": None if hamming_r is None else int(hamming_r),
            "from_incumbent": bool(from_incumbent),
            "source": None if source is None else str(source),
        },
    }


def _try_add_infeasible_sample(
    samples: list[Dict[str, object]],
    seen: set[tuple[int, ...]],
    instance_id: int,
    best_mask: np.ndarray,
    mask: np.ndarray,
    violation: float,
    optimal: bool,
    source: str,
) -> bool:
    if violation <= EPS:
        return False
    key = _mask_key(mask)
    if key in seen:
        return False
    seen.add(key)
    samples.append(
        _sample_row(
            instance_id=instance_id,
            label_type="ham_neg_infeas",
            feasible=False,
            objective=None,
            regret_rel=None,
            regret_abs=None,
            mask=mask,
            violation=float(violation),
            hamming_r=int(np.count_nonzero(mask != best_mask)),
            optimal=optimal,
            from_incumbent=False,
            seed_solver=None,
            source=source,
        )
    )
    return True


def _try_add_feasible_sample(
    samples: list[Dict[str, object]],
    seen: set[tuple[int, ...]],
    instance_id: int,
    best_mask: np.ndarray,
    best_obj: float,
    mask: np.ndarray,
    objective: float,
    violation: float,
    optimal: bool,
    label_type: str,
    from_incumbent: bool,
    source: str,
) -> bool:
    if violation > EPS:
        return False
    key = _mask_key(mask)
    if key in seen:
        return False
    seen.add(key)
    regret_abs = float(max(best_obj - objective, 0.0))
    regret_rel = float(regret_abs / max(abs(best_obj), 1e-9))
    samples.append(
        _sample_row(
            instance_id=instance_id,
            label_type=label_type,
            feasible=True,
            objective=float(objective),
            regret_rel=regret_rel,
            regret_abs=regret_abs,
            mask=mask,
            violation=0.0,
            hamming_r=int(np.count_nonzero(mask != best_mask)),
            optimal=optimal,
            from_incumbent=from_incumbent,
            seed_solver=None,
            source=source,
        )
    )
    return True


def _promote_feasible_rows_to_positive(
    positive_rows: list[Dict[str, object]],
    feasible_rows: Sequence[Dict[str, object]],
    pos_threshold: float,
    max_positives: int,
) -> tuple[list[Dict[str, object]], list[Dict[str, object]]]:
    promoted_rows: list[Dict[str, object]] = list(positive_rows)
    remaining_rows: list[Dict[str, object]] = []
    n_positive = len(promoted_rows)

    for row in feasible_rows:
        regret_rel = row.get("regret_rel")
        if (
            regret_rel is not None
            and float(regret_rel) <= float(pos_threshold) + EPS
            and n_positive < int(max_positives)
        ):
            promoted = dict(row)
            promoted["label_type"] = "positive"
            promoted_rows.append(promoted)
            n_positive += 1
        else:
            remaining_rows.append(row)

    return promoted_rows, remaining_rows


def _bucket_target_counts(total: int, weights: Sequence[float]) -> list[int]:
    if total <= 0:
        return [0 for _ in weights]
    arr = np.asarray(weights, dtype=float)
    arr = arr / max(arr.sum(), 1e-12)
    raw = arr * float(total)
    counts = np.floor(raw).astype(int)
    remainder = int(total - counts.sum())
    if remainder > 0:
        order = np.argsort(-(raw - counts))
        for idx in order[:remainder]:
            counts[int(idx)] += 1
    return counts.tolist()


def _bucket_index_for_hamming(hamming_r: int) -> Optional[int]:
    for idx, (lo, hi) in enumerate(HAMMING_BUCKETS):
        if lo <= int(hamming_r) <= hi:
            return idx
    return None


def _choose_bucket_with_deficit(
    current_counts: Sequence[int],
    target_counts: Sequence[int],
    rng: np.random.Generator,
    weights: Optional[Sequence[float]] = None,
) -> Optional[int]:
    deficits = [max(int(t) - int(c), 0) for c, t in zip(current_counts, target_counts)]
    active = [i for i, d in enumerate(deficits) if d > 0]
    if not active:
        return None
    if weights is None:
        probs = np.asarray([deficits[i] for i in active], dtype=float)
    else:
        probs = np.asarray([float(weights[i]) * deficits[i] for i in active], dtype=float)
    probs = probs / probs.sum()
    return int(rng.choice(np.asarray(active, dtype=int), p=probs))


def _sample_exact_perturbation(
    opt_mask: np.ndarray,
    rng: np.random.Generator,
    target_hamming: int,
    require_remove: bool,
    add_biased: bool,
) -> Optional[np.ndarray]:
    selected = np.flatnonzero(opt_mask)
    unselected = np.flatnonzero(1 - opt_mask)

    remove_min = max(0, target_hamming - int(unselected.size))
    remove_max = min(int(selected.size), target_hamming)
    if require_remove:
        remove_min = max(remove_min, 1)
    if remove_min > remove_max:
        return None

    remove_choices = np.arange(remove_min, remove_max + 1, dtype=int)
    if add_biased:
        choice_weights = np.arange(remove_choices.size, 0, -1, dtype=float)
    else:
        center = float(max(remove_min, min(remove_max, int(np.ceil(target_hamming / 2.0)))))
        choice_weights = 1.0 / (1.0 + np.abs(remove_choices.astype(float) - center))
    choice_weights = choice_weights / choice_weights.sum()
    n_remove = int(rng.choice(remove_choices, p=choice_weights))
    n_add = int(target_hamming - n_remove)

    if n_remove > int(selected.size) or n_add > int(unselected.size):
        return None

    mask = opt_mask.copy()
    if n_remove > 0:
        remove_idx = rng.choice(selected, size=n_remove, replace=False)
        mask[np.asarray(remove_idx, dtype=int)] = 0
    if n_add > 0:
        add_idx = rng.choice(unselected, size=n_add, replace=False)
        mask[np.asarray(add_idx, dtype=int)] = 1
    if np.array_equal(mask, opt_mask):
        return None
    return mask.astype(np.int8, copy=False)


def _sample_feasible_exact_hamming(
    A_csc: sp.csc_matrix,
    opt_mask: np.ndarray,
    opt_allocation: np.ndarray,
    rng: np.random.Generator,
    target_hamming: int,
) -> Optional[np.ndarray]:
    selected = np.flatnonzero(opt_mask)
    unselected = np.flatnonzero(1 - opt_mask)
    if selected.size == 0:
        return None

    remove_floor = max(1, target_hamming - int(unselected.size))
    preferred_floor = max(remove_floor, int(np.ceil(target_hamming / 2.0)))
    remove_max = min(int(selected.size), target_hamming)
    if preferred_floor > remove_max:
        preferred_floor = remove_floor
    if preferred_floor > remove_max:
        return None

    remove_choices = np.arange(preferred_floor, remove_max + 1, dtype=int)
    if remove_choices.size == 0:
        return None
    probs = np.linspace(1.0, 2.0, remove_choices.size, dtype=float)
    probs = probs / probs.sum()
    n_remove = int(rng.choice(remove_choices, p=probs))
    n_add = int(target_hamming - n_remove)

    remove_idx = rng.choice(selected, size=n_remove, replace=False)
    mask = opt_mask.copy()
    allocation = opt_allocation.copy()
    for j in np.asarray(remove_idx, dtype=int).tolist():
        rows = _column_rows(A_csc, int(j))
        mask[int(j)] = 0
        allocation[rows] -= 1

    if n_add > 0:
        candidates = rng.permutation(np.flatnonzero(mask == 0)).tolist()
        added = 0
        for j in candidates:
            rows = _column_rows(A_csc, int(j))
            if rows.size and np.any(allocation[rows] >= 1):
                continue
            mask[int(j)] = 1
            allocation[rows] += 1
            added += 1
            if added >= n_add:
                break
        if added != n_add:
            return None

    if int(np.count_nonzero(mask != opt_mask)) != int(target_hamming):
        return None
    return mask.astype(np.int8, copy=False)


def _sample_guided_local_feasible(
    A_csc: sp.csc_matrix,
    costs: np.ndarray,
    opt_mask: np.ndarray,
    opt_allocation: np.ndarray,
    rng: np.random.Generator,
    target_hamming: int,
    max_attempts: int = 40,
) -> Optional[np.ndarray]:
    """
    Local-search neighborhood around the incumbent:
      - remove low-value selected bids first
      - greedily add high-value compatible bids
      - enforce an exact Hamming distance from the incumbent

    This is used for the `local_hamming` source, unlike `far_hamming`, which
    remains broader perturbation-based exploration.
    """
    selected = np.flatnonzero(opt_mask)
    unselected = np.flatnonzero(1 - opt_mask)
    if selected.size == 0 or target_hamming <= 0:
        return None

    if target_hamming == 1:
        remove_order = selected[np.argsort(costs[selected], kind="stable")]
        remove_pool = remove_order[: min(remove_order.size, 10)]
        if remove_pool.size == 0:
            return None
        weights = 1.0 / (1.0 + np.arange(remove_pool.size, dtype=float))
        weights = weights / weights.sum()
        remove_j = int(rng.choice(remove_pool, p=weights))
        mask = opt_mask.copy()
        mask[remove_j] = 0
        return mask.astype(np.int8, copy=False)

    preferred_pattern_map = {
        2: [(1, 1)],
        3: [(2, 1), (1, 2)],
    }
    fallback_pattern_map = {
        2: [(2, 0)],
        3: [(3, 0)],
    }
    preferred_patterns = preferred_pattern_map.get(int(target_hamming), [])
    fallback_patterns = fallback_pattern_map.get(int(target_hamming), [])
    if not preferred_patterns and not fallback_patterns:
        return None

    remove_order = selected[np.argsort(costs[selected], kind="stable")]
    remove_pool = remove_order[: min(remove_order.size, max(8, 4 * target_hamming))]
    if remove_pool.size == 0:
        return None

    for _ in range(max_attempts):
        use_fallback = _ >= max(1, int(0.75 * max_attempts))
        patterns = preferred_patterns + (fallback_patterns if use_fallback else [])
        for n_remove, n_add in patterns:
            if n_remove > int(remove_pool.size) or n_add > int(unselected.size):
                continue

            remove_weights = 1.0 / (1.0 + np.arange(remove_pool.size, dtype=float))
            remove_weights = remove_weights / remove_weights.sum()
            remove_idx = rng.choice(remove_pool, size=n_remove, replace=False, p=remove_weights)

            mask = opt_mask.copy()
            allocation = opt_allocation.copy()
            for j in np.asarray(remove_idx, dtype=int).tolist():
                rows = _column_rows(A_csc, int(j))
                mask[int(j)] = 0
                allocation[rows] -= 1

            added = 0
            while added < n_add:
                compatible: list[tuple[float, int]] = []
                for j in np.flatnonzero(mask == 0).tolist():
                    rows = _column_rows(A_csc, int(j))
                    if rows.size and np.any(allocation[rows] >= 1):
                        continue
                    compatible.append((float(costs[int(j)]), int(j)))
                if not compatible:
                    break

                compatible.sort(key=lambda pair: (-pair[0], pair[1]))
                top_k = compatible[: min(len(compatible), max(4, 2 * (n_add - added)))]
                add_scores = np.asarray([max(score, EPS) for score, _ in top_k], dtype=float)
                add_scores = add_scores / add_scores.sum()
                chosen_idx = int(rng.choice(np.arange(len(top_k), dtype=int), p=add_scores))
                chosen_j = int(top_k[chosen_idx][1])
                rows = _column_rows(A_csc, chosen_j)
                mask[chosen_j] = 1
                allocation[rows] += 1
                added += 1

            if added != n_add:
                continue
            if int(np.count_nonzero(mask != opt_mask)) != int(target_hamming):
                continue
            return mask.astype(np.int8, copy=False)

    return None


def _collect_incumbent_samples(
    instance_id: int,
    best_mask: np.ndarray,
    best_obj: float,
    incumbents: Sequence[IncumbentSolution],
    max_positives: int,
    pos_threshold: float,
    proven_optimal: bool,
    seen: set[tuple[int, ...]],
) -> tuple[list[Dict[str, object]], list[Dict[str, object]]]:
    positive_rows: list[Dict[str, object]] = []
    feasible_negative_rows: list[Dict[str, object]] = []
    n_positive = 0
    candidate_seen = set(seen)

    for row in sorted(incumbents, key=lambda item: (-float(item.objective), _mask_key(item.mask))):
        regret_abs = float(max(best_obj - float(row.objective), 0.0))
        regret_rel = float(regret_abs / max(abs(best_obj), 1e-9))
        label_type = "ham_neg_feas"
        if regret_rel <= float(pos_threshold) + EPS and n_positive < int(max_positives):
            label_type = "positive"
            n_positive += 1
        target_rows = positive_rows if label_type == "positive" else feasible_negative_rows
        target_seen = seen if label_type == "positive" else candidate_seen
        added = _try_add_feasible_sample(
            samples=target_rows,
            seen=target_seen,
            instance_id=instance_id,
            best_mask=best_mask,
            best_obj=best_obj,
            mask=row.mask,
            objective=float(row.objective),
            violation=0.0,
            optimal=proven_optimal,
            label_type=label_type,
            from_incumbent=True,
            source="incumbent",
        )
        if added:
            candidate_seen.add(_mask_key(row.mask))

    return positive_rows, feasible_negative_rows


def _collect_feasible_hamming_candidates(
    instance_id: int,
    A_csr: sp.csr_matrix,
    A_csc: sp.csc_matrix,
    costs: np.ndarray,
    best_mask: np.ndarray,
    best_obj: float,
    rng: np.random.Generator,
    n_target: int,
    seen: set[tuple[int, ...]],
    proven_optimal: bool,
) -> tuple[list[Dict[str, object]], list[Dict[str, object]]]:
    if int(n_target) <= 0:
        return [], []

    candidate_seen = set(seen)
    feasible_by_bucket: list[list[Dict[str, object]]] = [[] for _ in HAMMING_BUCKETS]
    candidate_target = max(int(n_target) * 2, 48)
    feasible_targets = _bucket_target_counts(candidate_target, FEASIBLE_BUCKET_WEIGHTS)
    best_allocation = np.asarray(A_csr.dot(best_mask.astype(np.int16))).ravel().astype(np.int32)

    raw_attempts = max(1800, candidate_target * 60)
    for _ in range(raw_attempts):
        current_feasible = [len(rows) for rows in feasible_by_bucket]
        bucket_idx = _choose_bucket_with_deficit(
            current_counts=current_feasible,
            target_counts=feasible_targets,
            rng=rng,
            weights=FEASIBLE_BUCKET_WEIGHTS,
        )
        if bucket_idx is None:
            break

        lo, hi = HAMMING_BUCKETS[bucket_idx]
        target_hamming = int(rng.integers(lo, hi + 1))
        if bucket_idx <= 1:
            mask = _sample_guided_local_feasible(
                A_csc=A_csc,
                costs=costs,
                opt_mask=best_mask,
                opt_allocation=best_allocation,
                rng=rng,
                target_hamming=target_hamming,
            )
        else:
            mask = _sample_exact_perturbation(
                opt_mask=best_mask,
                rng=rng,
                target_hamming=target_hamming,
                require_remove=True,
                add_biased=False,
            )
        if mask is None:
            continue

        objective, feasible, violation = evaluate_cauction_mask(A_csr, costs, mask)
        actual_bucket = _bucket_index_for_hamming(int(np.count_nonzero(mask != best_mask)))
        if feasible and objective < best_obj - EPS and actual_bucket is not None:
            if len(feasible_by_bucket[actual_bucket]) < feasible_targets[actual_bucket]:
                source = "local_hamming" if actual_bucket <= 1 else "far_hamming"
                _try_add_feasible_sample(
                    samples=feasible_by_bucket[actual_bucket],
                    seen=candidate_seen,
                    instance_id=instance_id,
                    best_mask=best_mask,
                    best_obj=best_obj,
                    mask=mask,
                    objective=objective,
                    violation=violation,
                    optimal=proven_optimal,
                    label_type="ham_neg_feas",
                    from_incumbent=False,
                    source=source,
                )

    repair_attempts = max(1500, candidate_target * 60)
    for _ in range(repair_attempts):
        current_feasible = [len(rows) for rows in feasible_by_bucket]
        bucket_idx = _choose_bucket_with_deficit(
            current_counts=current_feasible,
            target_counts=feasible_targets,
            rng=rng,
            weights=FEASIBLE_BUCKET_WEIGHTS,
        )
        if bucket_idx is None:
            break

        lo, hi = HAMMING_BUCKETS[bucket_idx]
        target_hamming = int(rng.integers(lo, hi + 1))
        if bucket_idx <= 1:
            mask = _sample_guided_local_feasible(
                A_csc=A_csc,
                costs=costs,
                opt_mask=best_mask,
                opt_allocation=best_allocation,
                rng=rng,
                target_hamming=target_hamming,
            )
        else:
            mask = _sample_feasible_exact_hamming(
                A_csc=A_csc,
                opt_mask=best_mask,
                opt_allocation=best_allocation,
                rng=rng,
                target_hamming=target_hamming,
            )
        if mask is None:
            continue
        objective, feasible, violation = evaluate_cauction_mask(A_csr, costs, mask)
        if feasible and objective < best_obj - EPS and len(feasible_by_bucket[bucket_idx]) < feasible_targets[bucket_idx]:
            source = "local_hamming" if bucket_idx <= 1 else "far_hamming"
            _try_add_feasible_sample(
                samples=feasible_by_bucket[bucket_idx],
                seen=candidate_seen,
                instance_id=instance_id,
                best_mask=best_mask,
                best_obj=best_obj,
                mask=mask,
                objective=objective,
                violation=violation,
                optimal=proven_optimal,
                label_type="ham_neg_feas",
                from_incumbent=False,
                source=source,
            )

    top_up_attempts = max(1000, candidate_target * 80)
    attempts = 0
    while sum(len(rows) for rows in feasible_by_bucket) < candidate_target and attempts < top_up_attempts:
        attempts += 1
        current_feasible = [len(rows) for rows in feasible_by_bucket]
        bucket_idx = _choose_bucket_with_deficit(
            current_counts=current_feasible,
            target_counts=feasible_targets,
            rng=rng,
            weights=FEASIBLE_BUCKET_WEIGHTS,
        )
        if bucket_idx is None:
            break

        lo, hi = HAMMING_BUCKETS[bucket_idx]
        target_hamming = int(rng.integers(lo, hi + 1))
        if bucket_idx <= 1:
            mask = _sample_guided_local_feasible(
                A_csc=A_csc,
                costs=costs,
                opt_mask=best_mask,
                opt_allocation=best_allocation,
                rng=rng,
                target_hamming=target_hamming,
            )
        else:
            mask = _sample_exact_perturbation(
                opt_mask=best_mask,
                rng=rng,
                target_hamming=target_hamming,
                require_remove=True,
                add_biased=False,
            )
        if mask is None:
            continue
        objective, feasible, violation = evaluate_cauction_mask(A_csr, costs, mask)
        actual_bucket = _bucket_index_for_hamming(int(np.count_nonzero(mask != best_mask)))
        if feasible and objective < best_obj - EPS and actual_bucket is not None:
            if len(feasible_by_bucket[actual_bucket]) < feasible_targets[actual_bucket]:
                source = "local_hamming" if actual_bucket <= 1 else "far_hamming"
                _try_add_feasible_sample(
                    samples=feasible_by_bucket[actual_bucket],
                    seen=candidate_seen,
                    instance_id=instance_id,
                    best_mask=best_mask,
                    best_obj=best_obj,
                    mask=mask,
                    objective=objective,
                    violation=violation,
                    optimal=proven_optimal,
                    label_type="ham_neg_feas",
                    from_incumbent=False,
                    source=source,
                )

    local_rows: list[Dict[str, object]] = []
    far_rows: list[Dict[str, object]] = []
    for bucket_idx, bucket_rows in enumerate(feasible_by_bucket):
        if bucket_idx <= 1:
            local_rows.extend(bucket_rows)
        else:
            far_rows.extend(bucket_rows)
    local_rows.sort(key=lambda row: (_row_regret(row), _row_hamming(row)))
    far_rows.sort(key=lambda row: (_row_regret(row), _row_hamming(row)))
    return local_rows, far_rows


def _select_feasible_negative_rows(
    incumbent_rows: Sequence[Dict[str, object]],
    local_rows: Sequence[Dict[str, object]],
    far_rows: Sequence[Dict[str, object]],
    n_target: int,
) -> list[Dict[str, object]]:
    if int(n_target) <= 0:
        return []

    source_targets_raw = _bucket_target_counts(int(n_target), FEASIBLE_SOURCE_WEIGHTS)
    source_targets = {
        "incumbent": int(source_targets_raw[0]),
        "local_hamming": int(source_targets_raw[1]),
        "far_hamming": int(source_targets_raw[2]),
    }
    regret_targets = _feasible_regret_targets(int(n_target))

    all_rows: list[Dict[str, object]] = list(incumbent_rows) + list(local_rows) + list(far_rows)
    if not all_rows:
        return []

    selected: list[Dict[str, object]] = []
    selected_keys: set[tuple[int, ...]] = set()

    def _add_row(row: Dict[str, object]) -> bool:
        if len(selected) >= int(n_target):
            return False
        key = _mask_key_from_sample_row(row)
        if key in selected_keys:
            return False
        selected_keys.add(key)
        selected.append(row)
        return True

    low_regret_source_priority = {
        "incumbent": 0,
        "local_hamming": 1,
        "far_hamming": 2,
        "other": 3,
    }
    ranked_for_regret = sorted(
        all_rows,
        key=lambda row: (
            _row_regret(row),
            low_regret_source_priority.get(_feasible_source_group(_row_source(row)), 9),
            _row_hamming(row),
        ),
    )
    for threshold, target in zip(FEASIBLE_REGRET_THRESHOLDS, regret_targets):
        current = sum(1 for row in selected if _row_regret(row) <= float(threshold) + EPS)
        needed = min(int(target), int(n_target)) - current
        if needed <= 0:
            continue
        for row in ranked_for_regret:
            if needed <= 0 or len(selected) >= int(n_target):
                break
            if _row_regret(row) > float(threshold) + EPS:
                continue
            if _add_row(row):
                needed -= 1

    for source_name in ("incumbent", "local_hamming", "far_hamming"):
        current = sum(1 for row in selected if _feasible_source_group(_row_source(row)) == source_name)
        needed = min(int(source_targets[source_name]), int(n_target)) - current
        if needed <= 0:
            continue
        source_rows = [
            row for row in all_rows
            if _feasible_source_group(_row_source(row)) == source_name
        ]
        source_rows.sort(key=lambda row: (_row_regret(row), _row_hamming(row)))
        for row in source_rows:
            if needed <= 0 or len(selected) >= int(n_target):
                break
            if _add_row(row):
                needed -= 1

    top_up_source_priority = {
        "far_hamming": 0,
        "local_hamming": 1,
        "incumbent": 2,
        "other": 3,
    }
    ranked_for_top_up = sorted(
        all_rows,
        key=lambda row: (
            top_up_source_priority.get(_feasible_source_group(_row_source(row)), 9),
            _row_regret(row),
            _row_hamming(row),
        ),
    )
    for row in ranked_for_top_up:
        if len(selected) >= int(n_target):
            break
        _add_row(row)

    return selected[: int(n_target)]


def _generate_near_feasible_infeasible(
    base_masks: Sequence[np.ndarray],
    A_csr: sp.csr_matrix,
    costs: np.ndarray,
    rng: np.random.Generator,
    max_attempts: int = 100,
) -> Optional[tuple[np.ndarray, float]]:
    if not base_masks:
        return None
    for _ in range(max_attempts):
        base_mask = base_masks[int(rng.integers(0, len(base_masks)))]
        zero_positions = np.flatnonzero(base_mask == 0)
        if zero_positions.size == 0:
            continue
        candidate = base_mask.copy()
        pos = int(rng.choice(zero_positions))
        candidate[pos] = 1
        _, feasible, violation = evaluate_cauction_mask(A_csr, costs, candidate)
        if (not feasible) and abs(float(violation) - 1.0) <= EPS:
            return candidate, float(violation)
    return None


def _generate_band_infeasible_from_base(
    base_masks: Sequence[np.ndarray],
    A_csr: sp.csr_matrix,
    costs: np.ndarray,
    rng: np.random.Generator,
    band_idx: int,
    max_attempts: int = 100,
) -> Optional[tuple[np.ndarray, float]]:
    if band_idx == 0:
        return _generate_near_feasible_infeasible(
            base_masks=base_masks,
            A_csr=A_csr,
            costs=costs,
            rng=rng,
            max_attempts=max_attempts,
        )
    if not base_masks:
        return None

    for _ in range(max_attempts):
        base_mask = base_masks[int(rng.integers(0, len(base_masks)))]
        zero_positions = np.flatnonzero(base_mask == 0)
        if zero_positions.size == 0:
            continue

        if band_idx == 1:
            min_add = 1
            max_add = min(4, int(zero_positions.size))
        else:
            min_add = 2 if int(zero_positions.size) >= 2 else 1
            max_add = min(8, int(zero_positions.size))
        if min_add > max_add:
            continue

        n_add = int(rng.integers(min_add, max_add + 1))
        chosen = rng.choice(zero_positions, size=n_add, replace=False)
        candidate = base_mask.copy()
        candidate[np.asarray(chosen, dtype=int)] = 1
        _, feasible, violation = evaluate_cauction_mask(A_csr, costs, candidate)
        if (not feasible) and _infeasible_violation_bucket(violation) == int(band_idx):
            return candidate, float(violation)
    return None


def _generate_random_infeasible_for_band(
    A_csr: sp.csr_matrix,
    costs: np.ndarray,
    n_bids: int,
    rng: np.random.Generator,
    band_idx: int,
    max_attempts: int = 100,
) -> Optional[tuple[np.ndarray, float]]:
    density_ranges = {
        0: (0.12, 0.40),
        1: (0.20, 0.55),
        2: (0.45, 0.90),
    }
    lo, hi = density_ranges.get(int(band_idx), (0.20, 0.80))
    for _ in range(max_attempts):
        density = float(rng.uniform(lo, hi))
        mask = (rng.random(int(n_bids)) < density).astype(np.int8)
        _, feasible, violation = evaluate_cauction_mask(A_csr, costs, mask)
        if feasible:
            continue
        if _infeasible_violation_bucket(violation) == int(band_idx):
            return mask, float(violation)
    return None


def _generate_infeasible_negatives(
    instance_id: int,
    A_csr: sp.csr_matrix,
    costs: np.ndarray,
    best_mask: np.ndarray,
    feasible_base_masks: Sequence[np.ndarray],
    rng: np.random.Generator,
    n_target: int,
    seen: set[tuple[int, ...]],
    proven_optimal: bool,
) -> list[Dict[str, object]]:
    if int(n_target) <= 0:
        return []

    target_counts = _bucket_target_counts(int(n_target), INFEASIBLE_VIOLATION_WEIGHTS)
    samples_by_band: list[list[Dict[str, object]]] = [[], [], []]

    for band_idx, target in enumerate(target_counts):
        if int(target) <= 0:
            continue

        base_attempts = max(300, int(target) * 80)
        attempts = 0
        while len(samples_by_band[band_idx]) < int(target) and attempts < base_attempts:
            attempts += 1
            candidate = _generate_band_infeasible_from_base(
                base_masks=feasible_base_masks,
                A_csr=A_csr,
                costs=costs,
                rng=rng,
                band_idx=band_idx,
                max_attempts=20,
            )
            if candidate is None:
                continue
            mask, violation = candidate
            base_source = (
                "near_feasible" if band_idx == 0
                else ("base_small_violation" if band_idx == 1 else "base_large_violation")
            )
            _try_add_infeasible_sample(
                samples=samples_by_band[band_idx],
                seen=seen,
                instance_id=instance_id,
                best_mask=best_mask,
                mask=mask,
                violation=violation,
                optimal=proven_optimal,
                source=base_source,
            )

        random_attempts = max(400, int(target) * 100)
        attempts = 0
        while len(samples_by_band[band_idx]) < int(target) and attempts < random_attempts:
            attempts += 1
            candidate = _generate_random_infeasible_for_band(
                A_csr=A_csr,
                costs=costs,
                n_bids=int(best_mask.size),
                rng=rng,
                band_idx=band_idx,
                max_attempts=20,
            )
            if candidate is None:
                continue
            mask, violation = candidate
            _try_add_infeasible_sample(
                samples=samples_by_band[band_idx],
                seen=seen,
                instance_id=instance_id,
                best_mask=best_mask,
                mask=mask,
                violation=violation,
                optimal=proven_optimal,
                source=f"random_{_infeasible_violation_label(band_idx)}",
            )

    samples: list[Dict[str, object]] = []
    for band_rows in samples_by_band:
        samples.extend(band_rows)

    top_up_attempts = max(500, int(n_target) * 80)
    attempts = 0
    while len(samples) < int(n_target) and attempts < top_up_attempts:
        attempts += 1
        density = float(rng.uniform(0.20, 0.90))
        mask = (rng.random(int(best_mask.size)) < density).astype(np.int8)
        _, feasible, violation = evaluate_cauction_mask(A_csr, costs, mask)
        if feasible:
            continue
        if _try_add_infeasible_sample(
            samples=samples,
            seen=seen,
            instance_id=instance_id,
            best_mask=best_mask,
            mask=mask,
            violation=violation,
            optimal=proven_optimal,
            source="random_top_up",
        ):
            continue

    if len(samples) < int(n_target):
        all_ones = np.ones(int(best_mask.size), dtype=np.int8)
        _, feasible, violation = evaluate_cauction_mask(A_csr, costs, all_ones)
        if not feasible:
            _try_add_infeasible_sample(
                samples=samples,
                seen=seen,
                instance_id=instance_id,
                best_mask=best_mask,
                mask=all_ones,
                violation=violation,
                optimal=proven_optimal,
                source="deterministic_top_up",
            )

    return samples[: int(n_target)]


def _save_instance_npz(npz_path: Path, instance: CauctionInstance) -> None:
    np.savez_compressed(
        npz_path,
        data=instance.A_csr.data,
        indices=instance.A_csr.indices,
        indptr=instance.A_csr.indptr,
        shape=np.asarray(instance.A_csr.shape, dtype=np.int64),
        costs=np.asarray(instance.costs, dtype=float),
        n_items=np.asarray([instance.real_n_items], dtype=np.int64),
        n_dummy_items=np.asarray([instance.n_dummy_items], dtype=np.int64),
        n_constraints=np.asarray([instance.n_constraints], dtype=np.int64),
    )


def _generate_one_instance(task: TaskSpec) -> Dict[str, object]:
    total_start = time.perf_counter()
    split_dir = Path(task.out_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    random_state = np.random.RandomState(int(task.seed))
    rng = np.random.default_rng(int(task.seed) + 1)
    solver_seed = int(task.seed) + 2

    lp_path = split_dir / f"_tmp_instance_{task.instance_id:04d}_{int(task.seed)}.lp"
    try:
        instance = generate_cauctions(
            random=random_state,
            filename=lp_path.as_posix(),
            n_items=int(task.n_items),
            n_bids=int(task.n_bids),
            min_value=int(task.min_value),
            max_value=int(task.max_value),
            value_deviation=float(task.value_deviation),
            add_item_prob=float(task.add_item_prob),
            max_n_sub_bids=int(task.max_n_sub_bids),
            additivity=float(task.additivity),
            budget_factor=float(task.budget_factor),
            resale_factor=float(task.resale_factor),
            integers=bool(task.integers),
            warnings=bool(task.warnings),
        )
    finally:
        lp_path.unlink(missing_ok=True)

    best_mask, best_obj, incumbents, proven_optimal, solver_time = solve_cauction_gurobi(
        instance.A_csr,
        instance.costs,
        seed=solver_seed,
        threads=task.solver_threads,
        solve_time_limit=task.solve_time_limit,
    )

    npz_path = split_dir / f"instance_{task.instance_id:04d}.npz"
    _save_instance_npz(npz_path, instance)

    seen: set[tuple[int, ...]] = set()
    positive_rows, incumbent_feasible_rows = _collect_incumbent_samples(
        instance_id=task.instance_id,
        best_mask=best_mask,
        best_obj=float(best_obj),
        incumbents=incumbents,
        max_positives=int(task.max_positives),
        pos_threshold=float(task.pos_threshold),
        proven_optimal=proven_optimal,
        seen=seen,
    )

    A_csc = instance.A_csr.tocsc()
    local_feasible_rows, far_feasible_rows = _collect_feasible_hamming_candidates(
        instance_id=task.instance_id,
        A_csr=instance.A_csr,
        A_csc=A_csc,
        costs=instance.costs,
        best_mask=best_mask,
        best_obj=float(best_obj),
        rng=rng,
        n_target=int(task.n_feasible_negs),
        seen=seen,
        proven_optimal=proven_optimal,
    )
    positive_rows, local_feasible_rows = _promote_feasible_rows_to_positive(
        positive_rows=positive_rows,
        feasible_rows=local_feasible_rows,
        pos_threshold=float(task.pos_threshold),
        max_positives=int(task.max_positives),
    )
    positive_rows, far_feasible_rows = _promote_feasible_rows_to_positive(
        positive_rows=positive_rows,
        feasible_rows=far_feasible_rows,
        pos_threshold=float(task.pos_threshold),
        max_positives=int(task.max_positives),
    )
    feasible_rows = _select_feasible_negative_rows(
        incumbent_rows=incumbent_feasible_rows,
        local_rows=local_feasible_rows,
        far_rows=far_feasible_rows,
        n_target=int(task.n_feasible_negs),
    )
    for row in feasible_rows:
        seen.add(_mask_key_from_sample_row(row))

    feasible_base_masks = [best_mask.copy()]
    feasible_base_masks.extend(row.mask.copy() for row in incumbents)
    feasible_base_masks.extend(_mask_from_sample_row(row) for row in feasible_rows)
    infeasible_rows = _generate_infeasible_negatives(
        instance_id=task.instance_id,
        A_csr=instance.A_csr,
        costs=instance.costs,
        best_mask=best_mask,
        feasible_base_masks=feasible_base_masks,
        rng=rng,
        n_target=int(task.n_infeasible_negs),
        seen=seen,
        proven_optimal=proven_optimal,
    )
    total_time = float(time.perf_counter() - total_start)

    instance_record = {
        "instance_id": int(task.instance_id),
        "problem": "ca",
        "n_items": int(instance.real_n_items),
        "n_bids": int(instance.n_bids),
        "n_constraints": int(instance.n_constraints),
        "n_dummy_items": int(instance.n_dummy_items),
        "costs": [float(v) for v in instance.costs.tolist()],
        "A": _sparse_to_json_dict(instance.A_csr),
        "optimal_obj": float(best_obj),
        "optimal_solution": [int(v) for v in best_mask.tolist()],
        "best_obj": float(best_obj),
        "optimal": bool(proven_optimal),
        "n_incumbents": int(len(incumbents)),
        "solve_time": float(total_time),
        "solver_time": float(solver_time),
        "npz_path": npz_path.name,
    }

    all_rows = positive_rows + feasible_rows + infeasible_rows
    return {
        "instance_id": int(task.instance_id),
        "instance_record": instance_record,
        "samples": all_rows,
        "n_constraints": int(instance.n_constraints),
        "n_dummy_items": int(instance.n_dummy_items),
        "n_positives": len(positive_rows),
        "n_feasible": len(feasible_rows),
        "n_infeasible": len(infeasible_rows),
        "solve_time": float(total_time),
        "solver_time": float(solver_time),
        "optimal": bool(proven_optimal),
    }


def _resolve_split_counts(args: argparse.Namespace, scales: Sequence[str]) -> Dict[str, int]:
    counts = dict(DEFAULT_SPLIT_COUNTS)
    if args.train_count is None and len(scales) == 1:
        counts["train"] = int(SCALE_DEFAULTS[scales[0]]["train_count"])
    if args.train_count is not None:
        counts["train"] = int(args.train_count)
    if args.val_count is not None:
        counts["val"] = int(args.val_count)
    if args.test_count is not None:
        counts["test"] = int(args.test_count)
    return counts


def _resolve_worker_count(workers: int) -> int:
    if int(workers) == 0:
        return max(int(mp.cpu_count()), 1)
    if int(workers) < 0:
        raise SystemExit("--workers must be >= 0, where 0 means auto.")
    return max(int(workers), 1)


def _resolve_solver_threads(solver_threads: int) -> int:
    if int(solver_threads) < 1:
        raise SystemExit("--solver-threads must be >= 1.")
    return int(solver_threads)


def _resolve_selected_scale_specs(
    selected_scales: Sequence[str],
    n_items_override: Optional[int],
    n_bids_override: Optional[int],
) -> Dict[str, ScaleSpec]:
    specs = {name: SCALES[name] for name in selected_scales}
    if n_items_override is None and n_bids_override is None:
        return specs

    if len(selected_scales) != 1:
        raise SystemExit("--n-items and --n-bids overrides require exactly one selected scale.")

    scale_name = selected_scales[0]
    base = SCALES[scale_name]
    n_items = int(base.n_items if n_items_override is None else n_items_override)
    n_bids = int(base.n_bids if n_bids_override is None else n_bids_override)
    if n_items <= 0 or n_bids <= 0:
        raise SystemExit("--n-items and --n-bids must be positive.")

    specs[scale_name] = ScaleSpec(scale_name, n_items=n_items, n_bids=n_bids)
    return specs


def _validate_generator_args(args: argparse.Namespace) -> None:
    if int(args.min_value) < 0:
        raise SystemExit("--min-value must be >= 0.")
    if int(args.max_value) < int(args.min_value):
        raise SystemExit("--max-value must be >= --min-value.")
    if not (0.0 <= float(args.add_item_prob) <= 1.0):
        raise SystemExit("--add-item-prob must be in [0, 1].")
    if int(args.max_n_sub_bids) < 0:
        raise SystemExit("--max-n-sub-bids must be >= 0.")


def _build_tasks(
    out_root: Path,
    scale_specs: Dict[str, ScaleSpec],
    splits: Sequence[str],
    split_counts: Dict[str, int],
    seed: int,
    n_feasible_negs: int,
    n_infeasible_negs: int,
    solve_time_limit: Optional[float],
    max_positives: Optional[int],
    pos_threshold: float,
    train_count_override: Optional[int],
    min_value: int,
    max_value: int,
    value_deviation: float,
    add_item_prob: float,
    max_n_sub_bids: int,
    additivity: float,
    budget_factor: float,
    resale_factor: float,
    integers: bool,
    warnings: bool,
    solver_threads: int,
) -> Dict[tuple[str, str], List[TaskSpec]]:
    tasks_by_split: Dict[tuple[str, str], List[TaskSpec]] = {}
    for scale_name, spec in scale_specs.items():
        for split_name in splits:
            split_dir = out_root / scale_name / split_name
            tasks: List[TaskSpec] = []
            if split_name == "train":
                n_instances = (
                    int(train_count_override)
                    if train_count_override is not None
                    else int(SCALE_DEFAULTS[scale_name]["train_count"])
                )
            else:
                n_instances = int(split_counts[split_name])
            for instance_id in range(n_instances):
                tasks.append(
                    TaskSpec(
                        scale=scale_name,
                        split=split_name,
                        instance_id=instance_id,
                        out_dir=split_dir.as_posix(),
                        n_items=int(spec.n_items),
                        n_bids=int(spec.n_bids),
                        seed=_build_seed(seed, scale_name, split_name, instance_id),
                        n_feasible_negs=int(n_feasible_negs),
                        n_infeasible_negs=int(n_infeasible_negs),
                        solve_time_limit=float(
                            solve_time_limit
                            if solve_time_limit is not None
                            else SCALE_DEFAULTS[scale_name]["solve_time_limit"]
                        ),
                        max_positives=int(
                            max_positives
                            if max_positives is not None
                            else SCALE_DEFAULTS[scale_name]["max_positives"]
                        ),
                        pos_threshold=float(pos_threshold),
                        min_value=int(min_value),
                        max_value=int(max_value),
                        value_deviation=float(value_deviation),
                        add_item_prob=float(add_item_prob),
                        max_n_sub_bids=int(max_n_sub_bids),
                        additivity=float(additivity),
                        budget_factor=float(budget_factor),
                        resale_factor=float(resale_factor),
                        integers=bool(integers),
                        warnings=bool(warnings),
                        solver_threads=int(solver_threads),
                    )
                )
            tasks_by_split[(scale_name, split_name)] = tasks
    return tasks_by_split


def _validate_output_dirs(out_root: Path, scales: Sequence[str], splits: Sequence[str], overwrite: bool) -> None:
    existing = []
    for scale in scales:
        for split in splits:
            split_dir = out_root / scale / split
            if split_dir.exists():
                existing.append(split_dir)
    if existing and not overwrite:
        joined = ", ".join(path.as_posix() for path in existing[:5])
        raise SystemExit(
            f"Output directories already exist ({joined}). Rerun with --overwrite to replace them."
        )


def _clear_split_dir(split_dir: Path) -> None:
    for child in split_dir.glob("*"):
        if child.is_file():
            child.unlink()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate Combinatorial Auction datasets for EnerGNN."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/cauctions"))
    parser.add_argument(
        "--scale",
        choices=sorted(SCALES.keys()),
        default=None,
        help="Optional single scale shortcut. If set, overrides --scales.",
    )
    parser.add_argument(
        "--scales",
        nargs="+",
        default=None,
        choices=sorted(SCALES.keys()),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
    )
    parser.add_argument("--train-count", type=int, default=None)
    parser.add_argument("--val-count", type=int, default=None)
    parser.add_argument("--test-count", type=int, default=None)
    parser.add_argument(
        "--n-items",
        type=int,
        default=None,
        help="Optional item-count override. Requires exactly one selected scale.",
    )
    parser.add_argument(
        "--n-bids",
        type=int,
        default=None,
        help="Optional bid-count override. Requires exactly one selected scale.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of instance-generation worker processes. Use 0 for auto.",
    )
    parser.add_argument(
        "--solver-threads",
        type=int,
        default=1,
        help="Gurobi threads per worker process.",
    )
    parser.add_argument(
        "--n-feasible-negs",
        type=int,
        default=DEFAULT_FEASIBLE_NEGS,
        help="Target feasible negatives per instance.",
    )
    parser.add_argument("--n-infeasible-negs", type=int, default=DEFAULT_INFEASIBLE_NEGS)
    parser.add_argument(
        "--solve-time-limit",
        type=float,
        default=None,
        help="Optional per-instance Gurobi time limit in seconds. Defaults by scale: 60/3600/3600.",
    )
    parser.add_argument(
        "--max-positives",
        type=int,
        default=None,
        help="Maximum number of positive incumbent samples per instance. Defaults by scale.",
    )
    parser.add_argument(
        "--pos-threshold",
        type=float,
        default=DEFAULT_POS_THRESHOLD,
        help="Maximum regret_rel for incumbent solutions to be labelled positive.",
    )
    parser.add_argument("--min-value", type=int, default=1)
    parser.add_argument("--max-value", type=int, default=100)
    parser.add_argument("--value-deviation", type=float, default=0.5)
    parser.add_argument("--add-item-prob", type=float, default=0.7)
    parser.add_argument("--max-n-sub-bids", type=int, default=5)
    parser.add_argument("--additivity", type=float, default=0.2)
    parser.add_argument("--budget-factor", type=float, default=1.5)
    parser.add_argument("--resale-factor", type=float, default=0.5)
    parser.add_argument(
        "--integers",
        action="store_true",
        help="Round generated bid prices to integers.",
    )
    parser.add_argument(
        "--generator-warnings",
        action="store_true",
        help="Print generator warnings for filtered bids/bundles.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    _validate_generator_args(args)
    selected_scales = [args.scale] if args.scale is not None else (
        args.scales if args.scales is not None else ["small", "medium", "large"]
    )
    scale_specs = _resolve_selected_scale_specs(
        selected_scales,
        n_items_override=args.n_items,
        n_bids_override=args.n_bids,
    )
    split_counts = _resolve_split_counts(args, selected_scales)
    out_root = args.output_dir.resolve()
    _validate_output_dirs(out_root, selected_scales, args.splits, args.overwrite)

    tasks_by_split = _build_tasks(
        out_root=out_root,
        scale_specs=scale_specs,
        splits=args.splits,
        split_counts=split_counts,
        seed=int(args.seed),
        n_feasible_negs=int(args.n_feasible_negs),
        n_infeasible_negs=int(args.n_infeasible_negs),
        solve_time_limit=args.solve_time_limit,
        max_positives=args.max_positives,
        pos_threshold=float(args.pos_threshold),
        train_count_override=args.train_count,
        min_value=int(args.min_value),
        max_value=int(args.max_value),
        value_deviation=float(args.value_deviation),
        add_item_prob=float(args.add_item_prob),
        max_n_sub_bids=int(args.max_n_sub_bids),
        additivity=float(args.additivity),
        budget_factor=float(args.budget_factor),
        resale_factor=float(args.resale_factor),
        integers=bool(args.integers),
        warnings=bool(args.generator_warnings),
        solver_threads=_resolve_solver_threads(int(args.solver_threads)),
    )

    mp_ctx = mp.get_context("spawn")
    max_workers = _resolve_worker_count(int(args.workers))

    summary = {
        "seed": int(args.seed),
        "workers": int(max_workers),
        "solver_threads": int(args.solver_threads),
        "n_feasible_negs": int(args.n_feasible_negs),
        "n_infeasible_negs": int(args.n_infeasible_negs),
        "solve_time_limit": None if args.solve_time_limit is None else float(args.solve_time_limit),
        "max_positives": None if args.max_positives is None else int(args.max_positives),
        "pos_threshold": float(args.pos_threshold),
        "generator": {
            "min_value": int(args.min_value),
            "max_value": int(args.max_value),
            "value_deviation": float(args.value_deviation),
            "add_item_prob": float(args.add_item_prob),
            "max_n_sub_bids": int(args.max_n_sub_bids),
            "additivity": float(args.additivity),
            "budget_factor": float(args.budget_factor),
            "resale_factor": float(args.resale_factor),
            "integers": bool(args.integers),
            "warnings": bool(args.generator_warnings),
        },
        "sampling_policy": {
            "feasible": {
                "source": "quota-driven mix of incumbent, guided local-search Hamming, and far Hamming candidates",
                "incumbent_threshold": float(args.pos_threshold),
                "max_positives": None if args.max_positives is None else int(args.max_positives),
                "regret_thresholds": list(FEASIBLE_REGRET_THRESHOLDS),
                "regret_targets_for_n_feasible": _feasible_regret_targets(int(args.n_feasible_negs)),
                "source_weights": list(FEASIBLE_SOURCE_WEIGHTS),
                "hamming_buckets": [list(b) for b in HAMMING_BUCKETS],
                "bucket_weights": list(FEASIBLE_BUCKET_WEIGHTS),
            },
            "infeasible": {
                "source": "quota-driven violation bands with base-mask and random generators",
                "violation_bands": ["violation_1", "violation_2_3", "violation_gt_3"],
                "violation_weights": list(INFEASIBLE_VIOLATION_WEIGHTS),
            },
        },
        "scales": {
            name: {
                "n_items": spec.n_items,
                "n_bids": spec.n_bids,
                "solve_time_limit": SCALE_DEFAULTS[name]["solve_time_limit"],
                "max_positives": SCALE_DEFAULTS[name]["max_positives"],
                "train_count": SCALE_DEFAULTS[name]["train_count"],
            }
            for name, spec in SCALES.items()
        },
        "selected_scales": list(selected_scales),
        "selected_scale_specs": {
            name: {
                "n_items": spec.n_items,
                "n_bids": spec.n_bids,
            }
            for name, spec in scale_specs.items()
        },
        "splits": split_counts,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    for (scale_name, split_name), tasks in tasks_by_split.items():
        split_dir = out_root / scale_name / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        if args.overwrite:
            _clear_split_dir(split_dir)

        scale_time_limit = (
            float(args.solve_time_limit)
            if args.solve_time_limit is not None
            else float(SCALE_DEFAULTS[scale_name]["solve_time_limit"])
        )
        scale_max_positives = (
            int(args.max_positives)
            if args.max_positives is not None
            else int(SCALE_DEFAULTS[scale_name]["max_positives"])
        )
        print(
            f"[start] {scale_name}/{split_name}: instances={len(tasks)} "
            f"(items={scale_specs[scale_name].n_items}, bids={scale_specs[scale_name].n_bids}, "
            f"solve_time_limit={scale_time_limit:.0f}, max_pos={scale_max_positives}, "
            f"workers={max_workers}, solver_threads={int(args.solver_threads)})",
            flush=True,
        )

        results: Dict[int, Dict[str, object]] = {}
        if tasks:
            if max_workers == 1:
                for completed, task in enumerate(tasks, start=1):
                    result = _generate_one_instance(task)
                    inst_id = int(result["instance_id"])
                    results[inst_id] = result
                    if completed % 50 == 0 or completed == len(tasks):
                        print(
                            f"[done] {scale_name}/{split_name} {completed}/{len(tasks)} "
                            f"inst={inst_id:04d} pos={result['n_positives']} "
                            f"feas={result['n_feasible']} infeas={result['n_infeasible']} "
                            f"solve_time={result['solve_time']:.2f}s optimal={result['optimal']}",
                            flush=True,
                        )
            else:
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=max_workers,
                    mp_context=mp_ctx,
                ) as executor:
                    future_map = {executor.submit(_generate_one_instance, task): task for task in tasks}
                    completed = 0
                    for future in concurrent.futures.as_completed(future_map):
                        result = future.result()
                        inst_id = int(result["instance_id"])
                        results[inst_id] = result
                        completed += 1
                        if completed % 50 == 0 or completed == len(tasks):
                            print(
                                f"[done] {scale_name}/{split_name} {completed}/{len(tasks)} "
                                f"inst={inst_id:04d} pos={result['n_positives']} "
                                f"feas={result['n_feasible']} infeas={result['n_infeasible']} "
                                f"solve_time={result['solve_time']:.2f}s optimal={result['optimal']}",
                                flush=True,
                            )

        ordered_instance_rows: List[Dict[str, object]] = []
        ordered_sample_rows: List[Dict[str, object]] = []
        for inst_id in sorted(results):
            ordered_instance_rows.append(results[inst_id]["instance_record"])
            ordered_sample_rows.extend(results[inst_id]["samples"])

        _write_jsonl(split_dir / "instances.jsonl", ordered_instance_rows)
        _write_jsonl(split_dir / "samples.jsonl", ordered_sample_rows)
        print(
            f"[write] {scale_name}/{split_name}: instances={len(ordered_instance_rows)} "
            f"samples={len(ordered_sample_rows)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
