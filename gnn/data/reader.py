# gnn/data/reader.py
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp
from .schema import InstanceMeta, SampleRow

# Missing/None/<=0 means CART has no open-facility cardinality cap.
DEFAULT_MAX_FACILITIES = None
LAMBDA_CARD = 1000.0

def split_pools_for_training(
    pools: Dict[int, List[SampleRow]]
) -> Tuple[Dict[int, List[SampleRow]], Dict[int, List[SampleRow]]]:
    pools_listwise = {}
    pools_feasonly = {}
    for inst_id, rows in pools.items():
        has_pos = any(r.label_type == "positive" for r in rows)
        if has_pos:
            pools_listwise[inst_id] = rows
        else:
            pools_feasonly[inst_id] = rows
    return pools_listwise, pools_feasonly


def load_static_config(path: str):
    with open(path, "r") as f:
        config = json.load(f)
    if "CAPMAX" not in config and "FCAP" in config:
        config["CAPMAX"] = config["FCAP"]
    return config


def _compute_peak_demand(inc_data: Dict[str, Dict[str, int]], T_horizon: int, TMFE: int) -> float:
    """
    Calculates the Peak Active Demand.
    Active Demand on day t = Sum of starts in window [t - TMFE + 1, t]
    """
    # 1. Aggregate starts per day across all customers
    # We size array to T_horizon + 1 to handle 1-based indexing safely
    daily_starts = np.zeros(T_horizon + 1)
    
    for c_data in inc_data.values():
        for t_str, val in c_data.items():
            t = int(t_str)
            if 1 <= t <= T_horizon:
                daily_starts[t] += val
                
    # 2. Calculate Rolling Sum (Active Demand)
    peak_val = 0.0
    
    # We check every day up to the full horizon
    for d in range(1, T_horizon + 1):
        # Window start: if d=5, TMFE=7, window is [1, 5]
        # if d=10, TMFE=7, window is [4, 10]
        start_t = max(1, d - TMFE + 1)
        active = np.sum(daily_starts[start_t : d + 1])
        if active > peak_val:
            peak_val = active
            
    return float(peak_val)


def load_instances_meta(path: str, static_config: dict) -> Dict[int, InstanceMeta]:
    metas = {}
    TMFE = int(static_config.get("TMFE", 7))
    capacity_by_facility = static_config.get("CAPMAX", static_config.get("FCAP"))
    if capacity_by_facility is None:
        raise KeyError("CART static config must define CAPMAX.")
    
    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)
            instance_id = int(obj["instance_id"])

            C = list(static_config["C"])
            M = list(static_config["M"])
            
            # Include the run-out period after the last arrival in the horizon.
            any_c = next(iter(obj["inc"]))
            arrival_days = [int(k) for k in obj["inc"][any_c].keys()]
            last_arrival = max(arrival_days) if arrival_days else 0
            
            declared_horizon = obj.get("time_horizon")
            if declared_horizon is None:
                declared_horizon = static_config.get("time_horizon")
            T_horizon = int(declared_horizon) if declared_horizon is not None else last_arrival + TMFE

            peak_demand_val = _compute_peak_demand(obj["inc"], T_horizon, TMFE)

            meta = InstanceMeta(
                instance_id=instance_id,
                C=C,
                M=M,
                CAPMAX={m: float(capacity_by_facility[m]) for m in M},
                U1={c: {m: float(static_config["U1"][c][m]) for m in M} for c in C},
                CIM={m: float(static_config["CIM"][m]) for m in M},
                CVM={m: float(static_config["CVM"][m]) for m in M},
                INC={c: {int(t): int(v) for t, v in obj["inc"][c].items()} for c in C},
                T=T_horizon,
                globals={
                    "TMFE": TMFE,
                    "material_qc_cost": float(static_config.get("material_qc_cost", 0.0)),
                    "total_therapies": obj.get("total_therapies"),
                    "demand_kwargs": obj.get("demand_kwargs", {}),
                    "max_facilities": static_config.get("max_facilities"),
                    "peak_active_demand": peak_demand_val # Stored for reader
                },
            )
            metas[instance_id] = meta
    return metas


def _load_sparse_instance_npz(npz_path: Path) -> tuple[sp.csr_matrix, np.ndarray, dict]:
    with np.load(npz_path, allow_pickle=False) as data:
        shape = tuple(int(x) for x in data["shape"])
        A_csr = sp.csr_matrix(
            (data["data"], data["indices"], data["indptr"]),
            shape=shape,
        )
        costs = np.asarray(data["costs"], dtype=float)
        extras = {
            "n_items": int(np.asarray(data["n_items"]).reshape(-1)[0]) if "n_items" in data else int(shape[0]),
            "n_dummy_items": int(np.asarray(data["n_dummy_items"]).reshape(-1)[0]) if "n_dummy_items" in data else 0,
            "n_constraints": int(np.asarray(data["n_constraints"]).reshape(-1)[0]) if "n_constraints" in data else int(shape[0]),
        }
    return A_csr, costs, extras


def load_instances_meta_ca(path: str) -> Dict[int, InstanceMeta]:
    metas: Dict[int, InstanceMeta] = {}
    base = Path(path)
    for npz_path in sorted(base.glob("instance_*.npz")):
        stem = npz_path.stem
        try:
            inst_id = int(stem.split("_")[1])
        except (IndexError, ValueError):
            continue

        A_csr, costs, extras = _load_sparse_instance_npz(npz_path)
        n_constraints, n_bids = int(A_csr.shape[0]), int(A_csr.shape[1])
        n_real_items = int(extras.get("n_items", n_constraints))
        n_dummy_items = int(extras.get("n_dummy_items", max(0, n_constraints - n_real_items)))

        C = [str(i) for i in range(n_constraints)]
        M = [str(j) for j in range(n_bids)]

        meta = InstanceMeta(
            instance_id=inst_id,
            C=C,
            M=M,
            CAPMAX={},
            U1={},
            CIM={},
            CVM={},
            INC={},
            T=1,
            globals={
                "problem": "ca",
                "A_csr": A_csr,
                "costs": costs,
                "num_items": n_real_items,
                "num_constraints": n_constraints,
                "num_dummy_items": n_dummy_items,
                "density": float(A_csr.nnz) / max(1, n_constraints * n_bids),
                "objective_sense": "max",
                "constraint_sense": "<=",
            },
        )
        metas[inst_id] = meta
    return metas


def load_instances_meta_qkp(path: str) -> Dict[int, InstanceMeta]:
    metas: Dict[int, InstanceMeta] = {}
    base = Path(path)
    for npz_path in sorted(base.glob("instance_*.npz")):
        stem = npz_path.stem
        try:
            inst_id = int(stem.split("_")[1])
        except (IndexError, ValueError):
            continue

        with np.load(npz_path, allow_pickle=False) as data:
            p = np.asarray(data["p"], dtype=float).ravel()
            Q = np.asarray(data["Q"], dtype=float)
            w = np.asarray(data["w"], dtype=float).ravel()
            W = float(np.asarray(data["W"]).reshape(-1)[0])
            n_items = int(np.asarray(data["n_items"]).reshape(-1)[0]) if "n_items" in data else int(p.size)
            sparsity = float(np.asarray(data["sparsity"]).reshape(-1)[0]) if "sparsity" in data else 0.0
            opt_obj = float(np.asarray(data["opt_obj"]).reshape(-1)[0]) if "opt_obj" in data else 0.0

        if p.size != n_items or w.size != n_items or Q.shape != (n_items, n_items):
            raise ValueError(f"Malformed QKP instance file: {npz_path}")

        pair_rows, pair_cols = np.triu_indices(n_items, k=1)
        keep = np.abs(Q[pair_rows, pair_cols]) > 1e-6
        pair_indices = np.column_stack([pair_rows[keep], pair_cols[keep]]).astype(int)

        C = [f"{int(i)}_{int(j)}" for i, j in pair_indices.tolist()]
        M = [str(j) for j in range(n_items)]

        meta = InstanceMeta(
            instance_id=inst_id,
            C=C,
            M=M,
            CAPMAX={},
            U1={},
            CIM={},
            CVM={},
            INC={},
            T=1,
            globals={
                "problem": "qkp",
                "p": p,
                "Q": Q,
                "w": w,
                "W": float(W),
                "n_items": int(n_items),
                "sparsity": float(sparsity),
                "opt_obj": float(opt_obj),
                "pair_indices": pair_indices,
                "objective_sense": "max",
                "constraint_sense": "<=",
            },
        )
        metas[inst_id] = meta
    return metas


def _bit_or_zero(x):
    if x is None:
        return 0
    if isinstance(x, bool):
        return int(x)
    try:
        return 1 if int(x) != 0 else 0
    except Exception:
        return 1 if x else 0


def _calculate_violation(sample_obj: dict, meta: InstanceMeta) -> float:
    """
    Calculates combined Capacity + Cardinality violation.
    Uses Peak Active Demand vs Daily Capacity.
    """
    if sample_obj.get("feasible"):
        return 0.0

    # 1. Identify Open Facilities
    e1_raw = sample_obj.get("e1", {}) or {}
    open_facilities = []
    for m, val in e1_raw.items():
        if _bit_or_zero(val):
            open_facilities.append(m)

    # 2. Capacity Violation (Peak Day Logic)
    # Daily capacity upper bound from the selected facilities.
    daily_supply = sum(meta.CAPMAX.get(m, 0.0) for m in open_facilities)
    
    # Peak Demand (Bottleneck Day)
    peak_demand = meta.globals.get("peak_active_demand", 0.0)
    
    # Violation is the normalized gap on that worst day
    cap_gap = max(0.0, peak_demand - daily_supply)
    cap_viol = cap_gap / max(peak_demand, 1.0)

    # 3. Cardinality Violation
    max_facilities_raw = meta.globals.get("max_facilities", DEFAULT_MAX_FACILITIES)
    if max_facilities_raw is None or int(max_facilities_raw) <= 0:
        max_facilities = len(meta.M)
    else:
        max_facilities = min(len(meta.M), int(max_facilities_raw))
    card_gap = max(0.0, len(open_facilities) - max_facilities)
    card_viol = card_gap / max(max_facilities, 1)

    return 100*cap_viol + (card_viol * LAMBDA_CARD)


def _calculate_violation_qkp(sample_obj: dict, meta: InstanceMeta) -> float:
    if sample_obj.get("feasible"):
        return 0.0

    e1_raw = sample_obj.get("e1", {}) or {}
    weights = np.asarray(meta.globals.get("w", []), dtype=float).ravel()
    capacity = float(meta.globals.get("W", 0.0))
    if weights.size == 0:
        return 0.0

    mask = np.array([_bit_or_zero(e1_raw.get(str(i), 0)) for i in range(weights.size)], dtype=float)
    total_weight = float(np.dot(weights, mask))
    return max(0.0, total_weight - capacity)


def load_samples(path: str, metas: Dict[int, InstanceMeta] = None) -> Dict[int, List[SampleRow]]:
    """
    Requires 'metas' to calculate violations on the fly.
    """
    per_instance: Dict[int, List[SampleRow]] = {}
    
    if metas is None:
        print("Warning: 'metas' not provided. Violations will default to 0.0.")

    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)
            inst_id = int(obj["instance_id"])
            current_meta = metas.get(inst_id) if metas else None

            e1_raw = obj.get("e1", {}) or {}
            if not e1_raw and "mask" in obj and current_meta is not None:
                mask = obj.get("mask") or []
                m_keys = current_meta.M
                upto = min(len(mask), len(m_keys))
                e1_raw = {str(m_keys[i]): int(mask[i]) for i in range(upto)}
            e1 = {str(k): _bit_or_zero(v) for k, v in e1_raw.items()}
            feasible = bool(obj["feasible"])

            objective = None
            if feasible:
                try:
                    val = float(obj["objective"])
                    if not (val == val) or val == float("inf") or val == float("-inf"):
                        val = None
                    objective = val
                except Exception:
                    objective = None

            regret_rel = obj.get("regret_rel")
            try:
                regret_rel = float(regret_rel) if regret_rel is not None else None
            except Exception:
                regret_rel = None

            # --- CALCULATE VIOLATION ---
            num_uncovered = obj.get("num_uncovered")
            raw_extras = obj.get("extras") or {}
            if "violation" in obj:
                violation = float(obj["violation"])
            elif "violation" in raw_extras:
                violation = float(raw_extras["violation"])
            elif (not feasible) and (num_uncovered is not None):
                violation = float(num_uncovered)
            elif not feasible and current_meta and str(current_meta.globals.get("problem", "")).lower() == "qkp":
                violation = _calculate_violation_qkp(obj, current_meta)
            elif not feasible and current_meta:
                violation = _calculate_violation(obj, current_meta)
            else:
                violation = 0.0
            # ---------------------------

            seed_demand = int(obj.get("seed_demand", 0))
            seed_solver = obj.get("seed_solver")
            seed_solver = None if seed_solver is None else int(seed_solver)

            s = SampleRow(
                instance_id=inst_id,
                label_type=obj.get("label_type", ""),
                feasible=feasible,
                objective=objective,
                status=obj.get("status", ""),
                e1=e1,
                hamming_r=obj.get("hamming_r", raw_extras.get("hamming_dist")),
                regret_rel=regret_rel,
                regret_abs=obj.get("regret_abs"),
                seed_demand=seed_demand,
                seed_solver=seed_solver,
                extras={
                    "demand_kwargs": obj.get("demand_kwargs", {}),
                    "total_therapies": obj.get("total_therapies"),
                    "num_uncovered": num_uncovered,
                    "violation": violation,
                    "source": obj.get("source", raw_extras.get("source")),
                    "objective_model": obj.get("objective_model", raw_extras.get("objective_model")),
                    "optimal": bool(obj.get("optimal", False)),
                    "from_incumbent": bool(raw_extras.get("from_incumbent", False)),
                    "hamming_dist": raw_extras.get("hamming_dist", obj.get("hamming_r")),
                },
            )
            per_instance.setdefault(inst_id, []).append(s)
    return per_instance
