import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import sys
import time
import warnings
import numpy as np
import torch
from pathlib import Path
import gurobipy as gp
from gurobipy import GRB

# --- Path Setup ---
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
cart_src = ROOT / "cart_dataset"
if str(cart_src) not in sys.path:
    sys.path.insert(0, str(cart_src))

from cart_dataset.config import TMFE as CART_TMFE, FACILITY_UTIL_CAP, C_MATERIAL, C_QC
from gnn.training.utils import split_instance_ids
from gnn.inference.checkpoint import load_trained_model
from gnn.models.model import (
    model_display_name,
    resolve_ca_head_variant,
    resolve_ca_readout_variant,
    resolve_qkp_readout_variant,
)
from gnn.data.reader import (
    load_static_config,
    load_instances_meta,
    load_instances_meta_ca,
    load_instances_meta_qkp,
    load_samples,
)
from gnn.data.dataset import InstancePoolDataset

PAPER_TIME_LIMIT = 3600
PAPER_SOLVER_THREADS = 8
PAPER_FIXING_DEFAULTS = {
    "predict_and_search": {
        "cart": {
            "fixing_ratio": 0.0,
            "threshold_high": 0.6,
            "threshold_low": 0.4,
            "n_dives": 1,
        },
        "ca": {
            "fixing_ratio": 0.80,
            "threshold_high": 0.6,
            "threshold_low": 0.4,
            "n_dives": 1,
        },
        "qkp": {
            "fixing_ratio": 0.85,
            "threshold_high": 0.6,
            "threshold_low": 0.4,
            "n_dives": 1,
        },
    },
    "gnn_fixing_mip": {
        "cart": {
            "fixing_ratio": 0.0,
            "threshold_high": 0.6,
            "threshold_low": 0.4,
            "n_dives": 8,
        },
        "ca": {
            "fixing_ratio": 0.50,
            "threshold_high": 0.6,
            "threshold_low": 0.4,
            "n_dives": 100,
        },
        "qkp": {
            "fixing_ratio": 0.85,
            "threshold_high": 0.6,
            "threshold_low": 0.4,
            "n_dives": 20,
        },
    },
}


def _resolve_model_type(model_type: str) -> str:
    key = str(model_type).strip().lower()
    aliases = {
        "energnn": "energnn",
        "ebm": "energnn",
        "graph_ebm": "energnn",
        "linear_mip_contrastive": "linear_mip_contrastive",
        "linear_mip": "linear_mip_contrastive",
        "linear_additive": "linear_mip_contrastive",
        "linear_joint": "linear_mip_contrastive",
        "baseline": "linear_mip_contrastive",
        "linear_mip_bce": "linear_mip_bce",
        "linear_score_mip": "linear_mip_bce",
        "linear": "linear_mip_bce",
        "linear_score": "linear_mip_bce",
        "gnn_fixing_mip": "gnn_fixing_mip",
        "fixing": "gnn_fixing_mip",
        "gnn_fixing": "gnn_fixing_mip",
        "predict_and_search": "predict_and_search",
        "conpas_shared_encoder": "conpas_shared_encoder",
        "conpas": "conpas_shared_encoder",
        "exact_solver": "exact_solver",
        "exact": "exact_solver",
        "solver": "exact_solver",
    }
    if key not in aliases:
        raise ValueError(
            "Unsupported model_type "
            f"'{model_type}'. Expected 'energnn', 'linear_mip_bce', "
            "'linear_mip_contrastive', 'gnn_fixing_mip', 'predict_and_search', "
            "'conpas_shared_encoder', or 'exact_solver'."
        )
    return aliases[key]

# --- NEURAL LAYERS (Exact ReLU) ---
def add_relu_layer_gurobi(model, input_vars, weight, bias, name="relu", M=100.0):
    out_dim, in_dim = weight.shape
    output_vars = []
    for i in range(out_dim):
        Mi = float(M[i]) if hasattr(M, "__len__") else float(M)
        y = model.addVar(lb=0.0, name=f"{name}_y_{i}")
        z = model.addVar(vtype=GRB.BINARY, name=f"{name}_z_{i}")
        output_vars.append(y)
        
        lin_expr = gp.LinExpr()
        lin_expr += float(bias[i])
        for j, val in enumerate(input_vars):
            w = float(weight[i][j])
            if isinstance(val, gp.Var):
                lin_expr += w * val
            else:
                lin_expr += w * float(val)
            
        model.addConstr(y >= lin_expr, name=f"{name}_on_{i}")
        model.addConstr(y <= lin_expr + Mi * (1 - z), name=f"{name}_off_M_{i}")
        model.addConstr(y <= Mi * z, name=f"{name}_force_0_{i}")
        model.addConstr(lin_expr <= Mi * z, name=f"{name}_tight_{i}")
    return output_vars

def add_linear_layer_gurobi(model, input_vars, weight, bias, name="linear"):
    out_dim, in_dim = weight.shape
    output_vars = []
    for i in range(out_dim):
        y = model.addVar(lb=-GRB.INFINITY, name=f"{name}_y_{i}")
        output_vars.append(y)
        lin_expr = gp.LinExpr()
        lin_expr += float(bias[i])
        for j, val in enumerate(input_vars):
            w = float(weight[i][j])
            if isinstance(val, gp.Var):
                lin_expr += w * val
            else:
                lin_expr += w * float(val)
        model.addConstr(y == lin_expr, name=f"{name}_eq_{i}")
    return output_vars

def _bounds_sum_k(values, min_k, max_k):
    n = len(values)
    k_lo = max(0, int(min_k))
    k_hi = min(n, int(max_k))
    if k_lo > k_hi:
        k_lo, k_hi = k_hi, k_lo
    v = np.sort(np.asarray(values, dtype=float))
    prefix = np.zeros(n + 1, dtype=float)
    prefix[1:] = np.cumsum(v)
    total = prefix[-1]
    L = float("inf")
    U = float("-inf")
    for k in range(k_lo, k_hi + 1):
        smin = prefix[k]
        smax = total - prefix[n - k] if k > 0 else 0.0
        if smin < L:
            L = smin
        if smax > U:
            U = smax
    if not np.isfinite(L): L = 0.0
    if not np.isfinite(U): U = 0.0
    return float(L), float(U)

def _linear_bounds(weight, bias, in_L, in_U):
    out_dim, in_dim = weight.shape
    L = np.zeros(out_dim, dtype=float)
    U = np.zeros(out_dim, dtype=float)
    for i in range(out_dim):
        lo = float(bias[i])
        hi = float(bias[i])
        for j in range(in_dim):
            w = float(weight[i, j])
            if w >= 0.0:
                lo += w * float(in_L[j])
                hi += w * float(in_U[j])
            else:
                lo += w * float(in_U[j])
                hi += w * float(in_L[j])
        L[i] = lo
        U[i] = hi
    return L, U




def _fractional_knapsack_max(values, weights, capacity):
    vals = np.asarray(values, dtype=float).ravel()
    wts = np.asarray(weights, dtype=float).ravel()
    cap = max(float(capacity), 0.0)
    if vals.size == 0 or wts.size != vals.size or cap <= 1e-12:
        return 0.0

    positive = vals > 0.0
    if not np.any(positive):
        return 0.0

    vals = vals[positive]
    wts = wts[positive]
    density = vals / np.maximum(wts, 1e-12)
    order = np.argsort(-density)

    total = 0.0
    remaining = cap
    for idx in order.tolist():
        if remaining <= 1e-12:
            break
        wt = float(wts[idx])
        val = float(vals[idx])
        if wt <= remaining:
            total += val
            remaining -= wt
        else:
            total += val * (remaining / max(wt, 1e-12))
            break
    return float(total)


def _bounds_knapsack_fractional(values, weights, capacity):
    upper = _fractional_knapsack_max(values, weights, capacity)
    lower = -_fractional_knapsack_max(-np.asarray(values, dtype=float), weights, capacity)
    return float(lower), float(upper)







def add_symmetry_breaking(model, z_vars, capacities, skip_indices=None, tie_break_scores=None):
    """
    Detect duplicate capacities and enforce a deterministic preference order.

    For CART, lower tie_break_scores mean lower demand-weighted transport cost,
    so equal-capacity facilities are ordered by closeness to the instance demand.
    If no scores are supplied, fall back to facility order.
    """
    from collections import defaultdict
    cap_groups = defaultdict(list)
    skip = set() if skip_indices is None else {int(idx) for idx in skip_indices}
    scores = None
    if tie_break_scores is not None:
        scores = [float(v) for v in tie_break_scores]
    
    # 1. Group indices by Capacity value
    for i, cap in enumerate(capacities):
        cap_val = float(cap)
        cap_groups[cap_val].append(i)
        
    count = 0
    for cap, indices in cap_groups.items():
        if len(indices) < 2:
            continue
            
        # 2. Sort by closeness score when available, then by index for stability.
        if scores is None:
            ordered = sorted(indices)
        else:
            ordered = sorted(indices, key=lambda idx: (scores[int(idx)], int(idx)))
        ordered = [int(idx) for idx in ordered if int(idx) not in skip]
        if len(ordered) < 2:
            continue
        
        # 3. Chain Constraints: z[next] <= z[prev]
        # Constraint: Cannot open a worse equal-capacity facility unless the
        # better one is open.
        cap_label = f"{float(cap):g}".replace("-", "m").replace(".", "p")
        for k in range(len(ordered) - 1):
            better_idx = ordered[k]
            worse_idx = ordered[k + 1]
            model.addConstr(
                z_vars[worse_idx] <= z_vars[better_idx],
                name=f"SymBreak_Cap{cap_label}_{worse_idx}_{better_idx}"
            )
            count += 1
            
    # print(f"  [Gurobi] Added {count} symmetry breaking constraints.")


def _ca_cardinality_bounds(meta, k_max=None):
    n_bids = int(len(meta.M))
    lo = 0
    hi = n_bids if k_max is None else int(k_max)
    hi = max(0, min(n_bids, hi))
    if lo > hi:
        raise ValueError(f"Invalid CA cardinality bounds: k_max={hi}.")
    return lo, hi


def _eval_ca_mask(meta, mask, k_max=None):
    A_csr = meta.globals["A_csr"]
    costs = np.asarray(meta.globals["costs"], dtype=float)
    mask_arr = np.asarray(mask, dtype=np.int8).reshape(-1)
    selected_count = int(np.sum(mask_arr > 0))
    lo, hi = _ca_cardinality_bounds(meta, k_max=k_max)
    allocation = np.asarray(A_csr.dot(mask_arr.astype(np.int16))).ravel().astype(np.int32)
    violation = np.maximum(allocation - 1, 0)
    num_conflicts = int(violation.sum())
    num_conflicted_items = int(np.sum(violation > 0))
    obj = float(np.dot(costs, mask_arr.astype(float)))
    return {
        "objective": obj,
        "num_conflicts": num_conflicts,
        "num_conflicted_items": num_conflicted_items,
        "selected_count": selected_count,
        "k_max": hi,
        "cardinality_ok": bool(lo <= selected_count <= hi),
        "feasible": num_conflicts == 0 and bool(lo <= selected_count <= hi),
    }


def _eval_qkp_mask(meta, mask, k_min=None, k_max=None):
    p = np.asarray(meta.globals["p"], dtype=float)
    Q = np.asarray(meta.globals["Q"], dtype=float)
    w = np.asarray(meta.globals["w"], dtype=float)
    W = float(meta.globals["W"])
    opt_obj = float(meta.globals.get("opt_obj", 0.0))
    mask_arr = np.asarray(mask, dtype=float).reshape(-1)
    selected_count = int(np.sum(mask_arr > 0.5))
    lo, hi = _qkp_cardinality_bounds(meta, k_min=k_min, k_max=k_max)
    total_weight = float(np.dot(w, mask_arr))
    obj = float(np.dot(p, mask_arr) + 0.5 * mask_arr @ Q @ mask_arr)
    weight_ok = bool(total_weight <= W + 1e-6)
    cardinality_ok = bool(lo <= selected_count <= hi)
    feasible = bool(weight_ok and cardinality_ok)
    gap = max(0.0, opt_obj - obj) / max(abs(opt_obj), 1e-9)
    return {
        "objective": obj,
        "feasible": feasible,
        "total_weight": total_weight,
        "capacity": float(W),
        "selected_count": selected_count,
        "k_min": lo,
        "k_max": hi,
        "weight_ok": weight_ok,
        "cardinality_ok": cardinality_ok,
        "opt_gap": gap,
    }






def _qkp_cardinality_bounds(meta, k_min=None, k_max=None):
    n_items = int(len(meta.M))
    lo = 0 if k_min is None else int(k_min)
    hi = n_items if k_max is None else int(k_max)
    lo = max(0, min(n_items, lo))
    hi = max(0, min(n_items, hi))
    if lo > hi:
        raise ValueError(f"Invalid QKP cardinality bounds: k_min={lo}, k_max={hi}.")
    return lo, hi




def _cart_cardinality_bounds(num_vars, min_k=0, max_k=None):
    n = int(num_vars)
    lo = max(0, int(min_k))
    if max_k is None or int(max_k) <= 0:
        hi = n
    else:
        hi = min(n, int(max_k))
    return lo, hi


def _eval_cart_mask(meta, mask, min_k, max_k):
    mask_arr = np.asarray(mask, dtype=float)
    capacities = np.asarray([meta.CAPMAX.get(m, 0.0) for m in meta.M], dtype=float)
    open_count = int(np.sum(mask_arr > 0.5))
    total_capacity = float(np.dot(capacities, mask_arr))
    peak_active_demand = float(meta.globals.get("peak_active_demand", 0.0))
    lo_k, hi_k = _cart_cardinality_bounds(len(meta.M), min_k=min_k, max_k=max_k)
    return {
        "feasible": (lo_k <= open_count <= hi_k) and (total_capacity >= peak_active_demand - 1e-9),
        "open_count": open_count,
        "k_min": lo_k,
        "k_max": hi_k,
        "total_capacity": total_capacity,
        "peak_active_demand": peak_active_demand,
    }


def _mask_key(mask):
    return tuple(int(v) for v in mask)


def _objective_sense(meta) -> str:
    sense = str(meta.globals.get("objective_sense", "min")).strip().lower()
    return sense if sense in {"min", "max"} else "min"


def _is_better_objective(candidate, incumbent, sense, tol=1e-9):
    if incumbent is None:
        return True
    if sense == "max":
        return float(candidate) > float(incumbent) + float(tol)
    return float(candidate) < float(incumbent) - float(tol)


def _collect_sample_reference(meta, samples, obj_tol=1e-9, mask_filter=None):
    obj_sense = _objective_sense(meta)
    detailed_reference = any(
        s.extras.get("objective_model") == "cart_detailed" for s in samples
    )
    best_obj = None
    best_mask = None
    optimal_masks = set()
    feasible_obj_by_mask = {}

    for s in samples:
        if detailed_reference and s.extras.get("objective_model") != "cart_detailed":
            continue
        if not s.feasible or s.objective is None:
            continue
        obj = float(s.objective)
        if not math.isfinite(obj):
            continue
        mask_key = tuple(int(s.e1.get(m, 0) or 0) for m in meta.M)
        if mask_filter is not None and not mask_filter(mask_key):
            continue

        prev = feasible_obj_by_mask.get(mask_key)
        if _is_better_objective(obj, prev, obj_sense, obj_tol):
            feasible_obj_by_mask[mask_key] = obj

        if _is_better_objective(obj, best_obj, obj_sense, obj_tol):
            best_obj = obj
            best_mask = list(mask_key)
            optimal_masks = {mask_key}
        elif best_obj is not None and abs(obj - best_obj) <= obj_tol:
            optimal_masks.add(mask_key)

    return {
        "best_obj": best_obj,
        "best_mask": best_mask,
        "optimal_masks": optimal_masks,
        "feasible_obj_by_mask": feasible_obj_by_mask,
        "objective_model": "cart_detailed" if detailed_reference else None,
    }


def _eval_prediction_mask(
    problem,
    meta,
    mask,
    min_k,
    max_k,
    ref_stats=None,
    ca_k_max=None,
    qkp_k_min=None,
    qkp_k_max=None,
):
    mask_key = _mask_key(mask)
    if problem == "ca":
        ca_res = _eval_ca_mask(meta, mask, k_max=ca_k_max)
        return {
            **ca_res,
            "mask_key": mask_key,
            "objective_source": "exact",
            "objective": float(ca_res["objective"]),
        }
    if problem == "qkp":
        qkp_res = _eval_qkp_mask(meta, mask, k_min=qkp_k_min, k_max=qkp_k_max)
        return {
            **qkp_res,
            "mask_key": mask_key,
            "objective_source": "exact",
            "objective": float(qkp_res["objective"]),
        }
    cart_res = _eval_cart_mask(meta, mask, min_k=min_k, max_k=max_k)
    pred_obj = None
    if ref_stats is not None:
        pred_obj = ref_stats["feasible_obj_by_mask"].get(mask_key)
    return {
        **cart_res,
        "mask_key": mask_key,
        "objective_source": (
            "samples_cart_detailed"
            if pred_obj is not None and ref_stats.get("objective_model") == "cart_detailed"
            else "samples" if pred_obj is not None else None
        ),
        "objective": pred_obj,
    }


def _relative_opt_gap(pred_obj, best_obj, sense="min"):
    if pred_obj is None or best_obj is None:
        return None
    denom = max(abs(float(best_obj)), 1e-9)
    if sense == "max":
        return max(0.0, float(best_obj) - float(pred_obj)) / denom
    return max(0.0, float(pred_obj) - float(best_obj)) / denom


def _prediction_primal_gap(pred_obj, best_obj, feasible, sense="min"):
    if best_obj is None:
        return None
    if not feasible:
        return 1.0
    return _relative_opt_gap(pred_obj, best_obj, sense=sense)


def _fmt_gap_pct(gap_rel):
    if gap_rel is None or not math.isfinite(gap_rel):
        return "N/A"
    return f"{100.0 * gap_rel:.2f}%"


def _safe_ratio(num, den):
    return float(num) / float(den) if den else 0.0


def _append_jsonl(path: Path | None, row: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row))
        fh.write("\n")


def _normalize_problem_name(problem: str) -> str:
    aliases = {
        "cart": "cart",
        "ca": "ca",
        "qkp": "qkp",
        "quadratic_knapsack": "qkp",
        "quadratic-knapsack": "qkp",
        "auction": "ca",
        "auctions": "ca",
        "cauction": "ca",
        "cauctions": "ca",
        "combinatorial_auction": "ca",
        "combinatorial_auctions": "ca",
    }
    key = str(problem).strip().lower()
    if key not in aliases:
        raise ValueError(f"Unsupported problem '{problem}'. Expected one of: cart, ca, qkp.")
    return aliases[key]


def resolve_fixing_inference_defaults(
    problem: str,
    model_type: str,
    *,
    fixing_ratio=None,
    fixing_threshold_high=None,
    fixing_threshold_low=None,
    n_dives=None,
) -> dict:
    """Resolve the paper's Predict+Search and Neural Diving settings."""
    problem_key = _normalize_problem_name(problem)
    model_key = _resolve_model_type(model_type)
    if model_key not in PAPER_FIXING_DEFAULTS:
        raise ValueError(
            "Fixing defaults are only defined for 'predict_and_search' and "
            "'gnn_fixing_mip'."
        )
    defaults = PAPER_FIXING_DEFAULTS[model_key][problem_key]
    return {
        "fixing_ratio": (
            defaults["fixing_ratio"] if fixing_ratio is None else float(fixing_ratio)
        ),
        "fixing_threshold_high": (
            defaults["threshold_high"]
            if fixing_threshold_high is None
            else float(fixing_threshold_high)
        ),
        "fixing_threshold_low": (
            defaults["threshold_low"]
            if fixing_threshold_low is None
            else float(fixing_threshold_low)
        ),
        "n_dives": defaults["n_dives"] if n_dives is None else int(n_dives),
    }


def _static_config_from_dat(dat_path: str) -> dict:
    from cart_dataset.dataset_maker import load_static_from_dat

    static_sets, static_params = load_static_from_dat(dat_path)
    return {
        "C": list(static_sets["c"]),
        "M": list(static_sets["m"]),
        "TMFE": int(CART_TMFE),
        "util_cap": float(FACILITY_UTIL_CAP),
        "material_qc_cost": float(C_MATERIAL + C_QC),
        "CIM": {m: float(v) for m, v in static_params["CIM"].items()},
        "CVM": {m: float(v) for m, v in static_params["CVM"].items()},
        "CAPMAX": {m: int(round(v)) for m, v in static_params["CAPMAX"].items()},
        "U1": {
            c: {m: float(static_params["U1"][(c, m)]) for m in static_sets["m"]}
            for c in static_sets["c"]
        },
    }


def _load_cart_static_config(data_dir: str, static_config_path: str | None, dat_path: str | None) -> dict:
    candidate_paths = []
    if static_config_path:
        candidate_paths.append(Path(static_config_path))
    candidate_paths.append(Path(data_dir) / "static_config.json")

    for path in candidate_paths:
        if path.exists():
            print(f"Using static config: {path}")
            return load_static_config(path.as_posix())

    if dat_path and Path(dat_path).exists():
        print(f"static_config.json not found; rebuilding static config from {dat_path}")
        return _static_config_from_dat(dat_path)

    searched = ", ".join(str(p) for p in candidate_paths)
    raise FileNotFoundError(
        "Could not find cart static metadata. "
        f"Looked for [{searched}] and fallback dat-path [{dat_path}]."
    )


def _load_eval_data(problem: str, data_dir: str, static_config_path: str | None, dat_path: str | None):
    if problem == "ca":
        metas = load_instances_meta_ca(data_dir)
        pools = load_samples(f"{data_dir}/samples.jsonl", metas=metas)
        static = None
        tmfe = 1
    elif problem == "qkp":
        metas = load_instances_meta_qkp(data_dir)
        pools = load_samples(f"{data_dir}/samples.jsonl", metas=metas)
        static = None
        tmfe = 1
    else:
        static = _load_cart_static_config(data_dir, static_config_path, dat_path)
        metas = load_instances_meta(f"{data_dir}/instances_meta.jsonl", static)
        pools = load_samples(f"{data_dir}/samples.jsonl", metas=metas)
        tmfe = int(static.get("TMFE", 7))
    return metas, pools, static, tmfe


def _cart_symmetry_skip_indices(fixed_assignments, num_vars):
    fixed_map = _normalize_fixed_assignments(fixed_assignments, num_vars)
    return sorted(int(idx) for idx in fixed_map.keys())


def _cart_demand_weighted_transport_scores(meta):
    if meta is None:
        return None
    clinics = list(meta.C)
    facilities = list(meta.M)
    if not clinics or not facilities:
        return None

    demand_by_clinic = []
    for c in clinics:
        demand_by_clinic.append(float(sum((meta.INC.get(c, {}) or {}).values())))
    total_demand = float(sum(demand_by_clinic))
    if total_demand <= 0.0:
        weights = np.full(len(clinics), 1.0 / max(1, len(clinics)), dtype=float)
    else:
        weights = np.asarray(demand_by_clinic, dtype=float) / total_demand

    scores = []
    for m_name in facilities:
        weighted_u1 = 0.0
        for weight, c_name in zip(weights, clinics):
            weighted_u1 += float(weight) * float(meta.U1.get(c_name, {}).get(m_name, 0.0))
        scores.append(float(weighted_u1))
    return scores


def _add_cart_constraints(
    model,
    z,
    raw_capacities,
    peak_active_demand,
    min_k,
    max_k,
    fixed_assignments=None,
    meta=None,
    enable_symmetry_breaking=True,
):
    if enable_symmetry_breaking:
        add_symmetry_breaking(
            model,
            z,
            raw_capacities,
            skip_indices=_cart_symmetry_skip_indices(fixed_assignments, len(z)),
            tie_break_scores=_cart_demand_weighted_transport_scores(meta),
        )
    _add_cart_cardinality_constraints(model, z, min_k, max_k)
    raw_cap = gp.quicksum(float(raw_capacities[j]) * z[j] for j in range(len(z)))
    model.addConstr(raw_cap >= float(peak_active_demand), "Hard_Validity")


def _add_cart_cardinality_constraints(model, z, min_k, max_k):
    lo_k, hi_k = _cart_cardinality_bounds(len(z), min_k=min_k, max_k=max_k)
    z_sum = gp.quicksum(z)
    if lo_k > 0:
        model.addConstr(z_sum >= lo_k, "Cardinality_Lower")
    if hi_k < len(z):
        model.addConstr(z_sum <= hi_k, "Cardinality_Upper")


def _add_ca_constraints(model, z, A_csr, k_max=None):
    for i in range(A_csr.shape[0]):
        row = A_csr.getrow(i)
        if row.nnz <= 1:
            continue
        model.addConstr(
            gp.quicksum(z[j] for j in row.indices) <= 1,
            name=f"Pack_{i}",
        )
    if k_max is not None:
        model.addConstr(gp.quicksum(z) <= int(k_max), name="CACardinalityMax")




def _add_ca_occupancy_overload(model, z, A_csr, prefix="ca"):
    """
    Exact occupancy / overload features for CA without hard packing constraints.

    occupied_i   = 1(load_i >= 1)
    overload_i   = max(load_i - 1, 0)
    conflicted_i = 1(overload_i >= 1)
    """
    occupied = []
    overload = []
    conflicted = []

    for i in range(A_csr.shape[0]):
        row = A_csr.getrow(i)
        deg_i = int(row.nnz)

        occ_i = model.addVar(vtype=GRB.BINARY, name=f"{prefix}_occupied_{i}")
        over_i = model.addVar(vtype=GRB.INTEGER, lb=0, ub=max(deg_i - 1, 0), name=f"{prefix}_overload_{i}")
        conf_i = model.addVar(vtype=GRB.BINARY, name=f"{prefix}_conflicted_{i}")

        if deg_i == 0:
            model.addConstr(occ_i == 0, name=f"{prefix}_occupied_zero_{i}")
            model.addConstr(over_i == 0, name=f"{prefix}_overload_zero_{i}")
            model.addConstr(conf_i == 0, name=f"{prefix}_conflicted_zero_{i}")
        else:
            load_i = model.addVar(vtype=GRB.INTEGER, lb=0, ub=deg_i, name=f"{prefix}_load_{i}")
            model.addConstr(
                load_i == gp.quicksum(z[j] for j in row.indices),
                name=f"{prefix}_load_eq_{i}",
            )
            model.addConstr(load_i <= deg_i * occ_i, name=f"{prefix}_occupied_ub_{i}")
            model.addConstr(occ_i <= load_i, name=f"{prefix}_occupied_lb_{i}")
            model.addConstr(over_i == load_i - occ_i, name=f"{prefix}_overload_eq_{i}")
            if deg_i <= 1:
                model.addConstr(conf_i == 0, name=f"{prefix}_conflicted_fix_{i}")
            else:
                model.addConstr(over_i <= (deg_i - 1) * conf_i, name=f"{prefix}_conflicted_ub_{i}")
                model.addConstr(conf_i <= over_i, name=f"{prefix}_conflicted_lb_{i}")

        occupied.append(occ_i)
        overload.append(over_i)
        conflicted.append(conf_i)

    return occupied, overload, conflicted


def _add_head_objective(model, gnn_model, mlp_input, input_L, input_U):
    head = gnn_model.head
    head_variant = str(getattr(head, "variant", "mlp64")).strip().lower()

    if head_variant == "linear":
        w = head.linear.weight.detach().cpu().numpy()
        b = head.linear.bias.detach().cpu().numpy()
        energy_var = add_linear_layer_gurobi(model, mlp_input, w, b, name="Energy")[0]
    elif head_variant == "mlp16":
        mlp = head.mlp

        w1 = mlp[0].weight.detach().cpu().numpy()
        b1 = mlp[0].bias.detach().cpu().numpy()
        L1, U1 = _linear_bounds(w1, b1, input_L, input_U)
        M1 = np.maximum(0.0, np.maximum(-L1, U1))
        h1 = add_relu_layer_gurobi(model, mlp_input, w1, b1, name="MLP1", M=M1)

        w2 = mlp[3].weight.detach().cpu().numpy()
        b2 = mlp[3].bias.detach().cpu().numpy()
        energy_var = add_linear_layer_gurobi(model, h1, w2, b2, name="Energy")[0]
    elif head_variant in {"mlp32", "mlp64_shallow"}:
        mlp = head.mlp

        w1 = mlp[0].weight.detach().cpu().numpy()
        b1 = mlp[0].bias.detach().cpu().numpy()
        L1, U1 = _linear_bounds(w1, b1, input_L, input_U)
        M1 = np.maximum(0.0, np.maximum(-L1, U1))
        h1 = add_relu_layer_gurobi(model, mlp_input, w1, b1, name="MLP1", M=M1)

        w2 = mlp[3].weight.detach().cpu().numpy()
        b2 = mlp[3].bias.detach().cpu().numpy()
        energy_var = add_linear_layer_gurobi(model, h1, w2, b2, name="Energy")[0]
    elif head_variant in {"mlp64", "mlp32_deep", "mlp16_deep"}:
        mlp = head.mlp

        w1 = mlp[0].weight.detach().cpu().numpy()
        b1 = mlp[0].bias.detach().cpu().numpy()
        L1, U1 = _linear_bounds(w1, b1, input_L, input_U)
        M1 = np.maximum(0.0, np.maximum(-L1, U1))
        h1 = add_relu_layer_gurobi(model, mlp_input, w1, b1, name="MLP1", M=M1)

        w2 = mlp[3].weight.detach().cpu().numpy()
        b2 = mlp[3].bias.detach().cpu().numpy()
        L1_relu = np.maximum(0.0, L1)
        U1_relu = np.maximum(0.0, U1)
        L2, U2 = _linear_bounds(w2, b2, L1_relu, U1_relu)
        M2 = np.maximum(0.0, np.maximum(-L2, U2))
        h2 = add_relu_layer_gurobi(model, h1, w2, b2, name="MLP2", M=M2)

        w3 = mlp[6].weight.detach().cpu().numpy()
        b3 = mlp[6].bias.detach().cpu().numpy()
        energy_var = add_linear_layer_gurobi(model, h2, w3, b3, name="Energy")[0]
    else:
        raise NotImplementedError(f"Unsupported head variant for MILP encoding: {head_variant}")

    scaled_energy = model.addVar(lb=-GRB.INFINITY, name="ScaledEnergy")
    model.addConstr(scaled_energy == 1000.0 * energy_var)
    model.setObjective(scaled_energy, GRB.MINIMIZE)
    return energy_var


def _solve_and_score(model, z, gnn_model, container, energy_var):
    model.optimize()
    if model.SolCount <= 0:
        return None, None, None, "INFEASIBLE"

    mask = [int(v.X + 0.5) for v in z]
    est_energy = float(energy_var.X)
    status = "OPTIMAL" if model.Status == GRB.OPTIMAL else "SUBOPTIMAL"

    mask_t = torch.tensor([mask], dtype=torch.float32)
    with torch.no_grad():
        real_energy = gnn_model(container, mask_t).item()
    return mask, est_energy, real_energy, status


def _solve_and_score_with_details(
    model,
    z,
    gnn_model,
    container,
    energy_var,
    *,
    encoder_time_sec=None,
    return_details=False,
):
    result = _solve_and_score(model, z, gnn_model, container, energy_var)
    if not return_details:
        return result
    extra = {
        "encoder_time_sec": None if encoder_time_sec is None else float(encoder_time_sec),
        "solver_runtime_sec": float(getattr(model, "Runtime", 0.0)),
        "gurobi_status_code": int(getattr(model, "Status", 0)),
        "sol_count": int(getattr(model, "SolCount", 0)),
    }
    return (*result, extra)


def _is_linear_score_model(gnn_model) -> bool:
    return str(getattr(gnn_model, "model_type", "")).strip().lower() in {
        "linear_mip_bce",
        "linear_mip_contrastive",
    }


def _is_fixing_model(gnn_model) -> bool:
    return str(getattr(gnn_model, "model_type", "")).strip().lower() in {
        "gnn_fixing_mip", "predict_and_search"
    }


def _is_conpas_model(gnn_model) -> bool:
    return str(getattr(gnn_model, "model_type", "")).strip().lower() == "conpas_shared_encoder"


def _variable_scores_np(gnn_model, container) -> np.ndarray:
    if not hasattr(gnn_model, "score_variables"):
        raise TypeError("Model does not expose score_variables(graph).")
    with torch.no_grad():
        scores = gnn_model.score_variables(container).detach().cpu().numpy().reshape(-1)
    return scores.astype(float)


def _sigmoid_np(values) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def _objective_to_energy(meta, obj_value):
    if obj_value is None:
        return None
    return -float(obj_value) if _objective_sense(meta) == "max" else float(obj_value)


def _normalize_fixed_assignments(fixed_assignments, num_vars):
    if fixed_assignments is None:
        return {}
    if isinstance(fixed_assignments, dict):
        items = fixed_assignments.items()
    else:
        seq = list(fixed_assignments)
        if len(seq) != int(num_vars):
            raise ValueError(
                f"Fixed assignment length {len(seq)} does not match number of variables {num_vars}."
            )
        items = enumerate(seq)

    normalized = {}
    for raw_idx, raw_val in items:
        if raw_val is None:
            continue
        idx = int(raw_idx)
        if not (0 <= idx < int(num_vars)):
            raise ValueError(f"Fixed variable index {idx} out of range [0, {num_vars}).")
        normalized[idx] = int(raw_val)
    return normalized


def _apply_binary_fixes(model, binary_vars, fixed_assignments, prefix):
    fixed_map = _normalize_fixed_assignments(fixed_assignments, len(binary_vars))
    for idx, val in fixed_map.items():
        model.addConstr(binary_vars[idx] == int(val), name=f"{prefix}_{idx}")
    return fixed_map


def _cart_forced_assignments(force_off_indices, force_on_indices, num_vars):
    force_off = set() if force_off_indices is None else {int(idx) for idx in force_off_indices}
    force_on = set() if force_on_indices is None else {int(idx) for idx in force_on_indices}
    overlap = sorted(force_off & force_on)
    if overlap:
        raise ValueError(
            f"CART forced assignments conflict on indices {overlap}: "
            "cannot force the same facility both off and on."
        )
    forced = {idx: 0 for idx in force_off}
    forced.update({idx: 1 for idx in force_on})
    return _normalize_fixed_assignments(forced, num_vars)


def _solve_instance_linear_score(
    gnn_model,
    container,
    meta,
    problem="cart",
    min_k=0,
    max_k=0,
    time_limit=PAPER_TIME_LIMIT,
    solver_threads=PAPER_SOLVER_THREADS,
    mip_gap=None,
    enforce_problem_constraints=True,
    ca_k_max=None,
    qkp_k_min=None,
    qkp_k_max=None,
    fixed_assignments=None,
    cart_symmetry_breaking=True,
):
    problem = _normalize_problem_name(problem)
    scores = _variable_scores_np(gnn_model, container)
    num_vars = len(meta.M)
    if scores.size != num_vars:
        raise RuntimeError(
            f"Linear score count {scores.size} does not match number of variables {num_vars}."
        )

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    m = gp.Model("LinearScore_MIP", env=env)
    m.setParam("TimeLimit", time_limit)

    z = [m.addVar(vtype=GRB.BINARY, name=f"z_{j}") for j in range(num_vars)]
    _apply_binary_fixes(m, z, fixed_assignments, "fixed_z")

    if enforce_problem_constraints:
        if problem == "ca":
            _add_ca_constraints(m, z, meta.globals["A_csr"], k_max=ca_k_max)
        elif problem == "qkp":
            weights = np.asarray(meta.globals["w"], dtype=float).ravel()
            capacity = float(meta.globals["W"])
            m.addConstr(
                gp.quicksum(float(weights[j]) * z[j] for j in range(num_vars)) <= capacity,
                name="Capacity",
            )
            qkp_lo, qkp_hi = _qkp_cardinality_bounds(meta, k_min=qkp_k_min, k_max=qkp_k_max)
            if qkp_lo > 0:
                m.addConstr(gp.quicksum(z) >= qkp_lo, name="QKPCardinalityMin")
            if qkp_hi < num_vars:
                m.addConstr(gp.quicksum(z) <= qkp_hi, name="QKPCardinalityMax")
        elif problem == "cart":
            raw_capacities = np.asarray([meta.CAPMAX[fid] for fid in meta.M], dtype=float)
            peak_active_demand = float(meta.globals.get("peak_active_demand", 0.0))
            _add_cart_constraints(
                m,
                z,
                raw_capacities,
                peak_active_demand,
                min_k,
                max_k,
                fixed_assignments=fixed_assignments,
                meta=meta,
                enable_symmetry_breaking=cart_symmetry_breaking,
            )
        else:
            raise ValueError(f"Unsupported problem for linear-score MIP: {problem}")

    score_expr = gp.quicksum(float(scores[j]) * z[j] for j in range(num_vars))
    energy_var = m.addVar(lb=-GRB.INFINITY, name="LinearScoreEnergy")
    m.addConstr(energy_var == -score_expr, name="LinearScoreEnergy_eq")
    m.setObjective(energy_var, GRB.MINIMIZE)

    return _solve_and_score(m, z, gnn_model, container, energy_var)






def _solve_instance_ca_pooled_or_direct_y(
    gnn_model,
    container,
    meta,
    time_limit=PAPER_TIME_LIMIT,
    enforce_problem_constraints=True,
    ca_k_max=None,
    fixed_assignments=None,
    solver_threads=PAPER_SOLVER_THREADS,
    mip_gap=None,
    return_details=False,
    include_direct_y=False,
):
    encoder_start = time.perf_counter()
    with torch.no_grad():
        enc_out = gnn_model.encoder(container)
        hc = enc_out["hc"].cpu().numpy()          # [nC, dh]
        hm = enc_out["hm"].cpu().numpy()          # [nBids, dh]
        hg = enc_out["hg"].cpu().numpy()          # [2*dh]
        dh = hm.shape[1]
        num_bids = len(meta.M)
        nC = int(container["nC"])
        A_csr = meta.globals["A_csr"]
    encoder_time = time.perf_counter() - encoder_start

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    model_name = "Exact_GNN_CA_DirectY" if include_direct_y else "Exact_GNN_CA_PooledOnly"
    m = gp.Model(model_name, env=env)
    m.setParam("TimeLimit", time_limit)
    if solver_threads is not None:
        m.setParam("Threads", int(solver_threads))
    if mip_gap is not None:
        m.setParam("MIPGap", float(mip_gap))

    z = [m.addVar(vtype=GRB.BINARY, name=f"z_{j}") for j in range(num_bids)]
    _apply_binary_fixes(m, z, fixed_assignments, "fixed_z")
    if enforce_problem_constraints:
        _add_ca_constraints(m, z, A_csr, k_max=ca_k_max)

    selected_pool_vars = []
    selected_pool_L = []
    selected_pool_U = []
    for d in range(dh):
        y = m.addVar(lb=-GRB.INFINITY, name=f"selected_pool_{d}")
        m.addConstr(
            y == gp.quicksum(float(hm[j, d]) * z[j] for j in range(num_bids)),
            name=f"selected_pool_eq_{d}",
        )
        selected_pool_vars.append(y)
        lo, hi = _bounds_sum_k(hm[:, d], 0, num_bids)
        selected_pool_L.append(lo)
        selected_pool_U.append(hi)

    if enforce_problem_constraints:
        occupied = None
        occupied_pool_coeff = np.asarray(A_csr.T @ hc, dtype=float)
    else:
        occupied, _, _ = _add_ca_occupancy_overload(m, z, A_csr, prefix="pooled")
        occupied_pool_coeff = None

    occupied_pool_vars = []
    occupied_pool_L = []
    occupied_pool_U = []
    for d in range(dh):
        y = m.addVar(lb=-GRB.INFINITY, name=f"occupied_pool_{d}")
        if enforce_problem_constraints:
            m.addConstr(
                y == gp.quicksum(float(occupied_pool_coeff[j, d]) * z[j] for j in range(num_bids)),
                name=f"occupied_pool_eq_{d}",
            )
        else:
            m.addConstr(
                y == gp.quicksum(float(hc[i, d]) * occupied[i] for i in range(nC)),
                name=f"occupied_pool_eq_{d}",
            )
        occupied_pool_vars.append(y)
        lo, hi = _bounds_sum_k(hc[:, d], 0, nC)
        occupied_pool_L.append(lo)
        occupied_pool_U.append(hi)

    mlp_input = []
    mlp_input.extend(selected_pool_vars)          # [dh]
    mlp_input.extend(occupied_pool_vars)          # [dh]

    input_L_parts = [
        np.asarray(selected_pool_L, dtype=float),
        np.asarray(occupied_pool_L, dtype=float),
    ]
    input_U_parts = [
        np.asarray(selected_pool_U, dtype=float),
        np.asarray(occupied_pool_U, dtype=float),
    ]

    direct_y_dim = 0
    if include_direct_y:
        direct_y_dim = int(getattr(getattr(gnn_model, "readout", None), "direct_y_dim", 0))
        if direct_y_dim <= 0:
            raise RuntimeError("CA direct_y readout is missing a positive direct_y_dim.")
        if num_bids > direct_y_dim:
            raise RuntimeError(
                f"CA direct_y_dim={direct_y_dim} is smaller than the instance "
                f"width {num_bids}."
            )

        mlp_input.extend(z)
        if direct_y_dim > num_bids:
            mlp_input.extend(0.0 for _ in range(direct_y_dim - num_bids))

        input_L_parts.append(np.zeros(direct_y_dim, dtype=float))
        input_U_parts.append(
            np.concatenate(
                [
                    np.ones(num_bids, dtype=float),
                    np.zeros(direct_y_dim - num_bids, dtype=float),
                ]
            )
        )

    mlp_input.extend(float(v) for v in hg)        # [2*dh]
    input_L_parts.append(hg.astype(float))
    input_U_parts.append(hg.astype(float))

    input_L = np.concatenate(input_L_parts)
    input_U = np.concatenate(input_U_parts)

    fallback_input_dim = (4 * dh) + direct_y_dim if include_direct_y else 4 * dh
    expected_input_dim = int(getattr(getattr(gnn_model, "readout", None), "out_dim", fallback_input_dim))
    if len(mlp_input) != expected_input_dim:
        variant_name = "direct_y" if include_direct_y else "pooled_only"
        raise RuntimeError(
            f"CA {variant_name} MLP input length mismatch: "
            f"expected {expected_input_dim}, got {len(mlp_input)}."
        )
    if input_L.shape[0] != len(mlp_input) or input_U.shape[0] != len(mlp_input):
        raise RuntimeError(
            "CA pooled/direct_y IBP input bounds length mismatch with assembled "
            f"MLP input ({input_L.shape[0]}, {input_U.shape[0]}) vs {len(mlp_input)}."
        )

    energy_var = _add_head_objective(m, gnn_model, mlp_input, input_L, input_U)
    return _solve_and_score_with_details(
        m,
        z,
        gnn_model,
        container,
        energy_var,
        encoder_time_sec=encoder_time,
        return_details=return_details,
    )


def _solve_instance_ca_sum_only_plus(
    gnn_model,
    container,
    meta,
    time_limit=PAPER_TIME_LIMIT,
    enforce_problem_constraints=True,
    ca_k_max=None,
    fixed_assignments=None,
    solver_threads=PAPER_SOLVER_THREADS,
    mip_gap=None,
    return_details=False,
):
    encoder_start = time.perf_counter()
    with torch.no_grad():
        enc_out = gnn_model.encoder(container)
        hc = enc_out["hc"].cpu().numpy()          # [nC, dh]
        hm = enc_out["hm"].cpu().numpy()          # [nBids, dh]
        hg = enc_out["hg"].cpu().numpy()          # [2*dh]
        xm_raw = enc_out["Xm_raw"].cpu().numpy()  # [nBids, dm]
        xc_raw = enc_out["Xc_raw"].cpu().numpy()  # [nC, dc]
        dh = hm.shape[1]
        num_bids = len(meta.M)
        nC = int(container["nC"])
        A_csr = meta.globals["A_csr"]
    encoder_time = time.perf_counter() - encoder_start

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    m = gp.Model("Exact_GNN_CA_SumOnlyPlus", env=env)
    m.setParam("TimeLimit", time_limit)
    if solver_threads is not None:
        m.setParam("Threads", int(solver_threads))
    if mip_gap is not None:
        m.setParam("MIPGap", float(mip_gap))

    z = [m.addVar(vtype=GRB.BINARY, name=f"z_{j}") for j in range(num_bids)]
    _apply_binary_fixes(m, z, fixed_assignments, "fixed_z")
    if enforce_problem_constraints:
        _add_ca_constraints(m, z, A_csr, k_max=ca_k_max)

    selected_pool_vars = []
    selected_pool_L = []
    selected_pool_U = []
    for d in range(dh):
        y = m.addVar(lb=-GRB.INFINITY, name=f"selected_pool_{d}")
        m.addConstr(
            y == gp.quicksum(float(hm[j, d]) * z[j] for j in range(num_bids)),
            name=f"selected_pool_eq_{d}",
        )
        selected_pool_vars.append(y)
        lo, hi = _bounds_sum_k(hm[:, d], 0, num_bids)
        selected_pool_L.append(lo)
        selected_pool_U.append(hi)

    overload = None
    conflicted = None
    item_load_coeff = None
    occupied_pool_coeff = None
    if enforce_problem_constraints:
        item_load_coeff = np.asarray(A_csr.sum(axis=0)).ravel().astype(float)
        occupied_pool_coeff = np.asarray(A_csr.T @ hc, dtype=float)
    else:
        occupied, overload, conflicted = _add_ca_occupancy_overload(m, z, A_csr, prefix="sumonlyplus")

    occupied_pool_vars = []
    occupied_pool_L = []
    occupied_pool_U = []
    for d in range(dh):
        y = m.addVar(lb=-GRB.INFINITY, name=f"occupied_pool_{d}")
        if enforce_problem_constraints:
            m.addConstr(
                y == gp.quicksum(float(occupied_pool_coeff[j, d]) * z[j] for j in range(num_bids)),
                name=f"occupied_pool_eq_{d}",
            )
        else:
            m.addConstr(
                y == gp.quicksum(float(hc[i, d]) * occupied[i] for i in range(nC)),
                name=f"occupied_pool_eq_{d}",
            )
        occupied_pool_vars.append(y)
        lo, hi = _bounds_sum_k(hc[:, d], 0, nC)
        occupied_pool_L.append(lo)
        occupied_pool_U.append(hi)

    selected_value_sum = m.addVar(lb=-GRB.INFINITY, name="selected_value_sum")
    m.addConstr(
        selected_value_sum == gp.quicksum(float(xm_raw[j, 0]) * z[j] for j in range(num_bids)),
        name="selected_value_sum_eq",
    )
    selected_bundle_sum = m.addVar(lb=-GRB.INFINITY, name="selected_bundle_sum")
    m.addConstr(
        selected_bundle_sum == gp.quicksum(float(xm_raw[j, 1]) * z[j] for j in range(num_bids)),
        name="selected_bundle_sum_eq",
    )
    value_sum_L, value_sum_U = _bounds_sum_k(xm_raw[:, 0], 0, num_bids)
    bundle_sum_L, bundle_sum_U = _bounds_sum_k(xm_raw[:, 1], 0, num_bids)
    selected_conflict_risk_sum = m.addVar(lb=-GRB.INFINITY, name="selected_conflict_risk_sum")
    m.addConstr(
        selected_conflict_risk_sum == gp.quicksum(float(xm_raw[j, 3]) * z[j] for j in range(num_bids)),
        name="selected_conflict_risk_sum_eq",
    )
    conflict_risk_sum_L, conflict_risk_sum_U = _bounds_sum_k(xm_raw[:, 3], 0, num_bids)

    is_dummy = xc_raw[:, 1].astype(float) if xc_raw.shape[1] > 1 else np.zeros(nC, dtype=float)
    dummy_count = float(is_dummy.sum())
    dummy_denom = max(dummy_count, 1.0)
    occupied_dummy_coeff = (
        np.asarray(A_csr.T @ is_dummy, dtype=float).ravel()
        if enforce_problem_constraints
        else None
    )

    selected_frac = m.addVar(lb=0.0, ub=1.0, name="selected_frac")
    occupied_frac = m.addVar(lb=0.0, ub=1.0, name="occupied_frac")
    conflict_item_frac = m.addVar(lb=0.0, ub=1.0, name="conflict_item_frac")
    conflict_mass_ub = float(sum(max(int(A_csr.getrow(i).nnz) - 1, 0) for i in range(nC))) / max(float(nC), 1.0)
    conflict_mass_norm = m.addVar(lb=0.0, ub=conflict_mass_ub, name="conflict_mass_norm")
    occupied_dummy_frac = m.addVar(
        lb=0.0,
        ub=0.0 if dummy_count <= 0.0 else 1.0,
        name="occupied_dummy_frac",
    )
    selected_sum = gp.quicksum(z)
    occupied_sum = (
        gp.quicksum(float(item_load_coeff[j]) * z[j] for j in range(num_bids))
        if enforce_problem_constraints
        else gp.quicksum(occupied)
    )
    m.addConstr(selected_frac == (1.0 / max(float(num_bids), 1.0)) * selected_sum, name="selected_frac_eq")
    m.addConstr(
        occupied_frac == (1.0 / max(float(nC), 1.0)) * occupied_sum,
        name="occupied_frac_eq",
    )
    occupied_dummy_sum = (
        gp.quicksum(float(occupied_dummy_coeff[j]) * z[j] for j in range(num_bids))
        if enforce_problem_constraints
        else gp.quicksum(float(is_dummy[i]) * occupied[i] for i in range(nC))
    )
    m.addConstr(
        occupied_dummy_frac
        == (1.0 / dummy_denom) * occupied_dummy_sum,
        name="occupied_dummy_frac_eq",
    )
    if enforce_problem_constraints:
        m.addConstr(conflict_item_frac == 0.0, name="conflict_item_frac_zero")
        m.addConstr(conflict_mass_norm == 0.0, name="conflict_mass_norm_zero")
    else:
        m.addConstr(
            conflict_item_frac == (1.0 / max(float(nC), 1.0)) * gp.quicksum(conflicted),
            name="conflict_item_frac_eq",
        )
        m.addConstr(
            conflict_mass_norm == (1.0 / max(float(nC), 1.0)) * gp.quicksum(overload),
            name="conflict_mass_norm_eq",
        )

    mlp_input = []
    mlp_input.extend(selected_pool_vars)          # [dh]
    mlp_input.extend(occupied_pool_vars)          # [dh]
    mlp_input.extend(float(v) for v in hg)        # [2*dh]
    mlp_input.extend(
        [
            selected_frac,
            occupied_frac,
            occupied_dummy_frac,
            conflict_item_frac,
            conflict_mass_norm,
            selected_value_sum,
            selected_bundle_sum,
            selected_conflict_risk_sum,
        ]
    )

    input_L = np.concatenate(
        [
            np.asarray(selected_pool_L, dtype=float),
            np.asarray(occupied_pool_L, dtype=float),
            hg.astype(float),
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, value_sum_L, bundle_sum_L, conflict_risk_sum_L], dtype=float),
        ]
    )
    input_U = np.concatenate(
        [
            np.asarray(selected_pool_U, dtype=float),
            np.asarray(occupied_pool_U, dtype=float),
            hg.astype(float),
            np.array(
                [
                    1.0,
                    1.0,
                    0.0 if dummy_count <= 0.0 else 1.0,
                    1.0,
                    conflict_mass_ub,
                    value_sum_U,
                    bundle_sum_U,
                    conflict_risk_sum_U,
                ],
                dtype=float,
            ),
        ]
    )

    expected_input_dim = int(getattr(getattr(gnn_model, "readout", None), "out_dim", (4 * dh) + 8))
    if len(mlp_input) != expected_input_dim:
        raise RuntimeError(
            f"CA sum_only_plus MLP input length mismatch: expected {expected_input_dim}, got {len(mlp_input)}."
        )
    if input_L.shape[0] != len(mlp_input) or input_U.shape[0] != len(mlp_input):
        raise RuntimeError(
            "CA sum_only_plus IBP input bounds length mismatch with assembled MLP input "
            f"({input_L.shape[0]}, {input_U.shape[0]}) vs {len(mlp_input)}."
        )

    energy_var = _add_head_objective(m, gnn_model, mlp_input, input_L, input_U)
    return _solve_and_score_with_details(
        m,
        z,
        gnn_model,
        container,
        energy_var,
        encoder_time_sec=encoder_time,
        return_details=return_details,
    )


def _solve_instance_qkp(
    gnn_model,
    container,
    meta,
    time_limit=PAPER_TIME_LIMIT,
    enforce_problem_constraints=True,
    qkp_k_min=None,
    qkp_k_max=None,
    solver_threads=PAPER_SOLVER_THREADS,
    mip_gap=None,
):
    with torch.no_grad():
        enc_out = gnn_model.encoder(container)
        hm = enc_out["hm"].cpu().numpy()  # [nItems, dh]
        hg = enc_out["hg"].cpu().numpy()  # [2*dh]
        dh = hm.shape[1]
        num_items = len(meta.M)
        weights = np.asarray(meta.globals["w"], dtype=float).ravel()
        capacity = float(meta.globals["W"])
        readout_variant = resolve_qkp_readout_variant(
            "qkp",
            getattr(gnn_model, "qkp_readout_variant", "auto"),
        )

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    m = gp.Model("Exact_GNN_QKP", env=env)
    m.setParam("TimeLimit", time_limit)
    if solver_threads is not None:
        m.setParam("Threads", int(solver_threads))
    if mip_gap is not None:
        m.setParam("MIPGap", float(mip_gap))

    z = [m.addVar(vtype=GRB.BINARY, name=f"z_{j}") for j in range(num_items)]
    if enforce_problem_constraints:
        m.addConstr(
            gp.quicksum(float(weights[j]) * z[j] for j in range(num_items)) <= float(capacity),
            name="Capacity",
        )
        qkp_lo, qkp_hi = _qkp_cardinality_bounds(meta, k_min=qkp_k_min, k_max=qkp_k_max)
        if qkp_lo > 0:
            m.addConstr(gp.quicksum(z) >= qkp_lo, name="QKPCardinalityMin")
        if qkp_hi < num_items:
            m.addConstr(gp.quicksum(z) <= qkp_hi, name="QKPCardinalityMax")

    inv_sqrt_n = 1.0 / math.sqrt(max(float(num_items), 1.0))
    pool_coeffs = hm * inv_sqrt_n if readout_variant == "size_aware" else hm

    selected_pool_vars = []
    selected_pool_L = []
    selected_pool_U = []
    for d in range(dh):
        y = m.addVar(lb=-GRB.INFINITY, name=f"pool_sum_{d}")
        m.addConstr(
            y == gp.quicksum(float(pool_coeffs[j, d]) * z[j] for j in range(num_items)),
            name=f"pool_sum_eq_{d}",
        )
        selected_pool_vars.append(y)
        if enforce_problem_constraints:
            lo, hi = _bounds_knapsack_fractional(pool_coeffs[:, d], weights, capacity)
        else:
            lo, hi = _bounds_sum_k(pool_coeffs[:, d], 0, num_items)
        selected_pool_L.append(lo)
        selected_pool_U.append(hi)

    mlp_input = []
    input_L_parts = [np.asarray(selected_pool_L, dtype=float)]
    input_U_parts = [np.asarray(selected_pool_U, dtype=float)]
    mlp_input.extend(selected_pool_vars)

    if readout_variant == "size_aware":
        total_pool = np.sum(pool_coeffs, axis=0)
        unselected_pool_vars = []
        unselected_pool_L = []
        unselected_pool_U = []
        for d in range(dh):
            y = m.addVar(lb=-GRB.INFINITY, name=f"pool_unselected_{d}")
            m.addConstr(
                y == float(total_pool[d]) - selected_pool_vars[d],
                name=f"pool_unselected_eq_{d}",
            )
            unselected_pool_vars.append(y)
            unselected_pool_L.append(float(total_pool[d]) - float(selected_pool_U[d]))
            unselected_pool_U.append(float(total_pool[d]) - float(selected_pool_L[d]))

        selected_frac = m.addVar(lb=0.0, ub=1.0, name="selected_frac")
        selected_frac_scale = 1.0 / max(float(num_items), 1.0)
        m.addConstr(
            selected_frac == selected_frac_scale * gp.quicksum(z),
            name="selected_frac_eq",
        )

        if abs(capacity) > 1e-12:
            weight_mass = m.addVar(lb=0.0, name="weight_mass")
            weight_scale = 1.0 / float(capacity)
            m.addConstr(
                weight_mass == gp.quicksum(float(weights[j]) * weight_scale * z[j] for j in range(num_items)),
                name="weight_mass_eq",
            )
            weight_mass_ub = 1.0 if enforce_problem_constraints else float(np.sum(weights) * weight_scale)
        else:
            weight_mass = m.addVar(lb=0.0, ub=0.0, name="weight_mass")
            m.addConstr(weight_mass == 0.0, name="weight_mass_eq")
            weight_mass_ub = 0.0

        mlp_input.extend(unselected_pool_vars)
        mlp_input.append(selected_frac)
        mlp_input.append(weight_mass)
        input_L_parts.extend(
            [
                np.asarray(unselected_pool_L, dtype=float),
                np.asarray([0.0, 0.0], dtype=float),
            ]
        )
        input_U_parts.extend(
            [
                np.asarray(unselected_pool_U, dtype=float),
                np.asarray([1.0, weight_mass_ub], dtype=float),
            ]
        )

    if readout_variant == "direct_y":
        direct_y_dim = int(getattr(getattr(gnn_model, "readout", None), "direct_y_dim", 0))
        if direct_y_dim <= 0:
            raise RuntimeError("QKP direct_y readout is missing a positive direct_y_dim.")
        if num_items > direct_y_dim:
            raise RuntimeError(
                f"QKP direct_y_dim={direct_y_dim} is smaller than the instance "
                f"width {num_items}."
            )

        mlp_input.extend(z)
        if direct_y_dim > num_items:
            mlp_input.extend(0.0 for _ in range(direct_y_dim - num_items))

        input_L_parts.append(np.zeros(direct_y_dim, dtype=float))
        input_U_parts.append(
            np.concatenate(
                [
                    np.ones(num_items, dtype=float),
                    np.zeros(direct_y_dim - num_items, dtype=float),
                ]
            )
        )

    mlp_input.extend(float(v) for v in hg)  # [2*dh]

    input_L_parts.append(hg.astype(float))
    input_U_parts.append(hg.astype(float))
    input_L = np.concatenate(input_L_parts)
    input_U = np.concatenate(input_U_parts)

    if readout_variant == "size_aware":
        fallback_input_dim = (4 * dh) + 2
    elif readout_variant == "direct_y":
        fallback_input_dim = (3 * dh) + int(
            getattr(getattr(gnn_model, "readout", None), "direct_y_dim", num_items)
        )
    else:
        fallback_input_dim = 3 * dh
    expected_input_dim = int(getattr(getattr(gnn_model, "readout", None), "out_dim", fallback_input_dim))
    if len(mlp_input) != expected_input_dim:
        raise RuntimeError(
            f"QKP MLP input length mismatch: expected {expected_input_dim}, got {len(mlp_input)}."
        )
    if input_L.shape[0] != len(mlp_input) or input_U.shape[0] != len(mlp_input):
        raise RuntimeError(
            "QKP IBP input bounds length mismatch with assembled MLP input "
            f"({input_L.shape[0]}, {input_U.shape[0]}) vs {len(mlp_input)}."
        )

    energy_var = _add_head_objective(m, gnn_model, mlp_input, input_L, input_U)
    return _solve_and_score(m, z, gnn_model, container, energy_var)






def _solve_true_qkp_milp(
    meta,
    time_limit=PAPER_TIME_LIMIT,
    fixed_assignments=None,
    k_min=None,
    k_max=None,
    solver_threads=PAPER_SOLVER_THREADS,
    mip_gap=None,
):
    p = np.asarray(meta.globals["p"], dtype=float).ravel()
    Q = np.asarray(meta.globals["Q"], dtype=float)
    w = np.asarray(meta.globals["w"], dtype=float).ravel()
    capacity = float(meta.globals["W"])
    n_items = int(len(meta.M))

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    m = gp.Model("True_QKP", env=env)
    m.setParam("TimeLimit", float(time_limit))
    if solver_threads is not None:
        m.setParam("Threads", int(solver_threads))
    if mip_gap is not None:
        m.setParam("MIPGap", float(mip_gap))

    z = [m.addVar(vtype=GRB.BINARY, name=f"z_{j}") for j in range(n_items)]
    _apply_binary_fixes(m, z, fixed_assignments, "fixed_z")
    m.addConstr(
        gp.quicksum(float(w[j]) * z[j] for j in range(n_items)) <= capacity,
        name="Capacity",
    )
    lo_k, hi_k = _qkp_cardinality_bounds(meta, k_min=k_min, k_max=k_max)
    if lo_k > 0:
        m.addConstr(gp.quicksum(z) >= lo_k, name="QKPCardinalityMin")
    if hi_k < n_items:
        m.addConstr(gp.quicksum(z) <= hi_k, name="QKPCardinalityMax")

    obj = gp.quicksum(float(p[j]) * z[j] for j in range(n_items))
    for i in range(n_items):
        diag = 0.5 * float(Q[i, i]) if i < Q.shape[0] and i < Q.shape[1] else 0.0
        if abs(diag) > 1e-12:
            obj += diag * z[i]
        for j in range(i + 1, n_items):
            coef = 0.5 * (float(Q[i, j]) + float(Q[j, i]))
            if abs(coef) > 1e-12:
                obj += coef * z[i] * z[j]
    m.setObjective(obj, GRB.MAXIMIZE)
    m.optimize()

    if m.SolCount <= 0:
        status_map = {
            GRB.INFEASIBLE: "INFEASIBLE",
            GRB.INF_OR_UNBD: "INF_OR_UNBD",
            GRB.UNBOUNDED: "UNBOUNDED",
            GRB.TIME_LIMIT: "TIME_LIMIT_NO_SOLUTION",
        }
        return None, None, status_map.get(m.Status, f"STATUS_{m.Status}"), float(m.Runtime)

    mask = [int(v.X + 0.5) for v in z]
    obj_val = float(m.ObjVal)
    status = "OPTIMAL" if m.Status == GRB.OPTIMAL else "SUBOPTIMAL"
    return mask, obj_val, status, float(m.Runtime)


def _solve_true_ca_milp(
    meta,
    time_limit=PAPER_TIME_LIMIT,
    fixed_assignments=None,
    k_max=None,
    solver_threads=PAPER_SOLVER_THREADS,
    mip_gap=None,
    return_details=False,
):
    A_csr = meta.globals["A_csr"]
    costs = np.asarray(meta.globals["costs"], dtype=float).ravel()
    num_bids = int(len(meta.M))

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    m = gp.Model("True_CA", env=env)
    m.setParam("TimeLimit", float(time_limit))
    if solver_threads is not None:
        m.setParam("Threads", int(solver_threads))
    if mip_gap is not None:
        m.setParam("MIPGap", float(mip_gap))

    z = [m.addVar(vtype=GRB.BINARY, name=f"z_{j}") for j in range(num_bids)]
    _apply_binary_fixes(m, z, fixed_assignments, "fixed_z")
    _add_ca_constraints(m, z, A_csr, k_max=k_max)
    m.setObjective(
        gp.quicksum(float(costs[j]) * z[j] for j in range(num_bids)),
        GRB.MAXIMIZE,
    )
    m.optimize()

    if m.SolCount <= 0:
        status_map = {
            GRB.INFEASIBLE: "INFEASIBLE",
            GRB.INF_OR_UNBD: "INF_OR_UNBD",
            GRB.UNBOUNDED: "UNBOUNDED",
            GRB.TIME_LIMIT: "TIME_LIMIT_NO_SOLUTION",
        }
        status = status_map.get(m.Status, f"STATUS_{m.Status}")
        if return_details:
            try:
                bound = float(m.ObjBound)
            except Exception:
                bound = None
            return {
                "configuration": None,
                "objective": None,
                "status": status,
                "runtime": float(m.Runtime),
                "bound": bound,
                "proven_optimal": False,
                "solver_gap": None,
                "sol_count": int(m.SolCount),
                "gurobi_status_code": int(m.Status),
            }
        return None, None, status, float(m.Runtime)

    mask = [int(v.X + 0.5) for v in z]
    obj_val = float(m.ObjVal)
    status = "OPTIMAL" if m.Status == GRB.OPTIMAL else "SUBOPTIMAL"
    if return_details:
        try:
            bound = float(m.ObjBound)
        except Exception:
            bound = None
        try:
            solver_gap = float(m.MIPGap)
        except Exception:
            solver_gap = None
        return {
            "configuration": mask,
            "objective": obj_val,
            "status": status,
            "runtime": float(m.Runtime),
            "bound": bound,
            "proven_optimal": bool(m.Status == GRB.OPTIMAL),
            "solver_gap": solver_gap,
            "sol_count": int(m.SolCount),
            "gurobi_status_code": int(m.Status),
        }
    return mask, obj_val, status, float(m.Runtime)


def _solve_true_cart_milp(
    meta,
    min_k=0,
    max_k=0,
    time_limit=PAPER_TIME_LIMIT,
    fixed_mask=None,
    fixed_assignments=None,
    solver_threads=PAPER_SOLVER_THREADS,
    mip_gap=None,
    return_details=False,
):
    """Solve the detailed patient-level CART model."""
    record = _solve_cart_detailed_milp(
        meta, mask=fixed_mask, min_open_facilities=min_k,
        max_open_facilities=max_k if max_k > 0 else None,
        time_limit=time_limit, solver_threads=solver_threads, mip_gap=mip_gap,
        fixed_assignments=fixed_assignments,
    )
    opened = set(record.get("opened_facilities") or [])
    objective = record.get("objective")
    mask = [int(m in opened) for m in meta.M] if objective is not None else None
    status = ("OPTIMAL" if record.get("proven_optimal") else "SUBOPTIMAL"
              if objective is not None else
              "TIME_LIMIT_NO_SOLUTION" if record.get("status") == "time_limit"
              else str(record.get("status", "unknown")).upper())
    runtime = float(record.get("solver_time_sec") or 0.0)
    if return_details:
        return {
            "configuration": mask, "objective": objective, "status": status,
            "runtime": runtime, "bound": record.get("best_bound"),
            "proven_optimal": bool(record.get("proven_optimal")),
            "solver_gap": record.get("mip_gap_rel"),
            "sol_count": int(objective is not None),
            "gurobi_status_code": record.get("gurobi_status_code"),
        }
    return mask, objective, status, runtime


def _requires_exact_reference(args, sample_reference=None) -> bool:
    if (
        args.problem == "cart"
        and sample_reference is not None
        and sample_reference.get("objective_model") == "cart_detailed"
    ):
        # Learned predictions use the detailed reference and mask-scoring path below.
        return args.model_type == "exact_solver"
    if args.model_type == "exact_solver":
        return True
    if args.problem == "ca":
        return args.ca_k_max is not None
    if args.problem == "cart":
        return bool(args.cart_exact_reference or args.cart_force_off or args.cart_force_on
                    or getattr(args, "min_k", 0) > 0 or getattr(args, "max_k", 0) > 0)
    if args.problem == "qkp":
        return args.qkp_k_min is not None or args.qkp_k_max is not None
    return False


def _exact_reference_source_name(problem: str) -> str:
    problem = _normalize_problem_name(problem)
    if problem == "cart":
        return "exact_cart_milp"
    if problem == "qkp":
        return "exact_qkp_milp"
    if problem == "ca":
        return "exact_ca_milp"
    return f"exact_{problem}_milp"


def _cart_detailed_data_from_meta(meta):
    from cart_dataset.model_cart_detailed import build_detailed_data_from_aggregate

    inc_param = {
        (str(c), int(day)): int(value)
        for c, days in meta.INC.items()
        for day, value in days.items()
    }
    static_params = {
        "CAPMAX": {str(m): float(meta.CAPMAX[m]) for m in meta.M},
        "CIM": {str(m): float(meta.CIM[m]) for m in meta.M},
        "CVM": {str(m): float(meta.CVM[m]) for m in meta.M},
        "U1": {
            (str(c), str(m)): float(meta.U1[c][m])
            for c in meta.C
            for m in meta.M
        },
        "TMFE": int(meta.globals.get("TMFE", CART_TMFE)),
    }
    return build_detailed_data_from_aggregate(
        {"c": [str(c) for c in meta.C], "m": [str(m) for m in meta.M]},
        static_params,
        inc_param,
        time_horizon=int(meta.T),
    )


def _solve_cart_detailed_milp(
    meta,
    mask=None,
    *,
    time_limit=PAPER_TIME_LIMIT,
    solver_threads=PAPER_SOLVER_THREADS,
    mip_gap=None,
    max_open_facilities=None,
    min_open_facilities=None,
    fixed_assignments=None,
    formulation="detailed",
    conpas_x0_indices=None,
    conpas_x1_indices=None,
    conpas_delta=None,
    tee=False,
):
    from cart_dataset.candidate_scoring import resolve_formulation, solve_mask

    data = _cart_detailed_data_from_meta(meta)
    resolve_formulation(formulation)
    return solve_mask(
        data,
        mask=mask,
        time_horizon=int(meta.T),
        max_open_facilities=max_open_facilities,
        min_open_facilities=min_open_facilities,
        forced_e1_values={
            str(meta.M[idx]): bit
            for idx, bit in _normalize_fixed_assignments(fixed_assignments, len(meta.M)).items()
        },
        time_limit=time_limit,
        threads=solver_threads,
        mip_gap=mip_gap,
        tee=tee,
        formulation="detailed",
        conpas_x0_facilities=None if conpas_x0_indices is None else
        [str(meta.M[idx]) for idx in conpas_x0_indices],
        conpas_x1_facilities=None if conpas_x1_indices is None else
        [str(meta.M[idx]) for idx in conpas_x1_indices],
        conpas_delta=conpas_delta,
    )


def _cart_detailed_reference_stats_from_record(meta, record, *, require_optimal=False):
    if require_optimal and not (record or {}).get("proven_optimal", False):
        return None
    obj = None if record is None else record.get("objective")
    opened = set((record or {}).get("opened_facilities") or [])
    if obj is None or not opened:
        return None
    mask_key = tuple(1 if m in opened else 0 for m in meta.M)
    return {
        "best_obj": float(obj),
        "best_mask": [int(v) for v in mask_key],
        "optimal_masks": {mask_key},
        "feasible_obj_by_mask": {mask_key: float(obj)},
        "objective_model": "cart_detailed",
    }


def _apply_cart_detailed_prediction_record(pred_stats, ref_stats, record):
    obj = None if record is None else record.get("objective")
    if obj is not None:
        updated = {
            **pred_stats,
            "feasible": True,
            "objective_source": f"{record.get('formulation', 'detailed')}_cart_milp",
            "objective": float(obj),
        }
        ref_stats["feasible_obj_by_mask"][updated["mask_key"]] = float(obj)
        return updated
    if record is not None and record.get("feasible") is False:
        return {
            **pred_stats,
            "feasible": False,
            "objective_source": f"{record.get('formulation', 'detailed')}_cart_milp",
            "objective": None,
        }
    return pred_stats


def _solve_exact_reference_milp(args, meta, cart_fixed_assignments=None):
    if args.problem == "ca":
        return _solve_true_ca_milp(
            meta,
            time_limit=args.time_limit,
            k_max=args.ca_k_max,
            solver_threads=args.solver_threads,
            mip_gap=args.mip_gap,
        )
    if args.problem == "qkp":
        return _solve_true_qkp_milp(
            meta,
            time_limit=args.time_limit,
            k_min=args.qkp_k_min,
            k_max=args.qkp_k_max,
            solver_threads=args.solver_threads,
            mip_gap=args.mip_gap,
        )
    return _solve_true_cart_milp(
        meta,
        min_k=args.min_k,
        max_k=args.max_k,
        time_limit=args.time_limit,
        fixed_assignments=cart_fixed_assignments,
        solver_threads=args.solver_threads,
        mip_gap=args.mip_gap,
    )




def _solve_instance_gnn_fixing(
    gnn_model,
    container,
    meta,
    problem="cart",
    min_k=0,
    max_k=0,
    time_limit=PAPER_TIME_LIMIT,
    solver_threads=PAPER_SOLVER_THREADS,
    mip_gap=None,
    enforce_problem_constraints=True,
    ca_k_max=None,
    qkp_k_min=None,
    qkp_k_max=None,
    fixing_ratio=None,
    fixing_threshold_high=None,
    fixing_threshold_low=None,
    fixed_assignments=None,
    n_dives=None,
    parallel_dives=False,
    dive_workers=None,
):
    if not enforce_problem_constraints:
        raise NotImplementedError(
            "GNN Fixing + MIP requires the original hard constraints because it solves "
            "the reduced benchmark MIP after variable fixing."
        )

    problem = _normalize_problem_name(problem)
    model_type = str(getattr(gnn_model, "model_type", "gnn_fixing_mip"))
    fixing_defaults = resolve_fixing_inference_defaults(
        problem,
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

    logits = _variable_scores_np(gnn_model, container)
    probs = _sigmoid_np(logits)
    hard_pred_mask = [int(v >= 0.5) for v in probs.tolist()]

    n_dives = max(1, int(n_dives))
    # Per-dive time limit: split budget evenly across dives
    dive_time_limit = float(time_limit) / n_dives
    use_parallel_dives = bool(parallel_dives) and n_dives > 1
    if dive_workers is None:
        resolved_dive_workers = n_dives
    else:
        resolved_dive_workers = int(dive_workers)
    if resolved_dive_workers <= 0:
        raise ValueError(f"--dive_workers must be positive, got {resolved_dive_workers}.")
    resolved_dive_workers = min(resolved_dive_workers, n_dives)

    user_fixed_assignments = _normalize_fixed_assignments(fixed_assignments, probs.size)

    def _build_fixing_assignments(sampled_y):
        """Return fixing dict for one dive given a binary sample vector."""
        if problem == "cart":
            fixes = {}
            for idx, (prob, sy) in enumerate(zip(probs.tolist(), sampled_y.tolist())):
                if prob >= float(fixing_threshold_high):
                    fixes[idx] = int(sy)
                elif prob <= float(fixing_threshold_low):
                    fixes[idx] = int(sy)
            return fixes
        # For non-CART: fix the top fixing_ratio fraction by confidence to sampled value
        ratio = min(max(float(fixing_ratio), 0.0), 1.0)
        n_fix = min(probs.size, int(round(ratio * probs.size)))
        if n_fix <= 0:
            return {}
        confidence = np.abs(probs - 0.5)
        order = np.argsort(-confidence, kind="stable")
        return {int(idx): int(sampled_y[idx]) for idx in order[:n_fix].tolist()}

    def _run_sub_milp(solver_fixed_assignments):
        if problem == "ca":
            return _solve_true_ca_milp(
                meta, time_limit=dive_time_limit,
                fixed_assignments=solver_fixed_assignments, k_max=ca_k_max,
                solver_threads=solver_threads, mip_gap=mip_gap,
            )
        elif problem == "qkp":
            return _solve_true_qkp_milp(
                meta, time_limit=dive_time_limit,
                fixed_assignments=solver_fixed_assignments,
                k_min=qkp_k_min, k_max=qkp_k_max,
                solver_threads=solver_threads, mip_gap=mip_gap,
            )
        else:
            return _solve_true_cart_milp(
                meta, min_k=min_k, max_k=max_k, time_limit=dive_time_limit,
                fixed_assignments=solver_fixed_assignments,
                solver_threads=solver_threads, mip_gap=mip_gap,
            )

    best_mask = None
    best_energy = None
    best_status = "INFEASIBLE"
    best_num_fixed = 0
    last_runtime = 0.0
    dive_results = []
    dive_jobs = []

    for dive_idx in range(n_dives):
        if n_dives == 1:
            # Deterministic dive (original behaviour): fix by hard threshold/ratio
            sampled_y = (probs >= 0.5).astype(int)
        else:
            # Stochastic dive (Neural Diving): sample y_i ~ Bernoulli(p_i)
            sampled_y = np.random.binomial(1, probs).astype(int)

        solver_fixed_assignments = _build_fixing_assignments(sampled_y)
        solver_fixed_assignments.update(user_fixed_assignments)
        dive_jobs.append((int(dive_idx), dict(solver_fixed_assignments)))

    def _run_dive_job(dive_idx, solver_fixed_assignments):
        dive_start = time.perf_counter()
        mask, obj, status, runtime_sec = _run_sub_milp(solver_fixed_assignments)
        return {
            "dive_idx": int(dive_idx),
            "mask": mask,
            "objective": obj,
            "status": status,
            "solver_runtime_sec": float(runtime_sec or 0.0),
            "wall_time_sec": float(time.perf_counter() - dive_start),
            "num_fixed_variables": int(len(solver_fixed_assignments)),
        }

    dive_phase_start = time.perf_counter()
    if use_parallel_dives:
        with ThreadPoolExecutor(max_workers=resolved_dive_workers) as executor:
            future_to_idx = {
                executor.submit(_run_dive_job, dive_idx, assignments): dive_idx
                for dive_idx, assignments in dive_jobs
            }
            for future in as_completed(future_to_idx):
                dive_results.append(future.result())
    else:
        for dive_idx, assignments in dive_jobs:
            dive_results.append(_run_dive_job(dive_idx, assignments))
    dive_wall_time = float(time.perf_counter() - dive_phase_start)

    for dive in sorted(dive_results, key=lambda item: int(item["dive_idx"])):
        last_runtime = float(dive.get("solver_runtime_sec") or 0.0)
        obj = dive.get("objective")
        if dive.get("mask") is not None and obj is not None:
            energy_value = _objective_to_energy(meta, obj)
            # Energy is always lower-is-better: maximization objectives are
            # stored as negative energies by _objective_to_energy.
            is_better = best_energy is None or energy_value < best_energy
            if is_better:
                best_mask = dive.get("mask")
                best_energy = energy_value
                best_status = dive.get("status")
                best_num_fixed = int(dive.get("num_fixed_variables") or 0)
        elif best_mask is None:
            best_status = dive.get("status") or best_status

    dive_solver_runtimes = [float(d.get("solver_runtime_sec") or 0.0) for d in dive_results]
    dive_wall_times = [float(d.get("wall_time_sec") or 0.0) for d in dive_results]

    extra = {
        "variable_logits": [float(v) for v in logits.tolist()],
        "variable_probabilities": [float(v) for v in probs.tolist()],
        "predicted_y": [int(v) for v in hard_pred_mask],
        "num_fixed_variables": best_num_fixed,
        "fallback_count": 0,
        "n_dives": n_dives,
        "fixing_ratio_used": float(fixing_ratio),
        "fixing_threshold_high": float(fixing_threshold_high),
        "fixing_threshold_low": float(fixing_threshold_low),
        "parallel_dives": bool(use_parallel_dives),
        "dive_workers": int(resolved_dive_workers),
        "dive_time_limit": float(dive_time_limit),
        "dive_solver_runtime_sum_sec": float(sum(dive_solver_runtimes)),
        "dive_solver_runtime_max_sec": float(max(dive_solver_runtimes) if dive_solver_runtimes else 0.0),
        "dive_wall_time_sec": float(dive_wall_time),
        "dive_wall_time_max_sec": float(max(dive_wall_times) if dive_wall_times else 0.0),
        "dive_statuses": [str(d.get("status")) for d in sorted(dive_results, key=lambda item: int(item["dive_idx"]))],
        "dive_objectives": [
            None if d.get("objective") is None else float(d.get("objective"))
            for d in sorted(dive_results, key=lambda item: int(item["dive_idx"]))
        ],
        "solver_runtime_sec": float(dive_wall_time if use_parallel_dives else sum(dive_solver_runtimes)),
        "last_dive_solver_runtime_sec": float(last_runtime),
    }
    if best_mask is not None and best_energy is not None:
        return best_mask, best_energy, best_energy, best_status, extra
    return None, None, None, best_status, extra


# ---------------------------------------------------------------------------
# ConPaS-style (shared encoder) inference helpers
# ---------------------------------------------------------------------------

def _conpas_rank_defaults(problem: str, n_vars: int, benchmark_name: str = ""):
    problem = _normalize_problem_name(problem)
    key = str(benchmark_name or "").replace("\\", "/").lower()
    n_vars = int(n_vars)
    if problem == "ca" and ("100x175" in key or n_vars == 175):
        return 44, 0, 1
    if problem == "qkp" and ("qkp150" in key or n_vars == 150):
        return 55, 20, 5
    if problem == "cart" and ("cart_m16" in key or n_vars == 16):
        return 8, 5, 3
    return None


def _resolve_conpas_rank_config(
    problem: str,
    n_vars: int,
    *,
    benchmark_name: str = "",
    k0=None,
    k1=None,
    delta=None,
):
    defaults = _conpas_rank_defaults(problem, n_vars, benchmark_name)
    if defaults is None and (k0 is None or k1 is None or delta is None):
        raise ValueError(
            "No default ConPaS rank-neighbourhood configuration is defined for "
            f"problem={problem!r}, n_vars={int(n_vars)}, benchmark={benchmark_name!r}. "
            "Pass --k0, --k1 and --delta explicitly."
        )

    default_k0, default_k1, default_delta = defaults if defaults is not None else (None, None, None)
    resolved_k0 = int(default_k0 if k0 is None else k0)
    resolved_k1 = int(default_k1 if k1 is None else k1)
    resolved_delta = int(default_delta if delta is None else delta)

    if resolved_k0 < 0:
        raise ValueError(f"ConPaS requires k0 >= 0, got {resolved_k0}.")
    if resolved_k1 < 0:
        raise ValueError(f"ConPaS requires k1 >= 0, got {resolved_k1}.")
    if resolved_delta < 0:
        raise ValueError(f"ConPaS requires delta >= 0, got {resolved_delta}.")
    if resolved_k0 + resolved_k1 > int(n_vars):
        raise ValueError(
            f"ConPaS requires k0 + k1 <= n, got k0={resolved_k0}, "
            f"k1={resolved_k1}, n={int(n_vars)}."
        )
    if resolved_delta > resolved_k0 + resolved_k1:
        warnings.warn(
            "ConPaS delta is larger than k0 + k1, so the rank-neighbourhood "
            "constraint is redundant.",
            RuntimeWarning,
            stacklevel=2,
        )
    return resolved_k0, resolved_k1, resolved_delta


def _conpas_rank_sets(probabilities, k0: int, k1: int, excluded_indices=None):
    probs = np.asarray(probabilities, dtype=float).reshape(-1)
    n_vars = int(probs.size)
    excluded = set() if excluded_indices is None else {int(j) for j in excluded_indices}
    invalid = sorted(j for j in excluded if j < 0 or j >= n_vars)
    if invalid:
        raise ValueError(f"ConPaS excluded variable indices are out of range: {invalid}.")
    indices = sorted(
        (j for j in range(n_vars) if j not in excluded),
        key=lambda j: (float(probs[j]), int(j)),
    )
    n_rankable = len(indices)
    if int(k0) + int(k1) > n_rankable:
        raise ValueError(
            "ConPaS requires k0 + k1 to fit within the unfixed variables, got "
            f"k0={int(k0)}, k1={int(k1)}, rankable={n_rankable}, "
            f"excluded={len(excluded)}."
        )
    x0_indices = [int(j) for j in indices[: int(k0)]]
    x1_indices = [int(j) for j in indices[n_rankable - int(k1) :]] if int(k1) > 0 else []
    assert set(x0_indices).isdisjoint(x1_indices)
    return x0_indices, x1_indices


def _add_conpas_rank_neighbourhood(
    gurobi_model,
    binary_vars,
    x0_indices,
    x1_indices,
    delta,
    name="conpas_rank_neighbourhood",
):
    """
    Add the rank-based ConPaS neighbourhood:

        sum_{j in X0} x_j + sum_{j in X1} (1 - x_j) <= delta

    Variables outside X0 union X1 are left unrestricted. This preserves the
    original MIP objective and hard constraints.
    """
    if int(delta) < 0:
        raise ValueError(f"delta must be nonnegative, got {delta}.")
    if not x0_indices and not x1_indices:
        return
    lhs = gp.LinExpr()
    for idx in x0_indices:
        lhs += binary_vars[int(idx)]
    for idx in x1_indices:
        lhs += 1.0 - binary_vars[int(idx)]
    gurobi_model.addConstr(lhs <= float(delta), name=name)


def _solve_conpas_ca_milp(
    meta,
    x0_indices,
    x1_indices,
    delta,
    time_limit=PAPER_TIME_LIMIT,
    k_max=None,
    fixed_assignments=None,
    solver_threads=PAPER_SOLVER_THREADS,
    mip_gap=None,
):
    """Solve the CA MIP with the rank-based ConPaS neighbourhood."""
    A_csr  = meta.globals["A_csr"]
    costs  = np.asarray(meta.globals["costs"], dtype=float).ravel()
    n_bids = int(len(meta.M))
    env = gp.Env(empty=True); env.setParam("OutputFlag", 0); env.start()
    m = gp.Model("ConPaS_CA", env=env); m.setParam("TimeLimit", float(time_limit))
    if solver_threads is not None:
        m.setParam("Threads", int(solver_threads))
    if mip_gap is not None:
        m.setParam("MIPGap", float(mip_gap))
    z = [m.addVar(vtype=GRB.BINARY, name=f"z_{j}") for j in range(n_bids)]
    _add_ca_constraints(m, z, A_csr, k_max=k_max)
    _apply_binary_fixes(m, z, fixed_assignments, "user_fixed")
    _add_conpas_rank_neighbourhood(m, z, x0_indices, x1_indices, delta)
    m.setObjective(gp.quicksum(float(costs[j]) * z[j] for j in range(n_bids)), GRB.MAXIMIZE)
    m.optimize()
    if m.SolCount <= 0:
        status_map = {GRB.INFEASIBLE: "INFEASIBLE", GRB.TIME_LIMIT: "TIME_LIMIT_NO_SOLUTION"}
        return None, None, status_map.get(m.Status, f"STATUS_{m.Status}"), float(m.Runtime)
    mask = [int(v.X + 0.5) for v in z]
    return mask, float(m.ObjVal), "OPTIMAL" if m.Status == GRB.OPTIMAL else "SUBOPTIMAL", float(m.Runtime)


def _solve_conpas_qkp_milp(
    meta,
    x0_indices,
    x1_indices,
    delta,
    time_limit=PAPER_TIME_LIMIT,
    k_min=None,
    k_max=None,
    solver_threads=PAPER_SOLVER_THREADS,
    mip_gap=None,
):
    """Solve the QKP MIP with the rank-based ConPaS neighbourhood."""
    p = np.asarray(meta.globals["p"], dtype=float).ravel()
    Q = np.asarray(meta.globals["Q"], dtype=float)
    w = np.asarray(meta.globals["w"], dtype=float).ravel()
    capacity = float(meta.globals["W"])
    n_items  = int(len(meta.M))
    env = gp.Env(empty=True); env.setParam("OutputFlag", 0); env.start()
    m = gp.Model("ConPaS_QKP", env=env); m.setParam("TimeLimit", float(time_limit))
    if solver_threads is not None:
        m.setParam("Threads", int(solver_threads))
    if mip_gap is not None:
        m.setParam("MIPGap", float(mip_gap))
    z = [m.addVar(vtype=GRB.BINARY, name=f"z_{j}") for j in range(n_items)]
    m.addConstr(gp.quicksum(float(w[j]) * z[j] for j in range(n_items)) <= capacity, name="Capacity")
    lo_k, hi_k = _qkp_cardinality_bounds(meta, k_min=k_min, k_max=k_max)
    if lo_k > 0:
        m.addConstr(gp.quicksum(z) >= lo_k, name="QKPCardinalityMin")
    if hi_k < n_items:
        m.addConstr(gp.quicksum(z) <= hi_k, name="QKPCardinalityMax")
    _add_conpas_rank_neighbourhood(m, z, x0_indices, x1_indices, delta)
    obj = gp.quicksum(float(p[j]) * z[j] for j in range(n_items))
    for i in range(n_items):
        diag = 0.5 * float(Q[i, i]) if i < Q.shape[0] and i < Q.shape[1] else 0.0
        if abs(diag) > 1e-12:
            obj += diag * z[i]
        for j in range(i + 1, n_items):
            coef = 0.5 * (float(Q[i, j]) + float(Q[j, i]))
            if abs(coef) > 1e-12:
                obj += coef * z[i] * z[j]
    m.setObjective(obj, GRB.MAXIMIZE)
    m.optimize()
    if m.SolCount <= 0:
        status_map = {GRB.INFEASIBLE: "INFEASIBLE", GRB.TIME_LIMIT: "TIME_LIMIT_NO_SOLUTION"}
        return None, None, status_map.get(m.Status, f"STATUS_{m.Status}"), float(m.Runtime)
    mask = [int(v.X + 0.5) for v in z]
    return mask, float(m.ObjVal), "OPTIMAL" if m.Status == GRB.OPTIMAL else "SUBOPTIMAL", float(m.Runtime)




def _solve_conpas_cart_milp(
    meta,
    x0_indices,
    x1_indices,
    delta,
    min_k=0,
    max_k=0,
    time_limit=PAPER_TIME_LIMIT,
    fixed_assignments=None,
    solver_threads=PAPER_SOLVER_THREADS,
    mip_gap=None,
):
    """Solve detailed CART with the rank-based ConPaS neighbourhood."""
    record = _solve_cart_detailed_milp(
        meta, min_open_facilities=min_k,
        max_open_facilities=max_k if max_k > 0 else None,
        time_limit=time_limit, solver_threads=solver_threads, mip_gap=mip_gap,
        fixed_assignments=fixed_assignments,
        conpas_x0_indices=x0_indices, conpas_x1_indices=x1_indices,
        conpas_delta=delta,
    )
    objective = record.get("objective")
    opened = set(record.get("opened_facilities") or [])
    mask = [int(m in opened) for m in meta.M] if objective is not None else None
    status = ("OPTIMAL" if record.get("proven_optimal") else "SUBOPTIMAL"
              if objective is not None else
              "TIME_LIMIT_NO_SOLUTION" if record.get("status") == "time_limit"
              else str(record.get("status", "unknown")).upper())
    return mask, objective, status, float(record.get("solver_time_sec") or 0.0)


def _solve_instance_conpas(
    gnn_model,
    container,
    meta,
    problem="cart",
    min_k=0,
    max_k=0,
    time_limit=PAPER_TIME_LIMIT,
    solver_threads=PAPER_SOLVER_THREADS,
    mip_gap=None,
    enforce_problem_constraints=True,
    ca_k_max=None,
    qkp_k_min=None,
    qkp_k_max=None,
    conpas_k0=None,
    conpas_k1=None,
    conpas_delta=None,
    benchmark_name="",
    fixed_assignments=None,
):
    """
    Rank-based ConPaS-style (shared encoder) Predict-and-Search inference.

    Procedure (Huang et al., ICML 2024)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    1. Run GNN once (outside the solver) -> per-variable logits and probabilities.
    2. Rank variables by probability, breaking ties by variable index.
    3. X0 = k0 lowest-probability variables, X1 = k1 highest-probability variables.
    4. Add one neighbourhood constraint:
           sum_{j in X0} x_j + sum_{j in X1} (1 - x_j) <= delta
       Variables outside X0 union X1 are completely unrestricted.
    5. Solve the restricted original MIP with the unchanged original objective.

    Returns
    ~~~~~~~
    (mask, milp_energy, milp_energy, status, extra_info)
    """
    if not enforce_problem_constraints:
        raise NotImplementedError(
            "ConPaS inference requires the original hard constraints because it solves "
            "the rank-neighbourhood-restricted original MIP."
        )

    problem = _normalize_problem_name(problem)

    t_gnn_start = time.time()
    logits = _variable_scores_np(gnn_model, container)
    t_gnn = time.time() - t_gnn_start

    probs = _sigmoid_np(logits)
    n_vars = int(probs.size)
    k0, k1, delta = _resolve_conpas_rank_config(
        problem,
        n_vars,
        benchmark_name=benchmark_name,
        k0=conpas_k0,
        k1=conpas_k1,
        delta=conpas_delta,
    )
    # User/scenario-fixed variables are already hard constrained, so do not
    # include them in the ConPaS rank neighbourhood.
    user_fixed = _normalize_fixed_assignments(fixed_assignments, n_vars)
    n_user_fixed_zero = sum(1 for v in user_fixed.values() if int(v) == 0)
    n_user_fixed_one = sum(1 for v in user_fixed.values() if int(v) == 1)

    x0_indices, x1_indices = _conpas_rank_sets(
        probs,
        k0,
        k1,
        excluded_indices=user_fixed.keys(),
    )
    rank_target = [None for _ in range(n_vars)]
    for idx in x0_indices:
        rank_target[int(idx)] = 0
    for idx in x1_indices:
        rank_target[int(idx)] = 1

    t_mip_start = time.time()
    if problem == "ca":
        mask, obj, status, solver_rt = _solve_conpas_ca_milp(
            meta, x0_indices, x1_indices, delta,
            time_limit=time_limit, k_max=ca_k_max, fixed_assignments=user_fixed,
            solver_threads=solver_threads, mip_gap=mip_gap,
        )
    elif problem == "qkp":
        mask, obj, status, solver_rt = _solve_conpas_qkp_milp(
            meta, x0_indices, x1_indices, delta,
            time_limit=time_limit, k_min=qkp_k_min, k_max=qkp_k_max,
            solver_threads=solver_threads, mip_gap=mip_gap,
        )
    else:  # cart
        mask, obj, status, solver_rt = _solve_conpas_cart_milp(
            meta, x0_indices, x1_indices, delta,
            min_k=min_k, max_k=max_k, time_limit=time_limit,
            fixed_assignments=user_fixed,
            solver_threads=solver_threads, mip_gap=mip_gap,
        )
    t_mip = time.time() - t_mip_start
    t_total = t_gnn + t_mip

    extra = {
        "variable_logits":          [float(v) for v in logits.tolist()],
        "variable_probabilities":   [float(v) for v in probs.tolist()],
        "predicted_y":              rank_target,
        "conpas_rank_target":       rank_target,
        "conpas_k0":                int(k0),
        "conpas_k1":                int(k1),
        "conpas_delta":             int(delta),
        "conpas_X0":                [int(j) for j in x0_indices],
        "conpas_X1":                [int(j) for j in x1_indices],
        "conpas_X0_probabilities":  [float(probs[int(j)]) for j in x0_indices],
        "conpas_X1_probabilities":  [float(probs[int(j)]) for j in x1_indices],
        "conpas_excluded_fixed_indices": sorted(int(j) for j in user_fixed.keys()),
        "selected_zero_variables":  len(x0_indices),
        "selected_one_variables":   len(x1_indices),
        "allowed_disagreements":    int(delta),
        "num_hard_fixed_zero":      0,
        "num_hard_fixed_one":       0,
        "num_user_fixed_zero":      n_user_fixed_zero,
        "num_user_fixed_one":       n_user_fixed_one,
        "num_fixed_variables":      n_user_fixed_zero + n_user_fixed_one,
        "effective_fixed_variables": len(x0_indices) + len(x1_indices) if int(delta) == 0 else 0,
        "num_rank_neighbourhood_vars": len(x0_indices) + len(x1_indices),
        "fraction_restricted":      float(len(x0_indices) + len(x1_indices)) / max(1, n_vars),
        "restriction_type":         "rank_neighbourhood",
        "neighbourhood_redundant":  bool(int(delta) > int(k0) + int(k1)),
        "gnn_forward_time_sec":     float(t_gnn),
        "mip_solve_time_sec":       float(t_mip),
        "total_inference_time_sec": float(t_total),
        "solver_runtime_sec":       float(solver_rt) if solver_rt is not None else 0.0,
    }

    if mask is not None and obj is not None:
        energy_value = _objective_to_energy(meta, obj)
        return mask, energy_value, energy_value, status, extra
    return None, None, None, status, extra


# --- THE SIMPLE SOLVER ---
def solve_instance_simple(
    gnn_model,
    container,
    meta,
    problem="cart",
    min_k=0,
    max_k=0,
    time_limit=PAPER_TIME_LIMIT,
    solver_threads=PAPER_SOLVER_THREADS,
    mip_gap=None,
    enforce_problem_constraints=True,
    ca_k_max=None,
    qkp_k_min=None,
    qkp_k_max=None,
    fixing_ratio=None,
    fixing_threshold_high=None,
    fixing_threshold_low=None,
    fixed_assignments=None,
    cart_symmetry_breaking=True,
    n_dives=None,
    parallel_dives=False,
    dive_workers=None,
    # ConPaS-specific inference parameters
    conpas_k0=None,
    conpas_k1=None,
    conpas_delta=None,
    conpas_benchmark_name="",
):
    problem = _normalize_problem_name(problem)
    if _is_conpas_model(gnn_model):
        return _solve_instance_conpas(
            gnn_model,
            container,
            meta,
            problem=problem,
            min_k=min_k,
            max_k=max_k,
            time_limit=time_limit,
            solver_threads=solver_threads,
            mip_gap=mip_gap,
            enforce_problem_constraints=enforce_problem_constraints,
            ca_k_max=ca_k_max,
            qkp_k_min=qkp_k_min,
            qkp_k_max=qkp_k_max,
            conpas_k0=conpas_k0,
            conpas_k1=conpas_k1,
            conpas_delta=conpas_delta,
            benchmark_name=conpas_benchmark_name,
            fixed_assignments=fixed_assignments,
        )
    if _is_fixing_model(gnn_model):
        # predict_and_search uses deterministic K=1; gnn_fixing_mip uses K=n_dives
        effective_dives = (
            1
            if str(getattr(gnn_model, "model_type", "")) == "predict_and_search"
            else n_dives
        )
        return _solve_instance_gnn_fixing(
            gnn_model,
            container,
            meta,
            problem=problem,
            min_k=min_k,
            max_k=max_k,
            time_limit=time_limit,
            enforce_problem_constraints=enforce_problem_constraints,
            ca_k_max=ca_k_max,
            qkp_k_min=qkp_k_min,
            qkp_k_max=qkp_k_max,
            solver_threads=solver_threads,
            mip_gap=mip_gap,
            fixing_ratio=fixing_ratio,
            fixing_threshold_high=fixing_threshold_high,
            fixing_threshold_low=fixing_threshold_low,
            fixed_assignments=fixed_assignments,
            n_dives=effective_dives,
            parallel_dives=parallel_dives,
            dive_workers=dive_workers,
        )
    if _is_linear_score_model(gnn_model):
        mask, milp_e, true_e, status = _solve_instance_linear_score(
            gnn_model,
            container,
            meta,
            problem=problem,
            min_k=min_k,
            max_k=max_k,
            time_limit=time_limit,
            enforce_problem_constraints=enforce_problem_constraints,
            ca_k_max=ca_k_max,
            qkp_k_min=qkp_k_min,
            qkp_k_max=qkp_k_max,
            fixed_assignments=fixed_assignments,
            cart_symmetry_breaking=cart_symmetry_breaking,
            solver_threads=solver_threads,
            mip_gap=mip_gap,
        )
        return mask, milp_e, true_e, status, {}

    if problem == "ca":
        variant = resolve_ca_readout_variant(problem, getattr(gnn_model, "ca_readout_variant", "auto"))
        if variant == "pooled_only":
            mask, milp_e, true_e, status, extra = _solve_instance_ca_pooled_or_direct_y(
                gnn_model,
                container,
                meta,
                time_limit=time_limit,
                enforce_problem_constraints=enforce_problem_constraints,
                ca_k_max=ca_k_max,
                fixed_assignments=fixed_assignments,
                solver_threads=solver_threads,
                mip_gap=mip_gap,
                return_details=True,
                include_direct_y=False,
            )
            return mask, milp_e, true_e, status, extra
        if variant == "direct_y":
            mask, milp_e, true_e, status, extra = _solve_instance_ca_pooled_or_direct_y(
                gnn_model,
                container,
                meta,
                time_limit=time_limit,
                enforce_problem_constraints=enforce_problem_constraints,
                ca_k_max=ca_k_max,
                fixed_assignments=fixed_assignments,
                solver_threads=solver_threads,
                mip_gap=mip_gap,
                return_details=True,
                include_direct_y=True,
            )
            return mask, milp_e, true_e, status, extra
        if variant == "sum_only_plus":
            mask, milp_e, true_e, status, extra = _solve_instance_ca_sum_only_plus(
                gnn_model,
                container,
                meta,
                time_limit=time_limit,
                enforce_problem_constraints=enforce_problem_constraints,
                ca_k_max=ca_k_max,
                fixed_assignments=fixed_assignments,
                solver_threads=solver_threads,
                mip_gap=mip_gap,
                return_details=True,
            )
            return mask, milp_e, true_e, status, extra
        raise AssertionError(f"Unhandled CA readout variant: {variant}")
    if problem == "qkp":
        mask, milp_e, true_e, status = _solve_instance_qkp(
            gnn_model,
            container,
            meta,
            time_limit=time_limit,
            enforce_problem_constraints=enforce_problem_constraints,
            qkp_k_min=qkp_k_min,
            qkp_k_max=qkp_k_max,
            solver_threads=solver_threads,
            mip_gap=mip_gap,
        )
        return mask, milp_e, true_e, status, {}
    encoder_start = time.perf_counter()
    with torch.no_grad():
        enc_out = gnn_model.encoder(container)
        hc = enc_out["hc"].cpu().numpy()  # [nC, dh]
        hm = enc_out["hm"].cpu().numpy()  # [M, dh]
        hg = enc_out["hg"].cpu().numpy()  # [2*dh]
        dh = hm.shape[1]
        num_fac = len(meta.M)
        nC = int(container["nC"])
        ei = container["edge_index"].cpu().numpy()
        src_nodes = ei[0].astype(int)
        fac_nodes = (ei[1].astype(int) - nC)
    encoder_time_sec = time.perf_counter() - encoder_start

    raw_capacities = (
        np.array([meta.CAPMAX[fid] for fid in meta.M], dtype=float)
        if problem == "cart"
        else np.zeros(num_fac, dtype=float)
    )
    peak_active_demand = float(meta.globals.get("peak_active_demand", 0.0))

    neighbours = [[] for _ in range(nC)]
    for src, fac in zip(src_nodes.tolist(), fac_nodes.tolist()):
        if 0 <= src < nC and 0 <= fac < num_fac:
            neighbours[src].append(fac)

    # GUROBI MODEL
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    m = gp.Model("Simple_GNN_CFLP", env=env)
    m.setParam("TimeLimit", time_limit)
    if solver_threads is not None:
        m.setParam("Threads", int(solver_threads))
    if mip_gap is not None:
        m.setParam("MIPGap", float(mip_gap))
    
    # 1. Facility/Set Selection Variables
    z = [m.addVar(vtype=GRB.BINARY, name=f"z_{j}") for j in range(num_fac)]
    _apply_binary_fixes(m, z, fixed_assignments, "fixed_z")
    
    # --- FEATURE 1: SUM POOLING (Exact Linear) ---
    pool_sum_vars = [m.addVar(lb=-GRB.INFINITY, name=f"sum_{d}") for d in range(dh)]

    for d in range(dh):
        sum_expr = gp.LinExpr()
        for j in range(num_fac):
            sum_expr += z[j] * float(hm[j, d])
        m.addConstr(pool_sum_vars[d] == sum_expr)

    # --- FEATURE 2: COVERAGE SUMMARY (Exact Linearization) ---
    s_vars = []
    for c in range(nC):
        s_c = m.addVar(vtype=GRB.BINARY, name=f"s_{c}")
        s_vars.append(s_c)
        N_c = neighbours[c]
        if len(N_c) == 0:
            m.addConstr(s_c == 0, name=f"cov_zero_{c}")
        else:
            m.addConstr(
                s_c <= gp.quicksum(z[m_idx] for m_idx in N_c),
                name=f"cov_ub_{c}",
            )
            for m_idx in N_c:
                m.addConstr(s_c >= z[m_idx], name=f"cov_lb_{c}_{m_idx}")

    cov_pool_vars = []
    for d in range(dh):
        cpv = m.addVar(lb=-GRB.INFINITY, name=f"cov_{d}")
        cov_pool_vars.append(cpv)
        m.addConstr(
            cpv == gp.quicksum(float(hc[c, d]) * s_vars[c] for c in range(nC)),
            name=f"cov_eq_{d}",
        )

    if len(s_vars) != nC:
        raise RuntimeError(f"Expected exactly nC coverage binaries, got {len(s_vars)} vs {nC}.")

    # --- HARD CONSTRAINTS ---
    if enforce_problem_constraints:
        if problem == "cart":
            _add_cart_constraints(
                m,
                z,
                raw_capacities,
                peak_active_demand,
                min_k,
                max_k,
                fixed_assignments=fixed_assignments,
                meta=meta,
                enable_symmetry_breaking=cart_symmetry_breaking,
            )


    # --- ASSEMBLE MLP INPUT ---
    mlp_input = []
    mlp_input.extend(pool_sum_vars)   # [dh]
    mlp_input.extend(cov_pool_vars)   # [dh]
    for val in hg:                    # [2*dh] Global
        mlp_input.append(float(val))

    # --- Tight bounds for Big-M ---
    L_pool = np.zeros(dh, dtype=float)
    U_pool = np.zeros(dh, dtype=float)
    if problem != "cart":
        pool_min_k = 0
        pool_max_k = num_fac
    else:
        pool_min_k, pool_max_k = _cart_cardinality_bounds(num_fac, min_k=min_k, max_k=max_k)
    for d in range(dh):
        L_pool[d], U_pool[d] = _bounds_sum_k(hm[:, d], pool_min_k, pool_max_k)
    L_cov = np.zeros(dh, dtype=float)
    U_cov = np.zeros(dh, dtype=float)
    for d in range(dh):
        col = hc[:, d].astype(float)
        L_cov[d] = float(col[col < 0.0].sum())
        U_cov[d] = float(col[col > 0.0].sum())

    input_L = np.concatenate([L_pool, L_cov, hg.astype(float)])
    input_U = np.concatenate([U_pool, U_cov, hg.astype(float)])

    expected_input_dim = int(getattr(getattr(gnn_model, "readout", None), "out_dim", 4 * dh))
    if len(mlp_input) != expected_input_dim:
        raise RuntimeError(
            f"MLP input length mismatch: expected {expected_input_dim}, got {len(mlp_input)}."
        )
    if input_L.shape[0] != len(mlp_input) or input_U.shape[0] != len(mlp_input):
        raise RuntimeError(
            "IBP input bounds length mismatch with assembled MLP input "
            f"({input_L.shape[0]}, {input_U.shape[0]}) vs {len(mlp_input)}."
        )

    energy_var = _add_head_objective(m, gnn_model, mlp_input, input_L, input_U)
    mask, milp_e, true_e, status = _solve_and_score(m, z, gnn_model, container, energy_var)
    try:
        solver_runtime_sec = float(m.Runtime)
    except Exception:
        solver_runtime_sec = None
    return mask, milp_e, true_e, status, {
        "encoder_time_sec": float(encoder_time_sec),
        "solver_runtime_sec": solver_runtime_sec,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--problem", type=str, default="cart", choices=["cart", "ca", "qkp"])
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Dataset directory. Evaluate all its instances unless a split is explicitly requested.")
    parser.add_argument("--test_dir", type=str, default=None,
                        help="Optional fixed test directory. If set, use this exactly like training does.")
    parser.add_argument("--static-config", dest="static_config", type=str, default=None,
                        help="Optional path to cart static_config.json. CART only; defaults to <data_dir>/static_config.json.")
    parser.add_argument("--dat-path", dest="dat_path", type=str, default="cart_dataset/Data_CART_og.dat",
                        help="Fallback CART .dat path used to rebuild static metadata when static_config.json is missing.")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum evaluation instances to run. Default 0 runs all available instances.",
    )
    parser.add_argument("--instance_id", type=int, default=None,
                        help="Run only this instance_id (overrides split/limit).")
    parser.add_argument("--time_limit", type=int, default=PAPER_TIME_LIMIT,
                        help="Per-instance solver budget in seconds. Paper default: 3600.")
    parser.add_argument("--solver_threads", type=int, default=PAPER_SOLVER_THREADS,
                        help="Gurobi threads per MIP solve. Paper default: 8; use 1 with --parallel_dives unless the node has enough cores.")
    parser.add_argument("--mip_gap", type=float, default=None,
                        help="Optional Gurobi MIPGap for exact/benchmark MIP solves.")
    parser.add_argument("--min_k", type=int, default=0, help="CART only. Default 0 leaves open-facility count unconstrained.")
    parser.add_argument("--max_k", type=int, default=0, help="CART only. <=0 means no effective upper cap.")
    parser.add_argument("--cart_exact_reference", action="store_true",
                        help="CART only: recompute the reference optimum and score predicted masks.")
    parser.add_argument("--cart_reference_formulation", choices=["detailed"],
                        default="detailed", help="CART reference and mask-scoring formulation for detailed datasets.")
    parser.add_argument("--cart_force_off", type=int, nargs="*", default=None,
                        help="CART only: facility indices to force inactive during inference (open_j = 0).")
    parser.add_argument("--cart_force_on", type=int, nargs="*", default=None,
                        help="CART only: facility indices to force active during inference (open_j = 1).")
    parser.add_argument("--cart_no_symmetry_breaking", action="store_true",
                        help="CART only: disable equal-capacity ordering constraints during neural MILP inference.")
    parser.add_argument("--dh", type=int, default=128)
    parser.add_argument("--nlayers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--model_type", type=str, default="energnn",
                        choices=["energnn", "linear_mip_bce", "linear_mip_contrastive",
                                 "gnn_fixing_mip", "predict_and_search",
                                 "conpas_shared_encoder", "exact_solver"],
                        help="Model family. 'linear_mip_contrastive' is Linear+MIP(contrastive); "
                             "'gnn_fixing_mip' is Neural Diving with benchmark-specific dive counts; "
                             "'predict_and_search' is the original deterministic fixing baseline (linear head, K=1); "
                             "'linear_mip_bce' is Linear+MIP(BCE); "
                             "'conpas_shared_encoder' uses the rank-based ConPaS neighbourhood; "
                             "'exact_solver' runs the original Gurobi benchmark objective.")
    parser.add_argument("--ca_readout_variant", type=str, default="sum_only_plus",
                        choices=[
                            "auto",
                            "pooled_only",
                            "sum_only_plus",
                            "direct_y",
                        ],
                        help="CA only. 'pooled_only' removes scalar z_lin; "
                             "'sum_only_plus' uses the paper's structural summaries; "
                             "'direct_y' appends a fixed-width raw mask for ablation.")
    parser.add_argument("--ca_direct_y_dim", type=int, default=None,
                        help="CA direct_y only: fixed raw-mask width. Defaults to the largest loaded CA instance.")
    parser.add_argument("--qkp_readout_variant", type=str, default="auto",
                        choices=[
                            "auto",
                            "sum_only",
                            "size_aware",
                            "direct_y",
                        ],
                        help="QKP only. 'sum_only' removes structural summaries; "
                             "'size_aware' uses selected/unselected pools plus "
                             "fixed-denominator summaries; 'direct_y' appends a fixed-width raw mask.")
    parser.add_argument("--qkp_direct_y_dim", type=int, default=None,
                        help="QKP direct_y only: fixed raw-mask width. Defaults to the largest loaded QKP instance.")
    parser.add_argument("--ca_head_variant", type=str, default="mlp32", choices=["auto", "mlp16", "mlp16_deep", "mlp32", "mlp32_deep", "mlp64", "mlp64_shallow", "linear"],
                        help="Energy head variant. Default is 'mlp32' for CA and can also be set explicitly for CART.")
    parser.add_argument("--val_frac", type=float, default=None,
                        help="Opt into random splitting with this validation fraction (otherwise 0.1 when splitting).")
    parser.add_argument("--test_frac", type=float, default=None,
                        help="Opt into random splitting with this test fraction (otherwise 0.1 when splitting).")
    parser.add_argument("--split_seed", type=int, default=123)
    parser.add_argument("--prediction_log_jsonl", type=str, default=None,
                        help="Optional JSONL path for per-instance scores and solver-selected masks.")
    parser.add_argument("--no_problem_constraints", action="store_true",
                        help="Disable hard feasibility constraints for debugging.")
    parser.add_argument("--qkp_k_min", type=int, default=None,
                        help="QKP only: lower cardinality bound for zero-shot constrained inference.")
    parser.add_argument("--qkp_k_max", type=int, default=None,
                        help="QKP only: upper cardinality bound for zero-shot constrained inference.")
    parser.add_argument("--ca_k_max", type=int, default=None,
                        help="CA only: upper bound on the number of accepted bids (sum_b y_b <= K).")
    parser.add_argument("--fixing_ratio", type=float, default=None,
                        help="Predict+Search/Neural Diving fixing fraction. Paper defaults depend on method and benchmark.")
    parser.add_argument("--fixing_threshold_high", type=float, default=None,
                        help="CART fixing threshold for one. Paper default: 0.6.")
    parser.add_argument("--fixing_threshold_low", type=float, default=None,
                        help="CART fixing threshold for zero. Paper default: 0.4.")
    # ConPaS inference arguments
    parser.add_argument("--k0", type=int, default=None,
                        help="ConPaS: number of lowest-probability variables in X0.")
    parser.add_argument("--k1", type=int, default=None,
                        help="ConPaS: number of highest-probability variables in X1.")
    parser.add_argument("--delta", type=int, default=None,
                        help="ConPaS: maximum disagreements allowed on X0 union X1.")
    parser.add_argument("--conpas_verbose", action="store_true",
                        help="ConPaS: print X0/X1 indices and probabilities for each instance.")
    parser.add_argument("--n_dives", type=int, default=None,
                        help="Neural Diving stochastic dives. Paper defaults: CART 8, CA 100, QKP 20. "
                             "Predict+Search always uses one deterministic dive; time_limit is split across dives.")
    parser.add_argument("--parallel_dives", action="store_true",
                        help="Neural Diving: solve dives for one instance concurrently. "
                             "The outer instance loop remains sequential.")
    parser.add_argument("--dive_workers", type=int, default=None,
                        help="Neural Diving: number of concurrent dive solves when --parallel_dives is set. "
                             "Defaults to --n_dives.")
    args = parser.parse_args()
    args.problem = _normalize_problem_name(args.problem)
    args.model_type = _resolve_model_type(args.model_type)
    model_name = model_display_name(args.model_type)
    if args.model_type != "exact_solver" and not args.ckpt:
        raise ValueError("--ckpt is required unless --model_type exact_solver.")
    if args.solver_threads is not None and int(args.solver_threads) <= 0:
        raise ValueError(f"--solver_threads must be positive, got {args.solver_threads}.")
    if args.dive_workers is not None and int(args.dive_workers) <= 0:
        raise ValueError(f"--dive_workers must be positive, got {args.dive_workers}.")
    if args.parallel_dives and args.solver_threads is None:
        warnings.warn(
            "--parallel_dives runs several Gurobi solves concurrently. "
            "Pass --solver_threads 1 unless the node has enough cores for "
            "--dive_workers times Gurobi's default thread count.",
            RuntimeWarning,
        )
    if args.problem != "qkp" and (args.qkp_k_min is not None or args.qkp_k_max is not None):
        raise ValueError("--qkp_k_min/--qkp_k_max are only supported for --problem qkp.")
    if args.problem != "ca" and args.ca_k_max is not None:
        raise ValueError("--ca_k_max is only supported for --problem ca.")
    if args.problem != "cart" and (args.cart_force_off or args.cart_force_on):
        raise ValueError("--cart_force_off/--cart_force_on are only supported for --problem cart.")
    args.ca_readout_variant = resolve_ca_readout_variant(args.problem, args.ca_readout_variant)
    args.qkp_readout_variant = resolve_qkp_readout_variant(args.problem, args.qkp_readout_variant)
    args.ca_head_variant = resolve_ca_head_variant(args.problem, args.ca_head_variant)
    print(f"[model] type={args.model_type} name={model_name}")
    if args.problem == "ca":
        print(f"[ca_readout] variant={args.ca_readout_variant}")
        if args.ca_k_max is not None:
            print(f"[ca_cardinality] k_max={args.ca_k_max}")
    if args.problem == "qkp":
        print(f"[qkp_readout] variant={args.qkp_readout_variant}")
        if args.qkp_k_min is not None or args.qkp_k_max is not None:
            print(f"[qkp_cardinality] k_min={args.qkp_k_min} k_max={args.qkp_k_max}")
    if args.problem == "cart" and args.cart_force_off:
        print(f"[cart_force_off] indices={list(args.cart_force_off)}")
    if args.problem == "cart" and args.cart_force_on:
        print(f"[cart_force_on] indices={list(args.cart_force_on)}")
    if args.problem == "cart" and args.cart_no_symmetry_breaking:
        print("[cart] equal-capacity symmetry breaking disabled")
    print(f"[head] variant={args.ca_head_variant}")
    
    device = torch.device("cpu")
    print("Loading Data...")
    if args.problem == "ca":
        data_dir = args.data_dir or "data/CA/175/test"
    elif args.problem == "qkp":
        data_dir = args.data_dir or "data/qkp"
    else:
        data_dir = args.data_dir or "runs/CART16_test_detailed_labels"

    data_path = Path(data_dir)
    if (
        not (data_path / "instances_meta.jsonl").is_file()
        and (data_path / "train").is_dir()
        and (data_path / "val").is_dir()
        and (data_path / "test").is_dir()
    ):
        train_candidate = data_path / "train"
        test_candidate = data_path / "test"
        print(
            f"[data] resolved split root from {data_path.as_posix()} -> "
            f"model_data={train_candidate.as_posix()} "
            f"test={test_candidate.as_posix()}"
        )
        data_dir = train_candidate.as_posix()
        if args.test_dir is None:
            args.test_dir = test_candidate.as_posix()

    base_metas, base_pools, _, base_tmfe = _load_eval_data(
        args.problem, data_dir, args.static_config, args.dat_path
    )

    model = None
    if args.model_type != "exact_solver":
        print("Loading Model...")
        model_probe_pools = {k: [] for k in base_metas.keys()}
        model_probe_ds = InstancePoolDataset(base_metas, model_probe_pools, TMFE=base_tmfe)
        g0, _, _, _, _ = model_probe_ds[0]
        dc, dm, de = g0["Xc"].shape[1], g0["Xm"].shape[1], g0["edge_attr"].shape[1]
        effective_ca_direct_y_dim = None
        if args.problem == "ca" and args.ca_readout_variant == "direct_y":
            widths = [len(meta.M) for meta in base_metas.values()]
            max_width = max(widths) if widths else int(g0["Xm"].shape[0])
            min_width = min(widths) if widths else max_width
            effective_ca_direct_y_dim = (
                max_width if args.ca_direct_y_dim is None else int(args.ca_direct_y_dim)
            )
            if effective_ca_direct_y_dim < max_width:
                raise ValueError(
                    "--ca_direct_y_dim is smaller than the largest loaded CA "
                    f"instance ({effective_ca_direct_y_dim} < {max_width})."
                )
            print(
                f"[ca_direct_y] dim={effective_ca_direct_y_dim} "
                f"model_instance_width_range=[{min_width}, {max_width}]"
            )
        effective_qkp_direct_y_dim = None
        if args.problem == "qkp" and args.qkp_readout_variant == "direct_y":
            widths = [len(meta.M) for meta in base_metas.values()]
            max_width = max(widths) if widths else int(g0["Xm"].shape[0])
            min_width = min(widths) if widths else max_width
            effective_qkp_direct_y_dim = (
                max_width if args.qkp_direct_y_dim is None else int(args.qkp_direct_y_dim)
            )
            if effective_qkp_direct_y_dim < max_width:
                raise ValueError(
                    "--qkp_direct_y_dim is smaller than the largest loaded QKP "
                    f"instance ({effective_qkp_direct_y_dim} < {max_width})."
                )
            print(
                f"[qkp_direct_y] dim={effective_qkp_direct_y_dim} "
                f"model_instance_width_range=[{min_width}, {max_width}]"
            )

        model_cfg_dict = {
            "dc": dc, "dm": dm, "de": de,
            "dh": int(args.dh), "nlayers": int(args.nlayers), "dropout": float(args.dropout),
            "problem": args.problem,
        }
        if args.model_type == "energnn":
            model_cfg_dict.update(
                {
                    "ca_readout_variant": args.ca_readout_variant,
                    "ca_direct_y_dim": effective_ca_direct_y_dim,
                    "qkp_readout_variant": args.qkp_readout_variant,
                    "qkp_direct_y_dim": effective_qkp_direct_y_dim,
                    "ca_head_variant": args.ca_head_variant,
                }
            )
        model = load_trained_model(
            args.model_type,
            model_cfg_dict,
            args.ckpt,
            device,
        )
    
    if args.instance_id is not None:
        if args.test_dir is not None:
            metas, pools, _, tmfe = _load_eval_data(
                args.problem, args.test_dir, args.static_config, args.dat_path
            )
        else:
            metas = base_metas
            pools = base_pools
            tmfe = base_tmfe
        if args.instance_id not in metas:
            raise ValueError(f"instance_id {args.instance_id} not found in dataset.")
        ids = [args.instance_id]
        print(f"[split] using explicit instance_id={args.instance_id}")
    elif args.test_dir:
        metas, pools, _, tmfe = _load_eval_data(
            args.problem, args.test_dir, args.static_config, args.dat_path
        )
        all_test_ids = sorted(list({*metas.keys()} & {*pools.keys()}))
        ids = all_test_ids if args.limit <= 0 else all_test_ids[:args.limit]
        print(f"[split] using fixed test dir: {args.test_dir} ({len(all_test_ids)} instances)")
    elif args.val_frac is None and args.test_frac is None:
        metas = base_metas
        pools = base_pools
        tmfe = base_tmfe
        all_ids = sorted(list({*metas.keys()} & {*pools.keys()}))
        ids = all_ids if args.limit <= 0 else all_ids[:args.limit]
        print(f"[split] using full data dir: {data_dir} ({len(all_ids)} instances)")
    else:
        metas = base_metas
        pools = base_pools
        tmfe = base_tmfe
        all_ids = sorted(list({*metas.keys()} & {*pools.keys()}))
        tr_ids, va_ids, te_ids = split_instance_ids(
            all_ids,
            val_frac=0.1 if args.val_frac is None else args.val_frac,
            test_frac=0.1 if args.test_frac is None else args.test_frac,
            seed=args.split_seed,
        )
        print(f"[split] train={len(tr_ids)} val={len(va_ids)} test={len(te_ids)}")
        test_ids = te_ids if te_ids else va_ids
        if not test_ids:
            raise ValueError(
                "Split produced no test IDs. "
                "Pass --instance_id, increase --test_frac, or use --test_dir."
            )
        ids = test_ids if args.limit <= 0 else test_ids[:args.limit]

    dummy_pools = {k: [] for k in metas.keys()}
    ds = InstancePoolDataset(metas, dummy_pools, TMFE=tmfe)

    conpas_benchmark_name = " ".join(
        str(part)
        for part in [data_dir, args.test_dir, args.ckpt]
        if part is not None
    )
    conpas_rank_config = None
    if args.model_type == "conpas_shared_encoder":
        first_meta = metas[int(ids[0])]
        conpas_rank_config = _resolve_conpas_rank_config(
            args.problem,
            len(first_meta.M),
            benchmark_name=conpas_benchmark_name,
            k0=args.k0,
            k1=args.k1,
            delta=args.delta,
        )
        print(
            "[conpas] rank_neighbourhood "
            f"k0={conpas_rank_config[0]} "
            f"k1={conpas_rank_config[1]} "
            f"delta={conpas_rank_config[2]} "
            f"n_vars={len(first_meta.M)}"
        )

    prediction_log_path = Path(args.prediction_log_jsonl) if args.prediction_log_jsonl else None
    if prediction_log_path is not None:
        prediction_log_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_log_path.write_text("", encoding="utf-8")
        print(f"[log] writing prediction JSONL to {prediction_log_path.as_posix()}")

    summary = {
        "total": len(ids),
        "solved": 0,
        "failed": 0,
        "failed_infeasible": 0,
        "failed_other": 0,
        "failed_status_counts": {},
        "with_reference": 0,
        "no_reference": 0,
        "exact_optimal": 0,
        "misses": 0,
        "misses_feasible": 0,
        "misses_infeasible": 0,
        "misses_gap_known": 0,
        "misses_gap_unknown": 0,
        "miss_gap_values": [],
        "miss_gap_not_applicable": 0,
        "gap_all_known": 0,
        "gap_all_unknown": 0,
        "gap_all_not_applicable": 0,
        "reference_infeasible": 0,
        "reference_infeasible_ids": [],
        "runtime_values": [],
        "solver_runtime_values": [],
        "gap_values_all": [],
        "within_1pct": 0,
        "feasible_predictions": 0,
        "evaluated_predictions": 0,
        "fixed_variable_counts": [],
        "fixing_fallback_total": 0,
        "fixing_fallback_used": 0,
        "failed_ids": [],
        "failed_reference_feasible": 0,
        "failed_reference_infeasible": 0,
        "failed_reference_other": 0,
    }
    
    print(f"\n{'Inst':<6} | {'MILP Est':<10} | {'True Energy':<11} | {'Diff':<10} | {'Mask'}")
    print("-" * 75)
    
    for inst_id in ids:
        meta = metas[inst_id]
        container = None
        linear_scores = None
        extra_info = {}
        cart_fixed_assignments = (
            _cart_forced_assignments(args.cart_force_off, args.cart_force_on, len(meta.M))
            if args.problem == "cart"
            else None
        )
        samples = pools.get(inst_id, [])
        ref_mask_filter = None
        if args.problem == "qkp" and (args.qkp_k_min is not None or args.qkp_k_max is not None):
            ref_mask_filter = lambda mask_key: _eval_qkp_mask(
                meta,
                mask_key,
                k_min=args.qkp_k_min,
                k_max=args.qkp_k_max,
            )["feasible"]
        ref_stats = _collect_sample_reference(meta, samples, mask_filter=ref_mask_filter)
        if args.problem == "cart" and args.cart_reference_formulation == "detailed":
            ref_stats["objective_model"] = "cart_detailed"
        exact_reference_required = _requires_exact_reference(args, sample_reference=ref_stats)
        detailed_reference_source = None
        detailed_reference_status = None
        detailed_reference_runtime = None
        detailed_reference_record = None
        detailed_constrained_reference = (
            args.problem == "cart"
            and not exact_reference_required
            and ref_stats.get("objective_model") == "cart_detailed"
            and bool(cart_fixed_assignments or args.min_k > 0 or args.max_k > 0
                     or args.cart_exact_reference)
        )
        if detailed_constrained_reference:
            # Stored objectives belong to the original scenario, not the injected one.
            ref_stats = _collect_sample_reference(meta, [])
            ref_stats["objective_model"] = "cart_detailed"
            detailed_reference_source = f"{args.cart_reference_formulation}_cart_milp_constrained"
        if (
            args.problem == "cart"
            and not exact_reference_required
            and ref_stats.get("objective_model") == "cart_detailed"
            and ref_stats.get("best_obj") is None
        ):
            try:
                print(
                    f"[cart_reference] instance={inst_id}: building and solving the detailed "
                    f"CART model for a new reference (time_limit={args.time_limit}s).",
                    flush=True,
                )
                detailed_reference_record = _solve_cart_detailed_milp(
                    meta,
                    mask=None,
                    time_limit=args.time_limit,
                    solver_threads=args.solver_threads,
                    mip_gap=args.mip_gap,
                    max_open_facilities=args.max_k if args.max_k > 0 else None,
                    min_open_facilities=args.min_k,
                    fixed_assignments=cart_fixed_assignments,
                    formulation=args.cart_reference_formulation,
                    tee=True,
                )
                detailed_reference_status = detailed_reference_record.get("status")
                detailed_reference_runtime = detailed_reference_record.get("solver_time_sec")
                detailed_ref_stats = _cart_detailed_reference_stats_from_record(
                    meta,
                    detailed_reference_record,
                    require_optimal=detailed_constrained_reference,
                )
                if detailed_ref_stats is not None:
                    ref_stats = detailed_ref_stats
                    detailed_reference_source = (
                        f"{args.cart_reference_formulation}_cart_milp_constrained" if detailed_constrained_reference
                        else f"{args.cart_reference_formulation}_cart_milp"
                    )
                    print(
                        f"  cart_reference: {args.cart_reference_formulation} model supplied "
                        f"obj={float(ref_stats['best_obj']):.4f} "
                        f"status={detailed_reference_status}"
                    )
                else:
                    print(
                        "  cart_reference: no usable reference optimum "
                        f"(status={detailed_reference_status}); gap unavailable"
                    )
            except Exception as exc:
                detailed_reference_status = f"failed: {exc}"
                print(f"  cart_reference_fallback: detailed model failed ({exc})")
        if (
            args.problem == "cart"
            and ref_stats.get("objective_model") == "cart_detailed"
            and inst_id == ids[0]
        ):
            print(
                f"[cart_reference] using {args.cart_reference_formulation} reference and mask scoring"
                if detailed_constrained_reference else
                "[cart_reference] using detailed dataset facility masks and objectives; "
                "unknown predicted masks are scored with the detailed model"
            )
        if args.model_type != "exact_solver":
            container, _ = ds._build_graph(inst_id)
        if args.model_type in {"linear_mip_bce", "linear_mip_contrastive"}:
            linear_scores = _variable_scores_np(model, container)

        exact_ref_mask = None
        exact_ref_obj = None
        exact_ref_status = None
        exact_ref_runtime = None
        solver_runtime_sec = None
        t0 = time.perf_counter()
        if args.model_type == "exact_solver":
            mask, exact_ref_obj, status, solver_runtime = _solve_exact_reference_milp(
                args,
                meta,
                cart_fixed_assignments=cart_fixed_assignments,
            )
            exact_ref_mask = mask
            exact_ref_status = status
            exact_ref_runtime = float(solver_runtime)
            if exact_ref_obj is not None:
                milp_e = _objective_to_energy(meta, exact_ref_obj)
                true_e = milp_e
            else:
                milp_e = None
                true_e = None
            solver_runtime_sec = float(solver_runtime)
        else:
            mask, milp_e, true_e, status, extra_info = solve_instance_simple(
                model, container, meta, problem=args.problem,
                min_k=args.min_k, max_k=args.max_k, time_limit=args.time_limit,
                solver_threads=args.solver_threads,
                mip_gap=args.mip_gap,
                enforce_problem_constraints=not args.no_problem_constraints,
                ca_k_max=args.ca_k_max,
                qkp_k_min=args.qkp_k_min,
                qkp_k_max=args.qkp_k_max,
                fixing_ratio=args.fixing_ratio,
                fixing_threshold_high=args.fixing_threshold_high,
                fixing_threshold_low=args.fixing_threshold_low,
                fixed_assignments=cart_fixed_assignments,
                cart_symmetry_breaking=not args.cart_no_symmetry_breaking,
                n_dives=args.n_dives,
                parallel_dives=args.parallel_dives,
                dive_workers=args.dive_workers,
                conpas_k0=None if conpas_rank_config is None else conpas_rank_config[0],
                conpas_k1=None if conpas_rank_config is None else conpas_rank_config[1],
                conpas_delta=None if conpas_rank_config is None else conpas_rank_config[2],
                conpas_benchmark_name=conpas_benchmark_name,
            )
            if isinstance(extra_info, dict) and extra_info.get("solver_runtime_sec") is not None:
                solver_runtime_sec = float(extra_info.get("solver_runtime_sec"))
            if exact_reference_required:
                (
                    exact_ref_mask,
                    exact_ref_obj,
                    exact_ref_status,
                    exact_ref_runtime,
                ) = _solve_exact_reference_milp(
                    args,
                    meta,
                    cart_fixed_assignments=cart_fixed_assignments,
                )
        runtime_sec = float(time.perf_counter() - t0)
        
        if mask is not None:
            summary["solved"] += 1
            summary["runtime_values"].append(float(runtime_sec))
            if solver_runtime_sec is not None:
                summary["solver_runtime_values"].append(float(solver_runtime_sec))
            diff = abs(milp_e - true_e)
            mask_str = str(mask)
            print(f"{inst_id:<6} | {milp_e:<10.4f} | {true_e:<11.4f} | {diff:<10.4f} | {mask_str}")
            if linear_scores is not None:
                score_str = [round(float(v), 6) for v in linear_scores.tolist()]
                print(f"  linear_scores: {score_str}")
                print(f"  selected_y: {mask}")
            if args.model_type == "gnn_fixing_mip":
                print(
                    f"  fixing: fixed={int(extra_info.get('num_fixed_variables', 0))} "
                    f"fallback_count={int(extra_info.get('fallback_count', 0))} "
                    f"parallel_dives={bool(extra_info.get('parallel_dives', False))} "
                    f"dive_workers={int(extra_info.get('dive_workers') or 1)} "
                    f"dive_wall={float(extra_info.get('dive_wall_time_sec') or 0.0):.2f}s "
                    f"dive_solver_sum={float(extra_info.get('dive_solver_runtime_sum_sec') or 0.0):.2f}s"
                )
            if args.model_type == "conpas_shared_encoder":
                print(
                    f"  conpas: selected_zero_variables={int(extra_info.get('selected_zero_variables', 0))} "
                    f"selected_one_variables={int(extra_info.get('selected_one_variables', 0))} "
                    f"allowed_disagreements={int(extra_info.get('allowed_disagreements', 0))} "
                    f"user_fixed={int(extra_info.get('num_fixed_variables', 0))}"
                )
                if args.conpas_verbose:
                    print(
                        f"  conpas_X0={extra_info.get('conpas_X0', [])} "
                        f"p0={[round(float(v), 6) for v in extra_info.get('conpas_X0_probabilities', [])]}"
                    )
                    print(
                        f"  conpas_X1={extra_info.get('conpas_X1', [])} "
                        f"p1={[round(float(v), 6) for v in extra_info.get('conpas_X1_probabilities', [])]}"
                    )

            reference_source = (
                "samples_qkp_cardinality_filtered"
                if ref_mask_filter is not None
                else detailed_reference_source
                if detailed_reference_source is not None
                else "samples_cart_detailed"
                if ref_stats.get("objective_model") == "cart_detailed"
                else "samples"
            )
            reference_infeasible = detailed_reference_status == "infeasible"
            pred_exact_cart_status = None
            pred_exact_cart_runtime = None
            pred_detailed_cart_status = None
            pred_detailed_cart_runtime = None
            if exact_reference_required:
                reference_source = _exact_reference_source_name(args.problem)
                if exact_ref_mask is not None and exact_ref_obj is not None:
                    ref_stats = {
                        "best_obj": float(exact_ref_obj),
                        "best_mask": [int(v) for v in exact_ref_mask],
                        "optimal_masks": {_mask_key(exact_ref_mask)},
                        "feasible_obj_by_mask": {
                            _mask_key(exact_ref_mask): float(exact_ref_obj)
                        },
                    }
                else:
                    print(f"  exact_reference: failed ({exact_ref_status})")
                    if exact_ref_status == "INFEASIBLE":
                        reference_source = f"{reference_source}_infeasible"
                        reference_infeasible = True
            pred_stats = _eval_prediction_mask(
                args.problem,
                meta,
                mask,
                args.min_k,
                args.max_k,
                ref_stats=ref_stats,
                ca_k_max=args.ca_k_max,
                qkp_k_min=args.qkp_k_min,
                qkp_k_max=args.qkp_k_max,
            )
            if args.model_type == "exact_solver" and exact_ref_obj is not None:
                pred_stats = {
                    **pred_stats,
                    "feasible": True,
                    "objective_source": "gurobi_full_solve",
                    "objective": float(exact_ref_obj),
                }
            if args.problem == "cart" and exact_reference_required and args.model_type != "exact_solver":
                _, pred_exact_obj, pred_exact_cart_status, pred_exact_cart_runtime = _solve_true_cart_milp(
                    meta,
                    min_k=args.min_k,
                    max_k=args.max_k,
                    time_limit=args.time_limit,
                    fixed_mask=mask,
                    fixed_assignments=cart_fixed_assignments,
                )
                if pred_exact_obj is not None:
                    pred_stats = {
                        **pred_stats,
                        "feasible": True,
                        "objective_source": "exact_cart_milp",
                        "objective": float(pred_exact_obj),
                    }
                else:
                    pred_stats = {
                        **pred_stats,
                        "feasible": False,
                        "objective_source": "exact_cart_milp",
                        "objective": None,
                }
            if (
                args.problem == "cart"
                and ref_stats.get("objective_model") == "cart_detailed"
                and pred_stats["feasible"]
                and pred_stats["objective"] is None
            ):
                try:
                    print(
                        f"[cart_scoring] instance={inst_id}: no stored cost for predicted mask; "
                        "building and solving the detailed CART model with this mask fixed "
                        f"(time_limit={args.time_limit}s).",
                        flush=True,
                    )
                    pred_detailed = _solve_cart_detailed_milp(
                        meta,
                        mask=mask,
                        time_limit=args.time_limit,
                        solver_threads=args.solver_threads,
                        mip_gap=args.mip_gap,
                        max_open_facilities=args.max_k if args.max_k > 0 else None,
                        min_open_facilities=args.min_k,
                        fixed_assignments=cart_fixed_assignments,
                        formulation=args.cart_reference_formulation,
                        tee=True,
                    )
                    pred_detailed_cart_status = pred_detailed.get("status")
                    pred_detailed_cart_runtime = pred_detailed.get("solver_time_sec")
                    pred_stats = _apply_cart_detailed_prediction_record(
                        pred_stats,
                        ref_stats,
                        pred_detailed,
                    )
                except Exception as exc:
                    pred_detailed_cart_status = f"failed: {exc}"

            best_obj = ref_stats["best_obj"]
            best_mask = ref_stats["best_mask"]
            optimal_masks = ref_stats["optimal_masks"]
            summary["evaluated_predictions"] += 1
            if pred_stats["feasible"]:
                summary["feasible_predictions"] += 1
            if args.model_type == "gnn_fixing_mip":
                summary["fixed_variable_counts"].append(int(extra_info.get("num_fixed_variables", 0)))
                fallback_count = int(extra_info.get("fallback_count", 0))
                summary["fixing_fallback_total"] += fallback_count
                if fallback_count > 0:
                    summary["fixing_fallback_used"] += 1
            if args.model_type == "conpas_shared_encoder":
                summary.setdefault("conpas_selected_zero_counts", []).append(
                    int(extra_info.get("selected_zero_variables", 0))
                )
                summary.setdefault("conpas_selected_one_counts", []).append(
                    int(extra_info.get("selected_one_variables", 0))
                )
                summary.setdefault("conpas_allowed_disagreement_counts", []).append(
                    int(extra_info.get("allowed_disagreements", 0))
                )
                summary.setdefault("conpas_effective_fixed_counts", []).append(
                    int(extra_info.get("effective_fixed_variables", 0))
                )
                summary.setdefault("conpas_rank_var_counts", []).append(
                    int(extra_info.get("num_rank_neighbourhood_vars", 0))
                )
                summary.setdefault("conpas_delta_values", []).append(
                    int(extra_info.get("conpas_delta", 0))
                )
            has_reference = (best_obj is not None) and (not reference_infeasible)
            if has_reference:
                summary["with_reference"] += 1
            elif reference_infeasible:
                summary["reference_infeasible"] += 1
                summary["reference_infeasible_ids"].append(int(inst_id))
            else:
                summary["no_reference"] += 1
            exact_optimal = has_reference and (pred_stats["mask_key"] in optimal_masks)
            gap_rel = (
                _prediction_primal_gap(
                    pred_stats["objective"],
                    best_obj,
                    feasible=bool(pred_stats["feasible"]),
                    sense=_objective_sense(meta),
                )
                if has_reference
                else None
            )
            if has_reference:
                if gap_rel is not None:
                    summary["gap_values_all"].append(float(gap_rel))
                    summary["gap_all_known"] += 1
                    if gap_rel <= 0.01 + 1e-12:
                        summary["within_1pct"] += 1
                elif args.problem == "cart" and exact_reference_required:
                    summary["gap_all_not_applicable"] += 1
                else:
                    summary["gap_all_unknown"] += 1
            if (
                args.problem == "cart"
                and (exact_reference_required or detailed_constrained_reference)
                and has_reference
                and pred_stats["objective"] is not None
            ):
                exact_optimal = abs(float(pred_stats["objective"]) - float(best_obj)) <= (
                    1e-6 * max(abs(float(best_obj)), 1.0)
                )

            if reference_infeasible:
                print(
                    "  optimal: N/A "
                    f"(exact reference infeasible under injected constraints) source={reference_source}"
                )
            elif not has_reference:
                print("  optimal: N/A (no feasible samples)")
            else:
                print(
                    f"  optimal: obj={best_obj:.4f} mask={best_mask} "
                    f"n_opt_masks={len(optimal_masks)} exact={exact_optimal}"
                    f" source={reference_source}"
                )

            pred_obj_str = (
                f"{pred_stats['objective']:.4f}"
                if pred_stats["objective"] is not None
                else "N/A"
            )

            if args.problem == "ca":
                ca_cardinality_str = ""
                if args.ca_k_max is not None:
                    ca_cardinality_str = (
                        f"selected={pred_stats['selected_count']} "
                        f"k_max={pred_stats['k_max']} "
                    )
                print(
                    f"  pred: feasible={'yes' if pred_stats['feasible'] else 'no'} "
                    f"obj={pred_obj_str} gap={_fmt_gap_pct(gap_rel)} "
                    f"{ca_cardinality_str}"
                    f"conflicts={pred_stats['num_conflicts']} "
                    f"conflicted_items={pred_stats['num_conflicted_items']}"
                )
            elif args.problem == "qkp":
                qkp_cardinality_str = ""
                if args.qkp_k_min is not None or args.qkp_k_max is not None:
                    qkp_cardinality_str = (
                        f"selected={pred_stats['selected_count']} "
                        f"k=[{pred_stats['k_min']},{pred_stats['k_max']}] "
                    )
                print(
                    f"  pred: feasible={'yes' if pred_stats['feasible'] else 'no'} "
                    f"obj={pred_obj_str} gap={_fmt_gap_pct(gap_rel)} "
                    f"{qkp_cardinality_str}"
                    f"weight={pred_stats['total_weight']:.4f} "
                    f"capacity={pred_stats['capacity']:.4f}"
                )
            else:
                print(
                    f"  pred: feasible={'yes' if pred_stats['feasible'] else 'no'} "
                    f"obj={pred_obj_str} gap={_fmt_gap_pct(gap_rel)} "
                    f"open={pred_stats['open_count']} "
                    f"capacity={pred_stats['total_capacity']:.4f} "
                    f"peak_demand={pred_stats['peak_active_demand']:.4f}"
                )
                if pred_exact_cart_status is not None:
                    print(
                        f"  pred_exact_cart: status={pred_exact_cart_status} "
                        f"runtime={float(pred_exact_cart_runtime or 0.0):.2f}s "
                        f"source={pred_stats.get('objective_source')}"
                    )
                if pred_detailed_cart_status is not None:
                    print(
                        f"  pred_detailed_cart: status={pred_detailed_cart_status} "
                        f"runtime={float(pred_detailed_cart_runtime or 0.0):.2f}s "
                        f"source={pred_stats.get('objective_source')}"
                    )
                if pred_stats["feasible"] and pred_stats["objective"] is None:
                    print("  pred_note: feasible mask objective not found in sample pool")

            _append_jsonl(
                prediction_log_path,
                {
                    "instance_id": int(inst_id),
                    "model_type": args.model_type,
                    "model_name": model_name,
                    "problem": args.problem,
                    "status": status,
                    "milp_energy": float(milp_e),
                    "true_energy": float(true_e),
                    "runtime_sec": float(runtime_sec),
                    "solver_runtime_sec": (
                        float(solver_runtime_sec)
                        if solver_runtime_sec is not None
                        else None
                    ),
                    "variable_scores": (
                        [float(v) for v in linear_scores.tolist()]
                        if linear_scores is not None
                        else extra_info.get("variable_logits")
                    ),
                    "variable_probabilities": extra_info.get("variable_probabilities"),
                    "predicted_y": extra_info.get("predicted_y"),
                    "selected_y": [int(v) for v in mask],
                    "feasible": bool(pred_stats["feasible"]),
                    "objective": pred_stats["objective"],
                    "objective_gap_rel": gap_rel,
                    "objective_source": pred_stats.get("objective_source"),
                    "reference_source": reference_source,
                    "reference_status": detailed_reference_status,
                    "reference_runtime_sec": (
                        float(detailed_reference_runtime)
                        if detailed_reference_runtime is not None
                        else None
                    ),
                    "reference_objective": best_obj,
                    "reference_infeasible": bool(reference_infeasible),
                    "reference_mask": best_mask,
                    "reference_objective_model": ref_stats.get("objective_model"),
                    "detailed_reference_record": detailed_reference_record,
                    "pred_detailed_cart_status": pred_detailed_cart_status,
                    "pred_detailed_cart_runtime_sec": (
                        float(pred_detailed_cart_runtime)
                        if pred_detailed_cart_runtime is not None
                        else None
                    ),
                    "exact_optimal": bool(exact_optimal),
                    "selected_count": pred_stats.get("selected_count"),
                    "k_min": pred_stats.get("k_min"),
                    "k_max": pred_stats.get("k_max"),
                    "num_fixed_variables": extra_info.get("num_fixed_variables"),
                    "fallback_count": extra_info.get("fallback_count"),
                    "conpas_k0": extra_info.get("conpas_k0"),
                    "conpas_k1": extra_info.get("conpas_k1"),
                    "conpas_delta": extra_info.get("conpas_delta"),
                    "conpas_X0": extra_info.get("conpas_X0"),
                    "conpas_X1": extra_info.get("conpas_X1"),
                    "conpas_X0_probabilities": extra_info.get("conpas_X0_probabilities"),
                    "conpas_X1_probabilities": extra_info.get("conpas_X1_probabilities"),
                    "conpas_rank_target": extra_info.get("conpas_rank_target"),
                    "selected_zero_variables": extra_info.get("selected_zero_variables"),
                    "selected_one_variables": extra_info.get("selected_one_variables"),
                    "allowed_disagreements": extra_info.get("allowed_disagreements"),
                    "effective_fixed_variables": extra_info.get("effective_fixed_variables"),
                    "num_rank_neighbourhood_vars": extra_info.get("num_rank_neighbourhood_vars"),
                    "fixing_ratio_used": extra_info.get("fixing_ratio_used"),
                    "fixing_threshold_high": extra_info.get("fixing_threshold_high"),
                    "fixing_threshold_low": extra_info.get("fixing_threshold_low"),
                    "parallel_dives": extra_info.get("parallel_dives"),
                    "dive_workers": extra_info.get("dive_workers"),
                    "dive_time_limit": extra_info.get("dive_time_limit"),
                    "dive_solver_runtime_sum_sec": extra_info.get("dive_solver_runtime_sum_sec"),
                    "dive_solver_runtime_max_sec": extra_info.get("dive_solver_runtime_max_sec"),
                    "dive_wall_time_sec": extra_info.get("dive_wall_time_sec"),
                    "dive_wall_time_max_sec": extra_info.get("dive_wall_time_max_sec"),
                    "dive_statuses": extra_info.get("dive_statuses"),
                    "dive_objectives": extra_info.get("dive_objectives"),
                },
            )

            if has_reference:
                if exact_optimal:
                    summary["exact_optimal"] += 1
                else:
                    summary["misses"] += 1
                    if pred_stats["feasible"]:
                        summary["misses_feasible"] += 1
                    else:
                        summary["misses_infeasible"] += 1

                    if gap_rel is not None:
                        summary["misses_gap_known"] += 1
                        summary["miss_gap_values"].append(float(gap_rel))
                    elif args.problem == "cart" and exact_reference_required:
                        summary["miss_gap_not_applicable"] += 1
                    else:
                        summary["misses_gap_unknown"] += 1
        else:
            summary["failed"] += 1
            status_key = str(status)
            summary["failed_status_counts"][status_key] = (
                int(summary["failed_status_counts"].get(status_key, 0)) + 1
            )
            if status_key == "INFEASIBLE":
                summary["failed_infeasible"] += 1
            else:
                summary["failed_other"] += 1
            summary["failed_ids"].append(inst_id)
            print(f"{inst_id:<6} | FAILED ({status})")
            reference_source = None
            reference_infeasible = False
            failed_has_reference = False
            failed_reference_obj = None
            failed_reference_mask = None
            if exact_reference_required:
                reference_source = _exact_reference_source_name(args.problem)
                if exact_ref_mask is not None and exact_ref_obj is not None:
                    failed_has_reference = True
                    failed_reference_obj = float(exact_ref_obj)
                    failed_reference_mask = [int(v) for v in exact_ref_mask]
                    summary["failed_reference_feasible"] += 1
                    print(
                        f"  reference: obj={float(exact_ref_obj):.4f} "
                        f"mask={[int(v) for v in exact_ref_mask]} "
                        f"status={exact_ref_status} source={reference_source}"
                    )
                else:
                    if exact_ref_status == "INFEASIBLE":
                        reference_source = f"{reference_source}_infeasible"
                        reference_infeasible = True
                        summary["failed_reference_infeasible"] += 1
                        summary["reference_infeasible"] += 1
                        summary["reference_infeasible_ids"].append(int(inst_id))
                    else:
                        summary["failed_reference_other"] += 1
                    print(
                        f"  reference: status={exact_ref_status} "
                        f"source={reference_source}"
                    )
            elif detailed_constrained_reference:
                reference_source = detailed_reference_source
                reference_infeasible = detailed_reference_status == "infeasible"
                failed_reference_obj = ref_stats["best_obj"]
                failed_reference_mask = ref_stats["best_mask"]
                failed_has_reference = failed_reference_obj is not None
                if failed_has_reference:
                    summary["failed_reference_feasible"] += 1
                elif reference_infeasible:
                    summary["failed_reference_infeasible"] += 1
                    summary["reference_infeasible"] += 1
                    summary["reference_infeasible_ids"].append(int(inst_id))
                else:
                    summary["failed_reference_other"] += 1
            elif status_key == "INFEASIBLE":
                samples = pools.get(inst_id, [])
                ref_mask_filter = None
                if args.problem == "qkp" and (
                    args.qkp_k_min is not None or args.qkp_k_max is not None
                ):
                    ref_mask_filter = lambda mask_key: _eval_qkp_mask(
                        meta,
                        mask_key,
                        k_min=args.qkp_k_min,
                        k_max=args.qkp_k_max,
                    )["feasible"]
                failed_ref_stats = _collect_sample_reference(
                    meta,
                    samples,
                    mask_filter=ref_mask_filter,
                )
                reference_source = (
                    "samples_qkp_cardinality_filtered"
                    if ref_mask_filter is not None
                    else "samples_cart_detailed"
                    if failed_ref_stats.get("objective_model") == "cart_detailed"
                    else "samples"
                )
                if failed_ref_stats["best_obj"] is not None:
                    failed_has_reference = True
                    failed_reference_obj = float(failed_ref_stats["best_obj"])
                    failed_reference_mask = [
                        int(v) for v in failed_ref_stats["best_mask"]
                    ]

            failed_gap_rel = None
            if status_key == "INFEASIBLE":
                if failed_has_reference:
                    failed_gap_rel = _prediction_primal_gap(
                        None,
                        failed_reference_obj,
                        feasible=False,
                        sense=_objective_sense(meta),
                    )
                    summary["with_reference"] += 1
                    summary["misses"] += 1
                    summary["misses_infeasible"] += 1
                    summary["gap_values_all"].append(failed_gap_rel)
                    summary["gap_all_known"] += 1
                    summary["miss_gap_values"].append(failed_gap_rel)
                    summary["misses_gap_known"] += 1
                    print("  pred: feasible=no gap=100.00%")
                elif not reference_infeasible:
                    summary["no_reference"] += 1
            _append_jsonl(
                prediction_log_path,
                {
                    "instance_id": int(inst_id),
                    "model_type": args.model_type,
                    "model_name": model_name,
                    "problem": args.problem,
                    "status": status,
                    "runtime_sec": float(time.perf_counter() - t0),
                    "solver_runtime_sec": (
                        float(solver_runtime_sec)
                        if solver_runtime_sec is not None
                        else None
                    ),
                    "variable_scores": (
                        [float(v) for v in linear_scores.tolist()]
                        if linear_scores is not None
                        else extra_info.get("variable_logits")
                    ),
                    "variable_probabilities": extra_info.get("variable_probabilities"),
                    "predicted_y": extra_info.get("predicted_y"),
                    "selected_y": None,
                    "feasible": False,
                    "objective": None,
                    "objective_gap_rel": failed_gap_rel,
                    "reference_source": reference_source,
                    "reference_status": (
                        detailed_reference_status if detailed_constrained_reference else exact_ref_status
                    ),
                    "detailed_reference_record": detailed_reference_record,
                    "reference_objective": failed_reference_obj,
                    "reference_mask": failed_reference_mask,
                    "reference_objective_model": ref_stats.get("objective_model"),
                    "reference_infeasible": bool(reference_infeasible),
                    "exact_optimal": False,
                    "num_fixed_variables": extra_info.get("num_fixed_variables"),
                    "fallback_count": extra_info.get("fallback_count"),
                    "conpas_k0": extra_info.get("conpas_k0"),
                    "conpas_k1": extra_info.get("conpas_k1"),
                    "conpas_delta": extra_info.get("conpas_delta"),
                    "conpas_X0": extra_info.get("conpas_X0"),
                    "conpas_X1": extra_info.get("conpas_X1"),
                    "conpas_X0_probabilities": extra_info.get("conpas_X0_probabilities"),
                    "conpas_X1_probabilities": extra_info.get("conpas_X1_probabilities"),
                    "conpas_rank_target": extra_info.get("conpas_rank_target"),
                    "selected_zero_variables": extra_info.get("selected_zero_variables"),
                    "selected_one_variables": extra_info.get("selected_one_variables"),
                    "allowed_disagreements": extra_info.get("allowed_disagreements"),
                    "effective_fixed_variables": extra_info.get("effective_fixed_variables"),
                    "num_rank_neighbourhood_vars": extra_info.get("num_rank_neighbourhood_vars"),
                    "parallel_dives": extra_info.get("parallel_dives"),
                    "dive_workers": extra_info.get("dive_workers"),
                    "dive_time_limit": extra_info.get("dive_time_limit"),
                    "dive_solver_runtime_sum_sec": extra_info.get("dive_solver_runtime_sum_sec"),
                    "dive_solver_runtime_max_sec": extra_info.get("dive_solver_runtime_max_sec"),
                    "dive_wall_time_sec": extra_info.get("dive_wall_time_sec"),
                    "dive_wall_time_max_sec": extra_info.get("dive_wall_time_max_sec"),
                    "dive_statuses": extra_info.get("dive_statuses"),
                    "dive_objectives": extra_info.get("dive_objectives"),
                },
            )

    print("\nSummary")
    print("-" * 75)
    print(
        f"instances={summary['total']} solved={summary['solved']} failed={summary['failed']} "
        f"evaluated={summary['with_reference']} no_reference={summary['no_reference']} "
        f"reference_infeasible={summary['reference_infeasible']}"
    )
    print(
        f"exact_optimal={summary['exact_optimal']}/{summary['with_reference']} "
        f"({100.0 * _safe_ratio(summary['exact_optimal'], summary['with_reference']):.2f}%)"
    )
    if summary["gap_values_all"]:
        gap_all = np.asarray(summary["gap_values_all"], dtype=float)
        print(
            f"gap_pct: mean={100.0 * float(np.mean(gap_all)):.2f}% "
            f"median={100.0 * float(np.median(gap_all)):.2f}% "
            f"max={100.0 * float(np.max(gap_all)):.2f}% "
            f"(known={summary['gap_all_known']} unknown={summary['gap_all_unknown']} "
            f"n/a={summary['gap_all_not_applicable']})"
        )
        print(
            f"within_1pct={summary['within_1pct']}/{summary['with_reference']} "
            f"({100.0 * _safe_ratio(summary['within_1pct'], summary['with_reference']):.2f}%)"
        )
    else:
        print(
            f"gap_pct: N/A (known={summary['gap_all_known']} "
            f"unknown={summary['gap_all_unknown']} "
            f"n/a={summary['gap_all_not_applicable']})"
        )
    print(
        f"misses={summary['misses']} feasible={summary['misses_feasible']} "
        f"infeasible={summary['misses_infeasible']}"
    )
    if summary["miss_gap_values"]:
        gap_vals = np.asarray(summary["miss_gap_values"], dtype=float)
        print(
            f"miss_gap_pct: mean={100.0 * float(np.mean(gap_vals)):.2f}% "
            f"median={100.0 * float(np.median(gap_vals)):.2f}% "
            f"max={100.0 * float(np.max(gap_vals)):.2f}% "
            f"(known={summary['misses_gap_known']} unknown={summary['misses_gap_unknown']} "
            f"n/a={summary['miss_gap_not_applicable']})"
        )
    else:
        print(
            f"miss_gap_pct: N/A (known={summary['misses_gap_known']} "
            f"unknown={summary['misses_gap_unknown']} "
            f"n/a={summary['miss_gap_not_applicable']})"
        )
    if summary["failed_infeasible"] > 0:
        print(
            f"solver_infeasible={summary['failed_infeasible']}/{summary['total']} "
            f"({100.0 * _safe_ratio(summary['failed_infeasible'], summary['total']):.2f}%)"
        )
    if summary["failed_other"] > 0:
        other_counts = {
            k: v
            for k, v in summary["failed_status_counts"].items()
            if k != "INFEASIBLE"
        }
        print(
            f"solver_failed_other={summary['failed_other']}/{summary['total']} "
            f"({100.0 * _safe_ratio(summary['failed_other'], summary['total']):.2f}%) "
            f"statuses={other_counts}"
        )
    if (
        summary["failed_reference_feasible"] > 0
        or summary["failed_reference_infeasible"] > 0
        or summary["failed_reference_other"] > 0
    ):
        print(
            f"failed_exact_reference: feasible={summary['failed_reference_feasible']} "
            f"infeasible={summary['failed_reference_infeasible']} "
            f"other={summary['failed_reference_other']}"
        )
    if summary["failed_ids"]:
        print(f"failed_ids: {summary['failed_ids']}")
    if summary["reference_infeasible_ids"]:
        print(f"reference_infeasible_ids: {summary['reference_infeasible_ids']}")
    if summary["runtime_values"]:
        rt = np.asarray(summary["runtime_values"], dtype=float)
        print(
            f"runtime_sec: mean={float(np.mean(rt)):.2f} "
            f"median={float(np.median(rt)):.2f} max={float(np.max(rt)):.2f}"
        )
    if summary["solver_runtime_values"]:
        rt_solver = np.asarray(summary["solver_runtime_values"], dtype=float)
        print(
            f"solver_runtime_sec: mean={float(np.mean(rt_solver)):.2f} "
            f"median={float(np.median(rt_solver)):.2f} max={float(np.max(rt_solver)):.2f}"
        )
    print(
        f"overall_feasibility={summary['feasible_predictions']}/{summary['total']} "
        f"({100.0 * _safe_ratio(summary['feasible_predictions'], summary['total']):.2f}%)"
    )
    if summary["evaluated_predictions"] > 0:
        print(
            f"returned_mask_feasibility={summary['feasible_predictions']}/{summary['evaluated_predictions']} "
            f"({100.0 * _safe_ratio(summary['feasible_predictions'], summary['evaluated_predictions']):.2f}%)"
        )
    if summary["fixed_variable_counts"]:
        fixed = np.asarray(summary["fixed_variable_counts"], dtype=float)
        print(
            f"fixed_variables: mean={float(np.mean(fixed)):.2f} "
            f"median={float(np.median(fixed)):.2f} max={float(np.max(fixed)):.0f}"
        )
        print(
            f"fixing_fallbacks: used={summary['fixing_fallback_used']}/{summary['solved']} "
            f"total_steps={summary['fixing_fallback_total']}"
        )
    if summary.get("conpas_selected_zero_counts"):
        zero_counts = np.asarray(summary["conpas_selected_zero_counts"], dtype=float)
        one_counts = np.asarray(summary.get("conpas_selected_one_counts", []), dtype=float)
        delta_counts = np.asarray(summary.get("conpas_allowed_disagreement_counts", []), dtype=float)
        effective_fixed = np.asarray(summary.get("conpas_effective_fixed_counts", []), dtype=float)
        rank_counts = np.asarray(summary.get("conpas_rank_var_counts", []), dtype=float)
        print(
            "conpas_neighbourhood: "
            f"selected_zero_variables_mean={float(np.mean(zero_counts)):.2f} "
            f"selected_one_variables_mean={float(np.mean(one_counts)):.2f} "
            f"allowed_disagreements_mean={float(np.mean(delta_counts)):.2f} "
            f"rank_vars_mean={float(np.mean(rank_counts)):.2f} "
            f"effective_fixed_variables_mean={float(np.mean(effective_fixed)):.2f}"
        )

if __name__ == "__main__":
    main()
