import math
import numpy as np
from typing import Dict, Tuple
from .schema import InstanceMeta

# ----------------------------
# Utilities
# ----------------------------

def rolling_sum(arr: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return arr.copy()
    cs = np.cumsum(np.pad(arr, (1, 0)))  # prefix sum
    res = cs[w:] - cs[:-w]
    # pad tail to keep length T (not used in peak)
    return np.concatenate([res, np.zeros_like(arr[:(len(arr) - len(res))])])

def _quantile(x: np.ndarray, q: float) -> float:
    return float(np.quantile(x, q)) if x.size else 0.0

def _p95(x: np.ndarray) -> float:
    if x.size == 0:
        return 1.0
    val = np.quantile(x, 0.95)
    return float(val if val > 1e-12 else 1.0)

def _safe_div(a: float, b: float) -> float:
    b = float(b)
    return float(a) / (b if abs(b) > 1e-12 else 1.0)

def _u1_scale(u: float, u1_max: float, use_log: bool = True) -> float:
    if use_log:
        return math.log1p(u) / max(1e-12, math.log1p(u1_max))
    else:
        return float(u) / max(1e-12, float(u1_max))

# ----------------------------
# Demand node features (scaled)
# ----------------------------

def demand_features_for_c(
    inc_t: Dict[int, int],
    T: int,
    TMFE: int,
    per_day_cap: float = 8.0,
) -> np.ndarray:
    """
    Build scaled demand features for site c:

      total_norm     = total / (per_day_cap * T)
      stdev_norm     = stdev / per_day_cap
      tmfe_peak_norm = TMFE_peak / (per_day_cap * TMFE)
      q50_norm, q90_norm = quantiles / per_day_cap
      Fourier(7/30/90) sin/cos are normalized by total (so already in [-1,1])

    Returns: [ total_n, stdev_n, tmfe_peak_n, q50_n, q90_n,
               cos7, sin7, cos30, sin30, cos90, sin90 ]
    """
    # Build 1..T vector
    x = np.array([inc_t.get(t, 0) for t in range(1, T + 1)], dtype=float)

    total = float(x.sum())
    mean = total / max(1, T)
    stdev = float(np.sqrt(np.mean((x - mean) ** 2)))

    # TMFE peak
    win = min(TMFE, T)
    if T >= win:
        roll = rolling_sum(x, win)[: T - win + 1]
        tmfe_peak = float(roll.max())
    else:
        tmfe_peak = float(x.sum())

    # quantiles
    q50 = _quantile(x, 0.5)
    q90 = _quantile(x, 0.9)

    # Fourier (7/30/90), normalized by total so they live in [-1,1]
    feats = []
    denom = max(1.0, total)
    t_idx = np.arange(1, T + 1, dtype=float)
    for P in (7.0, 30.0, 90.0):
        ang = 2.0 * math.pi * (t_idx / P)
        cP = float(np.dot(x, np.cos(ang))) / denom
        sP = float(np.dot(x, np.sin(ang))) / denom
        feats.extend([cP, sP])

    # scaling by known caps
    total_n     = _safe_div(total,     per_day_cap * float(T))
    stdev_n     = _safe_div(stdev,     per_day_cap)
    tmfe_peak_n = _safe_div(tmfe_peak, per_day_cap * float(win))
    q50_n       = _safe_div(q50,       per_day_cap)
    q90_n       = _safe_div(q90,       per_day_cap)

    return np.array([total_n, stdev_n, tmfe_peak_n, q50_n, q90_n] + feats, dtype=float)

# ----------------------------
# Facility node features (scaled)
# ----------------------------

def demand_weighted_U1_stats(
    U1_for_m: Dict[str, float],
    weights_c: Dict[str, float],
) -> Tuple[float, float, float, float]:
    """
    Demand-weighted transport stats to facility m.
    Returns raw stats (we scale later): (mean_w, min, p50, p90)
    """
    vals, wts = [], []
    for c, u in U1_for_m.items():
        vals.append(float(u))
        wts.append(float(weights_c.get(c, 0.0)))
    v = np.array(vals, dtype=float)
    w = np.array(wts, dtype=float)
    if v.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    if w.sum() <= 0:
        w = np.ones_like(v)
    w = w / w.sum()
    Umean = float((v * w).sum())
    Umin  = float(v.min())
    Up50  = _quantile(v, 0.5)
    Up90  = _quantile(v, 0.9)
    return Umean, Umin, Up50, Up90

def facility_features_for_m(
    F: float,
    fixed_cost_rate: float,
    U1_dw_stats: Tuple[float, float, float, float],
) -> np.ndarray:
    """
    Returns raw (unscaled) facility features:
      [F, fixed_cost_rate, Umean, Umin, Up50, Up90]
    """
    Umean, Umin, Up50, Up90 = U1_dw_stats
    return np.array([F, fixed_cost_rate, Umean, Umin, Up50, Up90], dtype=float)

# ----------------------------
# Top-level builder with scaling
# ----------------------------

# gnn/data/featurizer.py

def build_node_features(
    meta: InstanceMeta,
    TMFE: int,
    material_qc_cost: float = 1.0,
    per_day_cap: float = 8.0,
    u1_max: float = 3500.0,
    use_log_u1: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Build *scaled* node features with ADDED GLOBAL CONTEXT.
    """
    C, M, T = meta.C, meta.M, meta.T

    # ---- Demand node features (scaled with per_day_cap)
    Xc_list = []
    totals_c = {}  # to weight U1 by demand mass
    
    # Track raw totals for global stats
    raw_total_demand = 0.0
    
    for c in C:
        # Re-calculate raw total just for global stats safety
        raw_c_demand = sum(meta.INC[c].values())
        raw_total_demand += raw_c_demand
        
        f = demand_features_for_c(meta.INC[c], T, TMFE, per_day_cap=per_day_cap)
        Xc_list.append(f)
        totals_c[c] = float(f[0]) 

    Xc = np.vstack(Xc_list)  # [|C|, dc]

    # ---- Facility node features
    Xm_raw = []
    fixed_rate_list = []
    CAPMAX_list = []
    all_u1_values = [] # For global cost stat

    for m in M:
        capmax = float(meta.CAPMAX[m])
        CAPMAX_list.append(capmax)
        fix_rate = (float(meta.CIM[m]) + float(meta.CVM[m])) / float(T)
        fixed_rate_list.append(fix_rate)

        # Collect U1s for global stats
        for c in C:
            all_u1_values.append(meta.U1[c][m])

        Ustats_raw = demand_weighted_U1_stats({c: meta.U1[c][m] for c in C}, totals_c)
        Xm_raw.append(facility_features_for_m(capmax, fix_rate, Ustats_raw))

    Xm_raw = np.vstack(Xm_raw)

    # ---- Scale facility features
    CAPMAX_max = max(1.0, float(np.max(CAPMAX_list)))
    Xm = Xm_raw.copy()
    Xm[:, 0] = Xm_raw[:, 0] / CAPMAX_max
    
    p95_fix = _p95(np.array(fixed_rate_list, dtype=float))
    Xm[:, 1] = Xm_raw[:, 1] / p95_fix 

    # Mixed scaling for distance stats:
    # - Umean uses linear scaling to preserve relative distance differences.
    # - Min/P50/P90 keep log scaling to damp outliers.
    soft_max_dist = 1000.0
    Xm[:, 2] = Xm_raw[:, 2] / soft_max_dist
    for j in (3, 4, 5):
        Xm[:, j] = np.array([_u1_scale(u, u1_max, use_log=True) for u in Xm_raw[:, j]], dtype=float)

    # Global load ratio: total demand divided by capacity over the horizon.
    total_system_cap = sum(CAPMAX_list) * T
    global_load_ratio = raw_total_demand / max(1.0, total_system_cap)
    
    # Normalized mean edge cost provides an instance-level cost baseline.
    global_avg_u1 = np.mean(all_u1_values) if all_u1_values else 0.0
    global_avg_u1_norm = global_avg_u1 / soft_max_dist

    peak_active_demand = float(meta.globals.get("peak_active_demand") or 0.0)
    if peak_active_demand <= 0.0:
        from .reader import _compute_peak_demand
        peak_active_demand = _compute_peak_demand(meta.INC, T, TMFE)
    peak_demand_scaled = peak_active_demand / max(1.0, CAPMAX_max)

    global_feats = np.array([global_load_ratio, global_avg_u1_norm, peak_demand_scaled], dtype=float)

    # Repeat the global context for every constraint node.
    global_feats_c = np.tile(global_feats, (Xc.shape[0], 1))
    Xc = np.hstack([Xc, global_feats_c])

    # Tile and concatenate to Xm
    # Xm becomes [|M|, dm + 3]
    global_feats_m = np.tile(global_feats, (Xm.shape[0], 1))
    Xm = np.hstack([Xm, global_feats_m])
    # ==========================================

    globals_feats = {
        "T": float(T),
        "TMFE": float(TMFE),
        "total_therapies": float(meta.globals.get("total_therapies") or 0.0),
        "material_qc_cost": float(material_qc_cost),
        "peak_active_demand": float(peak_active_demand),
        "per_day_cap": float(per_day_cap),
        "u1_max": float(u1_max),
    }
    return Xc, Xm, globals_feats


def build_node_features_ca(
    meta: InstanceMeta,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    A_csr = meta.globals.get("A_csr")
    costs = meta.globals.get("costs")
    if A_csr is None or costs is None:
        raise ValueError("CA meta missing A_csr or costs.")

    n_constraints, n_bids = A_csr.shape
    constraint_deg = np.asarray(A_csr.sum(axis=1)).ravel().astype(float)
    bid_size = np.asarray(A_csr.sum(axis=0)).ravel().astype(float)
    value_arr = np.asarray(costs, dtype=float).ravel()

    max_deg = float(constraint_deg.max()) if constraint_deg.size else 1.0
    max_size = float(bid_size.max()) if bid_size.size else 1.0
    max_value = float(value_arr.max()) if value_arr.size else 1.0

    constraint_deg_norm = constraint_deg / max(1.0, max_deg)
    bid_size_norm = bid_size / max(1.0, max_size)
    value_norm = value_arr / max(1.0, max_value)

    value_per_item = value_arr / np.maximum(bid_size, 1.0)
    max_vpi = float(value_per_item.max()) if value_per_item.size else 1.0
    value_per_item_norm = value_per_item / max(1.0, max_vpi)

    competing_bids_per_item = np.maximum(constraint_deg - 1.0, 0.0)
    conflict_risk = np.asarray(A_csr.transpose().dot(competing_bids_per_item)).ravel().astype(float)
    conflict_risk = conflict_risk / np.maximum(bid_size, 1.0)
    max_conflict_risk = float(conflict_risk.max()) if conflict_risk.size else 1.0
    conflict_risk_norm = conflict_risk / max(1.0, max_conflict_risk)

    num_items = int(meta.globals.get("num_items", n_constraints))
    num_dummy_items = int(meta.globals.get("num_dummy_items", max(0, n_constraints - num_items)))
    is_dummy = np.zeros(n_constraints, dtype=float)
    if num_dummy_items > 0 and num_items < n_constraints:
        is_dummy[num_items:] = 1.0

    density = float(A_csr.nnz) / max(1, n_constraints * n_bids)
    avg_value = float(value_arr.mean()) if value_arr.size else 0.0
    avg_bid_size = float(bid_size.mean()) if bid_size.size else 0.0

    global_feats = np.array(
        [
            density,
            avg_value / max(1.0, max_value),
            avg_bid_size / max(1.0, max_size),
        ],
        dtype=float,
    )

    Xc = np.column_stack([constraint_deg_norm, is_dummy])
    Xc = np.hstack([Xc, np.tile(global_feats, (n_constraints, 1))])

    # Bid feature layout:
    #  0 value_norm
    #  1 bid_size_norm
    #  2 value_per_item_norm
    #  3 conflict_risk_norm
    Xm = np.column_stack([value_norm, bid_size_norm, value_per_item_norm, conflict_risk_norm])
    Xm = np.hstack([Xm, np.tile(global_feats, (n_bids, 1))])

    globals_feats = {
        "problem": "ca",
        "density": density,
        "avg_value": avg_value,
        "avg_bid_size": avg_bid_size,
        "avg_conflict_risk": float(conflict_risk.mean()) if conflict_risk.size else 0.0,
        "num_items": float(num_items),
        "num_dummy_items": float(num_dummy_items),
    }
    return Xc, Xm, globals_feats


def build_node_features_qkp(
    meta: InstanceMeta,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    p = np.asarray(meta.globals.get("p"), dtype=float).ravel()
    Q = np.asarray(meta.globals.get("Q"), dtype=float)
    w = np.asarray(meta.globals.get("w"), dtype=float).ravel()
    n_items = int(meta.globals.get("n_items", p.size))

    if p.size != n_items or w.size != n_items:
        raise ValueError("QKP meta has inconsistent p/w lengths versus n_items.")
    if Q.shape != (n_items, n_items):
        raise ValueError("QKP meta has inconsistent Q shape versus n_items.")

    pair_indices = meta.globals.get("pair_indices")
    if pair_indices is None:
        upper = np.triu(np.abs(Q) > 1e-6, k=1)
        pair_indices = np.column_stack(upper).astype(int)
    else:
        pair_indices = np.asarray(pair_indices, dtype=int).reshape(-1, 2)

    num_pairs = int(pair_indices.shape[0])
    max_pairs = max(1, (n_items * max(n_items - 1, 0)) // 2)
    global_density = float(num_pairs) / float(max_pairs)
    n_items_soft_cap = float(meta.globals.get("n_items_soft_cap", 200.0))
    n_items_norm = float(n_items) / max(n_items_soft_cap, 1.0)

    max_abs_p = float(np.max(np.abs(p))) if p.size else 1.0
    max_abs_w = float(np.max(np.abs(w))) if w.size else 1.0

    if num_pairs > 0:
        pair_values = np.asarray([Q[i, j] for i, j in pair_indices], dtype=float)
        abs_pair_values = np.abs(pair_values)
        max_abs_q = float(abs_pair_values.max()) if abs_pair_values.size else 1.0
        global_avg_q = float(abs_pair_values.mean()) / max(1.0, max_abs_q)
    else:
        pair_values = np.zeros(0, dtype=float)
        abs_pair_values = np.zeros(0, dtype=float)
        max_abs_q = 1.0
        global_avg_q = 0.0

    degree = np.zeros(n_items, dtype=float)
    abs_q_sum = np.zeros(n_items, dtype=float)
    for (i, j), abs_q in zip(pair_indices.tolist(), abs_pair_values.tolist()):
        degree[int(i)] += 1.0
        degree[int(j)] += 1.0
        abs_q_sum[int(i)] += float(abs_q)
        abs_q_sum[int(j)] += float(abs_q)

    degree_norm = degree / max(float(max(n_items - 1, 1)), 1.0)
    avg_q = abs_q_sum / np.maximum(degree, 1.0)
    avg_q_norm = avg_q / max(1.0, max_abs_q)

    Xm = np.column_stack(
        [
            p / max(1.0, max_abs_p),
            w / max(1.0, max_abs_w),
            degree_norm,
            avg_q_norm,
            np.full(n_items, global_density, dtype=float),
            np.full(n_items, global_avg_q, dtype=float),
            np.full(n_items, n_items_norm, dtype=float),
        ]
    ).astype(float)

    if num_pairs > 0:
        pair_i = pair_indices[:, 0]
        pair_j = pair_indices[:, 1]
        q_norm = abs_pair_values / max(1.0, max_abs_q)
        q_sign = np.sign(pair_values)
        w_sum_norm = (w[pair_i] + w[pair_j]) / max(2.0 * max(1.0, max_abs_w), 1.0)
        p_sum_norm = (p[pair_i] + p[pair_j]) / max(2.0 * max(1.0, max_abs_p), 1.0)
        Xc = np.column_stack(
            [
                q_norm,
                q_sign,
                w_sum_norm,
                p_sum_norm,
                np.full(num_pairs, global_density, dtype=float),
                np.full(num_pairs, global_avg_q, dtype=float),
                np.full(num_pairs, n_items_norm, dtype=float),
            ]
        ).astype(float)
    else:
        Xc = np.zeros((0, 7), dtype=float)

    globals_feats = {
        "problem": "qkp",
        "density": global_density,
        "avg_abs_q": float(abs_pair_values.mean()) if abs_pair_values.size else 0.0,
        "global_avg_q": global_avg_q,
        "capacity": float(meta.globals.get("W", 0.0)),
        "capacity_ratio": float(meta.globals.get("W", 0.0)) / max(float(np.sum(w)), 1.0),
        "n_items": float(n_items),
        "n_items_norm": n_items_norm,
        "n_items_soft_cap": n_items_soft_cap,
        "n_pairs": float(num_pairs),
    }
    return Xc, Xm, globals_feats
