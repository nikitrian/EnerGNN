# GNN/data/graph_builder.py
from typing import Tuple, Dict, Any, Iterable, Optional
import math
import numpy as np
import torch
import scipy.sparse as sp

from .schema import InstanceMeta

def _scale_u1(u: float, u1_max: float = 3500.0, use_log: bool = True) -> float:
    """Scale a transport cost to ~[0,1]."""
    if use_log:
        return math.log1p(u) / max(1e-12, math.log1p(u1_max))
    return float(u) / max(1e-12, float(u1_max))

def rbf_expand_unit_interval(
    x: np.ndarray,
    centers: Optional[Iterable[float]] = None,
    gamma: float = 8.0,
) -> np.ndarray:
    """
    RBF features for inputs assumed to be in [0,1].
    x: [N] np.array of scaled values
    centers: list of centers in [0,1]; default: 5 evenly spaced
    gamma: width parameter (larger = tighter bumps)
    returns: [N, K]
    """
    if centers is None:
        centers = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=float)
    else:
        centers = np.array(list(centers), dtype=float)
    x = x.reshape(-1, 1)  # [N,1]
    return np.exp(-gamma * (x - centers.reshape(1, -1)) ** 2)

def build_bipartite_graph(
    meta: InstanceMeta,
    Xc: np.ndarray,
    Xm: np.ndarray,
    use_rbf_edge: bool = True,
    u1_max: float = 3500.0,
    use_log_u1: bool = True,
    rbf_gamma: float = 8.0,
    rbf_centers: Optional[Iterable[float]] = None,   # centers in [0,1]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Build a complete bipartite graph C->M with scaled U1 edge features.

    Returns:
      Xc_t:         [|C|, dc] float32
      Xm_t:         [|M|, dm] float32
      edge_index:   [2, |E|]  int64   (indices over the combined node set: C first, then M offset by |C|)
      edge_attr_t:  [|E|, de] float32 (de = 1 + K if RBF used, else 1)
      container:    dict with {"Xc","Xm","edge_index","edge_attr","nC","nM"}
    """
    C, M = meta.C, meta.M
    nC, nM = len(C), len(M)

    src: list[int] = []
    dst: list[int] = []
    edge_attr_list: list[np.ndarray] = []

    for i_c, c in enumerate(C):
        for i_m, m in enumerate(M):
            src.append(i_c)           # c node index
            dst.append(nC + i_m)      # m node index with offset
            u_raw = float(meta.U1[c][m])
            u_s   = _scale_u1(u_raw, u1_max=u1_max, use_log=use_log_u1)

            if use_rbf_edge:
                rbf = rbf_expand_unit_interval(
                    np.array([u_s], dtype=float),
                    centers=rbf_centers,
                    gamma=rbf_gamma,
                )[0]  # [K]
                ea = np.concatenate([np.array([u_s], dtype=float), rbf], axis=-1)
            else:
                ea = np.array([u_s], dtype=float)

            edge_attr_list.append(ea)

    edge_index  = torch.tensor([src, dst], dtype=torch.long)                  # [2, |E|]
    edge_attr_t = torch.tensor(np.vstack(edge_attr_list), dtype=torch.float32)  # [|E|, de]
    Xc_t        = torch.tensor(Xc, dtype=torch.float32)                       # [|C|, dc]
    Xm_t        = torch.tensor(Xm, dtype=torch.float32)                       # [|M|, dm]

    container: Dict[str, Any] = {
        "Xc": Xc_t,
        "Xm": Xm_t,
        "edge_index": edge_index,
        "edge_attr": edge_attr_t,
        "nC": nC,
        "nM": nM,
    }
    return Xc_t, Xm_t, edge_index, edge_attr_t, container

def build_bipartite_graph_ca(
    A_csr: sp.csr_matrix,
    costs: np.ndarray,
    Xc: np.ndarray,
    Xm: np.ndarray,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Build a sparse bipartite graph for combinatorial auctions using item-bid
    incidence edges with CA-specific edge features.

    Edge features:
      1. bid_value_by_item_degree_norm: bid value normalized by competing bids on the item
      2. item_demand_pressure_norm: normalized item degree
      3. bid_bundle_size_norm: normalized bundle size of the incident bid
    """
    A_coo = A_csr.tocoo()
    nC, nM = A_coo.shape

    src = A_coo.row.astype(int)
    dst = (A_coo.col + nC).astype(int)

    item_degree = np.asarray(A_csr.sum(axis=1)).ravel().astype(float)
    bid_bundle_size = np.asarray(A_csr.sum(axis=0)).ravel().astype(float)
    bid_values = np.asarray(costs, dtype=float).ravel()

    max_item_degree = float(item_degree.max()) if item_degree.size else 1.0
    max_bundle_size = float(bid_bundle_size.max()) if bid_bundle_size.size else 1.0

    item_degree_edge = item_degree[A_coo.row]
    item_demand_pressure_norm = item_degree_edge / max(1.0, max_item_degree)

    bundle_size_edge = bid_bundle_size[A_coo.col]
    bid_bundle_size_norm = bundle_size_edge / max(1.0, max_bundle_size)

    bid_value_by_item_degree = bid_values[A_coo.col] / np.maximum(item_degree_edge, 1.0)
    max_bid_value_by_item_degree = (
        float(bid_value_by_item_degree.max()) if bid_value_by_item_degree.size else 1.0
    )
    bid_value_by_item_degree_norm = (
        bid_value_by_item_degree / max(1.0, max_bid_value_by_item_degree)
    )

    edge_attr = np.column_stack(
        [
            bid_value_by_item_degree_norm,
            item_demand_pressure_norm,
            bid_bundle_size_norm,
        ]
    ).astype(float)

    edge_index = torch.tensor([src.tolist(), dst.tolist()], dtype=torch.long)
    edge_attr_t = torch.tensor(edge_attr, dtype=torch.float32)
    Xc_t = torch.tensor(Xc, dtype=torch.float32)
    Xm_t = torch.tensor(Xm, dtype=torch.float32)
    incidence_t = torch.tensor(A_csr.toarray(), dtype=torch.float32)

    container: Dict[str, Any] = {
        "Xc": Xc_t,
        "Xm": Xm_t,
        "edge_index": edge_index,
        "edge_attr": edge_attr_t,
        "incidence_t": incidence_t,
        "nC": nC,
        "nM": nM,
        "problem": "ca",
    }
    return Xc_t, Xm_t, edge_index, edge_attr_t, container


def build_bipartite_graph_qkp(
    meta: InstanceMeta,
    Xc: np.ndarray,
    Xm: np.ndarray,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Build a sparse bipartite graph for quadratic knapsack using interaction-item
    incidence edges. Each interaction node (i, j) connects to the two item nodes
    it couples in the quadratic profit matrix. The returned container also keeps
    fixed item weights and capacity so size-aware QKP readouts can stay exact-MIP
    representable.
    """
    Q = np.asarray(meta.globals.get("Q"), dtype=float)
    w = np.asarray(meta.globals.get("w"), dtype=float).ravel()
    W = float(meta.globals.get("W", 0.0))
    pair_indices = meta.globals.get("pair_indices")
    if pair_indices is None:
        upper = np.triu(np.abs(Q) > 1e-6, k=1)
        pair_indices = np.column_stack(upper).astype(int)
    else:
        pair_indices = np.asarray(pair_indices, dtype=int).reshape(-1, 2)

    nC = int(pair_indices.shape[0])
    nM = int(Xm.shape[0])
    if Q.shape != (nM, nM):
        raise ValueError("QKP graph builder expected Q to have shape [n_items, n_items].")

    if nC > 0:
        pair_vals = np.asarray([Q[i, j] for i, j in pair_indices], dtype=float)
        abs_pair_vals = np.abs(pair_vals)
        max_abs_q = float(abs_pair_vals.max()) if abs_pair_vals.size else 1.0

        src = np.repeat(np.arange(nC, dtype=int), 2)
        dst_items = pair_indices.reshape(-1)
        dst = dst_items + nC

        q_norm = (abs_pair_vals / max(1.0, max_abs_q)).repeat(2)
        q_sign = np.sign(pair_vals).repeat(2)
        weight_ratio_vals = (w[pair_indices[:, 0]] + w[pair_indices[:, 1]]) / max(float(W), 1.0)
        weight_ratio = weight_ratio_vals.repeat(2)
        edge_attr = np.column_stack([q_norm, q_sign, weight_ratio]).astype(float)
    else:
        src = np.zeros(0, dtype=int)
        dst = np.zeros(0, dtype=int)
        edge_attr = np.zeros((0, 3), dtype=float)

    edge_index = torch.tensor([src.tolist(), dst.tolist()], dtype=torch.long)
    edge_attr_t = torch.tensor(edge_attr, dtype=torch.float32)
    Xc_t = torch.tensor(Xc, dtype=torch.float32)
    Xm_t = torch.tensor(Xm, dtype=torch.float32)

    container: Dict[str, Any] = {
        "Xc": Xc_t,
        "Xm": Xm_t,
        "edge_index": edge_index,
        "edge_attr": edge_attr_t,
        "qkp_weights_t": torch.tensor(w, dtype=torch.float32),
        "qkp_capacity_t": torch.tensor(float(W), dtype=torch.float32),
        "nC": nC,
        "nM": nM,
        "problem": "qkp",
    }
    return Xc_t, Xm_t, edge_index, edge_attr_t, container
