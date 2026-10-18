# gnn/data/dataset.py
from typing import Dict, List
import numpy as np
import torch
from torch.utils.data import Dataset

from .schema import InstanceMeta, SampleRow
from .featurizer import (
    build_node_features,
    build_node_features_ca,
    build_node_features_qkp,
)
from .graph_builder import (
    build_bipartite_graph,
    build_bipartite_graph_ca,
    build_bipartite_graph_qkp,
)


class InstancePoolDataset(Dataset):
    """
    One item = one instance with its entire candidate pool (all samples for that instance).

    Returns (for each __getitem__):
      - container: dict with torch tensors:
          {
            "Xc":         (|C|, dc) float32,
            "Xm":         (|M|, dm) float32,
            "edge_index": (2, |E|)  int64,
            "edge_attr":  (|E|, de) float32,
          }
      - masks_t:    (N, |M|) float32 in {0,1}
      - targets_t:  (N,)     float32 (regret for feasible; scaled violation for infeasible)
      - feasible_t: (N,)     bool
      - meta_out:   dict with label_types, objectives, regret_rel, globals, etc.
    """
    def __init__(
        self,
        metas: Dict[int, InstanceMeta],
        pools: Dict[int, List[SampleRow]],
        TMFE: int,
        lambda_viol: float = 1.0,
        material_qc_cost: float = 1.0,
    ):
        self.metas = metas
        self.pools = pools
        self.instance_ids = sorted(set(metas.keys()) & set(pools.keys()))
        self.TMFE = TMFE
        self.lambda_viol = lambda_viol
        self.material_qc_cost = material_qc_cost

        # Cache (per-instance) of the static graph tensors
        self._cache_graph: Dict[int, tuple[dict, dict]] = {}

    def __len__(self) -> int:
        return len(self.instance_ids)

    def _build_graph(self, inst_id: int):
        """Build and cache the bipartite graph tensors for this instance."""
        if inst_id in self._cache_graph:
            return self._cache_graph[inst_id]

        meta = self.metas[inst_id]

        problem = meta.globals.get("problem", "cart")

        # Node features
        if problem == "ca":
            Xc, Xm, gfeats = build_node_features_ca(meta)
        elif problem == "qkp":
            Xc, Xm, gfeats = build_node_features_qkp(meta)
        else:
            Xc, Xm, gfeats = build_node_features(
                meta,
                TMFE=self.TMFE,
                material_qc_cost=self.material_qc_cost,
            )

        # Build graph; `container` must already be a dict of torch tensors
        # with keys {"Xc","Xm","edge_index","edge_attr"}.
        # The other returned values are not needed here since `container`
        # contains the tensors we pass to the model.
        if problem == "ca":
            _, _, _, _, container = build_bipartite_graph_ca(
                meta.globals["A_csr"],
                meta.globals["costs"],
                Xc,
                Xm,
            )
        elif problem == "qkp":
            _, _, _, _, container = build_bipartite_graph_qkp(
                meta,
                Xc,
                Xm,
            )
        else:
            _, _, _, _, container = build_bipartite_graph(
                meta, Xc, Xm, use_rbf_edge=True, u1_max=gfeats.get("u1_max", 3500.0)
            )

        if problem == "ca":
            container_globals = {
                "problem": "ca",
                "density": gfeats.get("density", 0.0),
                "avg_value": gfeats.get("avg_value", 0.0),
                "avg_bid_size": gfeats.get("avg_bid_size", 0.0),
                "avg_conflict_risk": gfeats.get("avg_conflict_risk", 0.0),
                "num_items": gfeats.get("num_items", 0.0),
                "num_dummy_items": gfeats.get("num_dummy_items", 0.0),
            }
        elif problem == "qkp":
            container_globals = {
                "problem": "qkp",
                "density": gfeats.get("density", 0.0),
                "avg_abs_q": gfeats.get("avg_abs_q", 0.0),
                "global_avg_q": gfeats.get("global_avg_q", 0.0),
                "capacity": gfeats.get("capacity", 0.0),
                "capacity_ratio": gfeats.get("capacity_ratio", 0.0),
                "n_items": gfeats.get("n_items", 0.0),
                "n_items_norm": gfeats.get("n_items_norm", 0.0),
                "n_pairs": gfeats.get("n_pairs", 0.0),
            }
        else:
            container_globals = {
                "T": gfeats["T"],
                "TMFE": gfeats["TMFE"],
                "total_therapies": gfeats["total_therapies"],
                "material_qc_cost": gfeats["material_qc_cost"],
                "peak_active_demand": gfeats["peak_active_demand"],
                "per_day_cap": gfeats.get("per_day_cap", 8.0),
                "u1_max": gfeats.get("u1_max", 0.0),
                "u1_linear_scale": gfeats.get("u1_linear_scale", 0.0),
            }

        self._cache_graph[inst_id] = (container, container_globals)
        return container, container_globals



    def __getitem__(self, idx: int):
        inst_id = self.instance_ids[idx]
        meta = self.metas[inst_id]

        container, glob = self._build_graph(inst_id)

        M = meta.M 
        masks = []
        targets = []
        feasible = []
        label_types = []
        objectives = []
        regrets = []

        for s in self.pools[inst_id]:
            # mask
            mask = [float(s.e1.get(m, 0) or 0) for m in M]
            masks.append(mask)

            label_types.append(s.label_type)
            feasible.append(bool(s.feasible))

            objectives.append(None if (not s.feasible) else float(s.objective))
            regrets.append(None if (not s.feasible) else float(s.regret_rel))

            if s.feasible and s.regret_rel is not None:
                t = float(s.regret_rel)
            else:
                viol = float(s.extras.get("violation", 0.0))
                viol_damped = float(np.log1p(max(viol, 0.0)))
                t = self.lambda_viol * viol_damped

            targets.append(t)

        masks_t = torch.tensor(np.asarray(masks, dtype=float), dtype=torch.float32)
        targets_t = torch.tensor(np.asarray(targets, dtype=float), dtype=torch.float32)
        feasible_t = torch.tensor(np.asarray(feasible, dtype=bool))

        meta_out = {
            "instance_id": inst_id,
            "label_types": label_types,
            "objectives": objectives,
            "regret_rel": regrets,
            "globals": glob,
            "M_order": M,
            "C_order": meta.C,
        }

        return container, masks_t, targets_t, feasible_t, meta_out
