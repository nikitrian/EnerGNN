# gnn/models/readout.py
import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class SetPoolReadout(nn.Module):
    """
    Coverage-aware readout for covering problems.

    Besides the usual sum of selected set/facility embeddings, this readout
    also summarizes which constraint/demand nodes become covered under the
    candidate mask. The coverage summary is MIP-representable because coverage
    is determined by binary indicators over graph neighborhoods.
    """

    def __init__(self, dh: int):
        super().__init__()
        self.out_dim = 4 * dh

    def forward(
        self,
        enc_out: Dict[str, torch.Tensor],
        masks: torch.Tensor,
        edge_index: torch.Tensor,
        nC: int,
    ) -> torch.Tensor:
        """
        enc_out: Output from BipartiteGNNEncoder
        masks: [N, M] binary tensor selecting sets/facilities
        edge_index: [2, E] with constraint nodes first and set nodes offset by nC
        nC: number of constraint nodes
        """
        hm = enc_out["hm"]  # [M, dh]
        hc = enc_out["hc"]  # [C, dh]
        hg = enc_out["hg"]  # [2*dh]

        n_constraints = int(nC)
        device = hm.device
        dtype = hm.dtype

        masks_f = masks.to(device=device, dtype=dtype)
        set_pool = masks_f @ hm  # [N, dh]

        edge_index = edge_index.to(device=device)
        src = edge_index[0].long()  # [E]
        tgt = edge_index[1].long() - n_constraints  # [E]

        if tgt.numel() > 0:
            edge_mask_values = masks_f.index_select(dim=1, index=tgt)  # [N, E]
            coverage_raw = torch.zeros(
                (masks_f.shape[0], n_constraints),
                dtype=dtype,
                device=device,
            )
            coverage_raw.scatter_add_(
                1,
                src.unsqueeze(0).expand(masks_f.shape[0], -1),
                edge_mask_values,
            )
        else:
            coverage_raw = torch.zeros(
                (masks_f.shape[0], n_constraints),
                dtype=dtype,
                device=device,
            )

        covered = coverage_raw.clamp_(0.0, 1.0)
        coverage_pool = covered @ hc  # [N, dh]

        hg_exp = hg.unsqueeze(0).expand(masks_f.shape[0], -1)
        return torch.cat([set_pool, coverage_pool, hg_exp], dim=-1)


class QKPReadout(nn.Module):
    """
    Selected-item-only QKP readout used by the readout ablation.
    """

    def __init__(self, dh: int):
        super().__init__()
        self.out_dim = 3 * dh

    def forward(
        self,
        enc_out: Dict[str, torch.Tensor],
        masks: torch.Tensor,
        graph: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        hm = enc_out["hm"]  # [M, dh]
        hg = enc_out["hg"]  # [2*dh]
        masks_f = masks.to(device=hm.device, dtype=hm.dtype)
        set_pool = masks_f @ hm
        hg_exp = hg.unsqueeze(0).expand(masks_f.shape[0], -1)
        return torch.cat([set_pool, hg_exp], dim=-1)


class QKPSizeAwareReadout(nn.Module):
    """
    QKP readout that stays MILP-friendly while exposing graph-size-aware
    summaries:
      - scaled selected-item embedding pool
      - scaled unselected-item embedding pool
      - selected item fraction
      - used capacity mass
      - global graph embedding

    The complement pool is implemented as const_sum_h - selected_pool, where
    const_sum_h is fixed for the instance. All scalar summaries use fixed
    denominators, so the exact MIP remains linear in the item binaries.
    """

    def __init__(self, dh: int):
        super().__init__()
        self.summary_dim = 2
        self.out_dim = (4 * dh) + self.summary_dim

    @staticmethod
    def _weights_and_capacity(
        graph: Optional[Dict[str, Any]],
        n_items: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, float]:
        if graph is not None and "qkp_weights_t" in graph:
            weights = graph["qkp_weights_t"].to(device=device, dtype=dtype).view(-1)
            if weights.numel() != n_items:
                raise ValueError(
                    f"QKPSizeAwareReadout expected {n_items} weights, got {weights.numel()}."
                )
            capacity_raw = graph.get("qkp_capacity_t", 0.0)
            if isinstance(capacity_raw, torch.Tensor):
                capacity = float(capacity_raw.reshape(-1)[0].item()) if capacity_raw.numel() else 0.0
            else:
                capacity = float(capacity_raw)
            return weights, capacity
        return torch.zeros(n_items, device=device, dtype=dtype), 0.0

    def forward(
        self,
        enc_out: Dict[str, torch.Tensor],
        masks: torch.Tensor,
        graph: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        hm = enc_out["hm"]  # [items, dh]
        hg = enc_out["hg"]  # [2*dh]
        masks_f = masks.to(device=hm.device, dtype=hm.dtype)
        n_samples, n_items = masks_f.shape

        inv_sqrt_n = 1.0 / math.sqrt(max(float(n_items), 1.0))
        selected_item_pool = (masks_f @ hm) * inv_sqrt_n
        const_item_pool = hm.sum(dim=0, keepdim=True).expand(n_samples, -1) * inv_sqrt_n
        unselected_item_pool = const_item_pool - selected_item_pool

        selected_frac = masks_f.sum(dim=1, keepdim=True) / max(float(n_items), 1.0)

        weights, capacity = self._weights_and_capacity(graph, n_items, hm.device, hm.dtype)
        weight_num = masks_f @ weights.view(-1, 1)
        nonzero_capacity = float(abs(capacity) > 1e-12)
        capacity_safe = capacity if abs(capacity) > 1e-12 else 1.0
        weight_mass = (weight_num / capacity_safe) * nonzero_capacity

        hg_exp = hg.unsqueeze(0).expand(n_samples, -1)
        return torch.cat(
            [
                selected_item_pool,
                unselected_item_pool,
                selected_frac,
                weight_mass,
                hg_exp,
            ],
            dim=-1,
        )


class QKPDirectYReadout(nn.Module):
    """
    QKP ablation readout that exposes the raw decision vector to the energy
    head. This is intended for fixed-size illustrative ablations only: it is
    order-dependent and pads masks to a configured width.

    The output order is:
      [
        sum_i y_i h_i,
        padded raw y,
        h_G,
      ]
    """

    def __init__(self, dh: int, direct_y_dim: int):
        super().__init__()
        self.direct_y_dim = int(direct_y_dim)
        if self.direct_y_dim <= 0:
            raise ValueError("QKPDirectYReadout requires direct_y_dim > 0.")
        self.out_dim = (3 * dh) + self.direct_y_dim

    def forward(
        self,
        enc_out: Dict[str, torch.Tensor],
        masks: torch.Tensor,
        graph: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        hm = enc_out["hm"]  # [items, dh]
        hg = enc_out["hg"]  # [2*dh]
        masks_f = masks.to(device=hm.device, dtype=hm.dtype)
        n_samples, n_items = masks_f.shape
        if n_items > self.direct_y_dim:
            raise ValueError(
                f"QKPDirectYReadout got mask width {n_items}, "
                f"but direct_y_dim={self.direct_y_dim}."
            )

        selected_item_pool = masks_f @ hm
        if n_items < self.direct_y_dim:
            pad = masks_f.new_zeros((n_samples, self.direct_y_dim - n_items))
            y_direct = torch.cat([masks_f, pad], dim=1)
        else:
            y_direct = masks_f

        hg_exp = hg.unsqueeze(0).expand(n_samples, -1)
        return torch.cat([selected_item_pool, y_direct, hg_exp], dim=-1)


class CAPooledOnlyReadout(nn.Module):
    """
    CA ablation readout with mask-conditioned pooling only.

    This removes the scalar z_lin summaries while retaining the two natural
    pooled quantities for auctions: selected bid embeddings and occupied item
    embeddings.
    """

    def __init__(self, dh: int):
        super().__init__()
        self.out_dim = 4 * dh

    def forward(
        self,
        enc_out: Dict[str, torch.Tensor],
        masks: torch.Tensor,
        graph: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        if graph is None or "incidence_t" not in graph:
            raise ValueError("CAPooledOnlyReadout requires graph['incidence_t'].")

        hm = enc_out["hm"]  # [bids, dh]
        hc = enc_out["hc"]  # [items, dh]
        hg = enc_out["hg"]  # [2*dh]
        masks_f = masks.to(device=hm.device, dtype=hm.dtype)
        incidence = graph["incidence_t"].to(device=hm.device, dtype=hm.dtype)  # [items, bids]

        selected_bid_pool = masks_f @ hm
        loads = masks_f @ incidence.transpose(0, 1)
        occupied = loads.clamp(max=1.0)
        occupied_item_pool = occupied @ hc

        hg_exp = hg.unsqueeze(0).expand(masks_f.shape[0], -1)
        return torch.cat([selected_bid_pool, occupied_item_pool, hg_exp], dim=-1)


class CASumOnlyPlusReadout(nn.Module):
    """
    CA readout that stays MILP-friendly while adding useful bid-side scalar sums.

    Compared with `sum_only`, this keeps the same pooled embeddings but adds:
      - sum of normalized selected bid values
      - sum of normalized selected bid bundle sizes
      - sum of normalized selected conflict risk

    These are linear in the bid mask, so they do not require any reciprocal or
    one-hot count gadget in the MILP.
    """

    def __init__(self, dh: int):
        super().__init__()
        self.summary_dim = 8
        self.out_dim = (4 * dh) + self.summary_dim

    def forward(
        self,
        enc_out: Dict[str, torch.Tensor],
        masks: torch.Tensor,
        graph: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        if graph is None or "incidence_t" not in graph:
            raise ValueError("CASumOnlyPlusReadout requires graph['incidence_t'].")

        hm = enc_out["hm"]          # [M, dh]
        hc = enc_out["hc"]          # [C, dh]
        hg = enc_out["hg"]          # [2*dh]
        xm_raw = enc_out["Xm_raw"]  # [M, dm]
        xc_raw = enc_out["Xc_raw"]  # [C, dc]
        incidence = graph["incidence_t"].float()  # [C, M]

        masks_f = masks.float()
        n_samples, n_bids = masks_f.shape
        n_constraints = incidence.shape[0]

        selected_bid_pool = masks_f @ hm
        selected_count = masks_f.sum(dim=1, keepdim=True)

        loads = masks_f @ incidence.transpose(0, 1)         # [N, C]
        occupied = loads.clamp(max=1.0)
        overload = torch.relu(loads - 1.0)

        occupied_item_pool = occupied @ hc

        value_norm = xm_raw[:, 0:1]
        bid_size_norm = xm_raw[:, 1:2]
        conflict_risk_norm = xm_raw[:, 3:4]
        selected_value_sum = masks_f @ value_norm
        selected_bundle_sum = masks_f @ bid_size_norm
        selected_conflict_risk_sum = masks_f @ conflict_risk_norm

        is_dummy = xc_raw[:, 1:2] if xc_raw.shape[1] > 1 else torch.zeros_like(xc_raw[:, :1])
        dummy_count = is_dummy.sum().clamp_min(1.0)
        occupied_dummy_frac = (occupied @ is_dummy) / dummy_count

        occupied_frac = occupied.sum(dim=1, keepdim=True) / max(float(n_constraints), 1.0)
        conflict_item_frac = (overload > 0).float().sum(dim=1, keepdim=True) / max(float(n_constraints), 1.0)
        conflict_mass_norm = overload.sum(dim=1, keepdim=True) / max(float(n_constraints), 1.0)
        selected_frac = selected_count / max(float(n_bids), 1.0)

        summary = torch.cat(
            [
                selected_frac,
                occupied_frac,
                occupied_dummy_frac,
                conflict_item_frac,
                conflict_mass_norm,
                selected_value_sum,
                selected_bundle_sum,
                selected_conflict_risk_sum,
            ],
            dim=-1,
        )

        hg_exp = hg.unsqueeze(0).expand(n_samples, -1)
        return torch.cat(
            [
                selected_bid_pool,
                occupied_item_pool,
                hg_exp,
                summary,
            ],
            dim=-1,
        )


class CADirectYReadout(nn.Module):
    """
    CA ablation readout that exposes the raw bid mask to the energy head.

    This is intended for fixed-size diagnostic ablations. It keeps the same
    mask-conditioned pools as CAPooledOnlyReadout and appends a padded raw y.
    """

    def __init__(self, dh: int, direct_y_dim: int):
        super().__init__()
        self.direct_y_dim = int(direct_y_dim)
        if self.direct_y_dim <= 0:
            raise ValueError("CADirectYReadout requires direct_y_dim > 0.")
        self.out_dim = (4 * dh) + self.direct_y_dim

    def forward(
        self,
        enc_out: Dict[str, torch.Tensor],
        masks: torch.Tensor,
        graph: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        if graph is None or "incidence_t" not in graph:
            raise ValueError("CADirectYReadout requires graph['incidence_t'].")

        hm = enc_out["hm"]  # [bids, dh]
        hc = enc_out["hc"]  # [items, dh]
        hg = enc_out["hg"]  # [2*dh]
        masks_f = masks.to(device=hm.device, dtype=hm.dtype)
        n_samples, n_bids = masks_f.shape
        if n_bids > self.direct_y_dim:
            raise ValueError(
                f"CADirectYReadout got mask width {n_bids}, "
                f"but direct_y_dim={self.direct_y_dim}."
            )

        incidence = graph["incidence_t"].to(device=hm.device, dtype=hm.dtype)  # [items, bids]
        selected_bid_pool = masks_f @ hm
        loads = masks_f @ incidence.transpose(0, 1)
        occupied = loads.clamp(max=1.0)
        occupied_item_pool = occupied @ hc

        if n_bids < self.direct_y_dim:
            pad = masks_f.new_zeros((n_samples, self.direct_y_dim - n_bids))
            y_direct = torch.cat([masks_f, pad], dim=1)
        else:
            y_direct = masks_f

        hg_exp = hg.unsqueeze(0).expand(n_samples, -1)
        return torch.cat(
            [selected_bid_pool, occupied_item_pool, y_direct, hg_exp],
            dim=-1,
        )
