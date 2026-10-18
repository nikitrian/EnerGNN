# gnn/losses/combined.py
from typing import Iterable, Dict, Any, Optional, Tuple
import torch

from .listwise import listwise_softmax_ce, listwise_softmax_ce_soft
from .calibration import adaptive_huber_delta, feasible_huber, infeasible_huber, infeasible_margin

def ebm_loss(
    energies: torch.Tensor,                      # [N]
    label_types: Iterable[str],                  # list[str] for this instance
    feasible_mask: torch.Tensor,                 # BoolTensor [N]
    regret_rel: Optional[torch.Tensor] = None,   # [N] with NaN for infeasible/missing
    infeasible_target: Optional[torch.Tensor] = None,  # [N] optional per-sample target for infeasible
    alpha: float = 1.0,
    temperature: float = 0.1,
    lambda_rank: float = 1.0,
    lambda_inf_cal: float = 0.1,
    rank_tau: float = 0.05,
    tie_epsilon: float = 0.015,
    inf_delta: float = 0.1,
    gap_aware_ranking: bool = True,
    rank_feasible_only: bool = False,
    infeasible_margin_loss: bool = False,
    infeasible_margin_anchor: str = "best",
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Combined per-instance loss."""

    labels_list = list(label_types)
    if regret_rel is None:
        regret_rel = torch.full_like(energies, float('nan'))

    delta_inst = adaptive_huber_delta(regret_rel)

    feas_with_reg = feasible_mask & torch.isfinite(regret_rel)
    if tie_epsilon and tie_epsilon > 0 and feas_with_reg.any():
        best_regret = torch.min(regret_rel[feas_with_reg])
        threshold = best_regret + tie_epsilon
        expanded = feas_with_reg & (regret_rel <= threshold)
        if expanded.any():
            for idx in expanded.nonzero(as_tuple=False).view(-1).tolist():
                labels_list[idx] = "positive"

    rank_energies = energies
    rank_regret = regret_rel
    rank_feasible = feasible_mask
    rank_labels = labels_list
    if rank_feasible_only:
        rank_indices = feasible_mask.nonzero(as_tuple=False).view(-1)
        rank_energies = energies[rank_indices]
        rank_regret = regret_rel[rank_indices]
        rank_feasible = feasible_mask[rank_indices]
        rank_labels = [labels_list[int(i)] for i in rank_indices.tolist()]

    rank_feas_with_reg = rank_feasible & torch.isfinite(rank_regret)
    if gap_aware_ranking and rank_feas_with_reg.any():
        L_rank = listwise_softmax_ce_soft(
            energies=rank_energies,
            regret_rel=rank_regret,
            feasible_mask=rank_feasible,
            temperature=temperature,
            tau=rank_tau,
        )
    else:
        if any(lt == "positive" for lt in rank_labels):
            L_rank = listwise_softmax_ce(rank_energies, rank_labels, temperature=temperature)
        else:
            L_rank = energies.new_tensor(0.0)

    L_cal = feasible_huber(energies, regret_rel, delta=delta_inst)
    L_inf_cal = energies.new_tensor(0.0)
    if infeasible_target is not None:
        if infeasible_target.shape != energies.shape:
            raise ValueError("infeasible_target must be [N].")
        if lambda_inf_cal > 0:
            if infeasible_margin_loss:
                L_inf_cal = infeasible_margin(
                    energies,
                    infeasible_target,
                    feasible_mask,
                    anchor=infeasible_margin_anchor,
                )
            else:
                L_inf_cal = infeasible_huber(
                    energies, infeasible_target, feasible_mask, delta=inf_delta
                )

    total = (
        lambda_rank * L_rank
        + alpha * L_cal
        + lambda_inf_cal * L_inf_cal
    )

    parts: Dict[str, Any] = {
        "rank": float(L_rank.detach().item()),
        "cal": float(L_cal.detach().item()),
        "inf_cal": float(L_inf_cal.detach().item()),
        "lambda_rank": float(lambda_rank),
        "delta_huber": float(delta_inst),
        "inf_delta": float(inf_delta),
        "tie_eps": float(tie_epsilon),
        "gap_aware_ranking": bool(gap_aware_ranking),
        "rank_feasible_only": bool(rank_feasible_only),
        "infeasible_margin_loss": bool(infeasible_margin_loss),
        "infeasible_margin_anchor": str(infeasible_margin_anchor),
    }
    return total, parts
