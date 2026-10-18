# gnn/losses/listwise.py
from typing import Iterable, List
import torch

def listwise_softmax_ce(
    energies: torch.Tensor,
    label_types: Iterable[str],
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Instance-wise listwise softmax cross-entropy over masks.
    - Lower energy should rank higher -> logits = -energies / T.
    - Supports multiple positives by distributing target mass uniformly.
    - Operates only one feasible instances (positives + feasible negatives).
    - Goal: make the global optimum masks lowest in energy and push other feasible ones higher.

    Args:
        energies: [N] tensor of energies for all masks in one instance.
        label_types: iterable of length N with strings; positives are "positive".
        temperature: softmax temperature (default 1.0). >1.0 softens, <1.0 sharpens.

    Returns:
        Scalar loss (torch.Tensor).
    """
    if energies.dim() != 1:
        raise ValueError("energies must be 1-D [N].")

    labels: List[str] = list(label_types)
    pos_idx = [i for i, lt in enumerate(labels) if lt == "positive"]
    if len(pos_idx) == 0:
        raise ValueError("listwise_softmax_ce requires at least one positive in the instance.")

    logits = -energies / float(temperature)   # lower E -> higher logit
    logZ = torch.logsumexp(logits, dim=0)
    # Uniform mass over positives: -E_pos/T averaged, minus logZ
    target_term = logits[pos_idx].mean()
    return -(target_term - logZ)


def listwise_softmax_ce_soft(
    energies: torch.Tensor,
    regret_rel: torch.Tensor,
    feasible_mask: torch.Tensor,
    temperature: float = 1.0,
    tau: float = 1.0,
) -> torch.Tensor:
    """
    Regret-weighted listwise CE over FEASIBLES.
    Target mass p_target(i) ∝ exp(-regret_i / tau) for feasible i; 0 for infeasible.
    Coupling is via logsumexp over *all* candidates in p_pred.

    Args:
        energies: [N]
        regret_rel: [N] with NaN on infeasible/missing
        feasible_mask: BoolTensor [N]
        temperature: softmax temperature on predictions
        tau: temperature on targets (smaller -> more mass on lowest-regret)

    Returns:
        Scalar loss (0 if no feasible regrets available).
    """
    if energies.dim() != 1:
        raise ValueError("energies must be 1-D [N].")
    if regret_rel.shape != energies.shape:
        raise ValueError("regret_rel must be [N].")
    if feasible_mask.shape != energies.shape or feasible_mask.dtype != torch.bool:
        raise ValueError("feasible_mask must be BoolTensor [N].")

    feas = feasible_mask & torch.isfinite(regret_rel)
    if feas.sum() == 0:
        return energies.new_tensor(0.0)

    w = torch.exp(-regret_rel[feas] / float(tau))
    s = w.sum()
    if not torch.isfinite(s) or s.item() <= 0:
        return energies.new_tensor(0.0)

    logits = -energies / float(temperature)
    log_probs = torch.log_softmax(logits, dim=0)

    p_tgt = torch.zeros_like(log_probs)
    p_tgt[feas] = (w / s).to(log_probs.dtype)

    # True listwise cross entropy. Do not use KL(..., reduction="batchmean")
    # here: for a 1-D candidate vector that divides the ranking signal by N,
    # which makes gap-aware ranking far too weak on large candidate pools.
    return -(p_tgt * log_probs).sum()
