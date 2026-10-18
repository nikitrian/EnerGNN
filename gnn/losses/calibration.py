# gnn/losses/calibration.py
import torch
import torch.nn.functional as F



def adaptive_huber_delta(
    regret_rel: torch.Tensor,
    kappa: float = 0.15,
    min_delta: float = 0.01,
    max_delta: float = 0.5,
) -> float:
    '''Scale the Huber transition based on the spread of feasible regrets.'''
    feas = regret_rel[torch.isfinite(regret_rel)]
    if feas.numel() == 0:
        return float(min_delta)
    span = float((feas.max() - feas.min()).item())
    delta = kappa * span
    delta = max(min_delta, min(delta, max_delta))
    return float(delta)


def feasible_huber(
    energies: torch.Tensor,
    regret_rel: torch.Tensor,
    delta: float = 0.01,
) -> torch.Tensor:
    """
    Calibrate feasible samples' energies to their regret_rel (scalar per mask).
    - regret_rel should be [N] with NaN for infeasible or missing.
    - Applies Huber on feasible subset only.

    Returns:
        Scalar loss (0 if no feasible entries are provided).
    """
    if energies.shape != regret_rel.shape:
        raise ValueError("energies and regret_rel must have same shape [N].")

    mask = torch.isfinite(regret_rel)  # True = has a target (feasible)
    if mask.sum() == 0:
        return energies.new_tensor(0.0)

    return F.huber_loss(energies[mask], regret_rel[mask], delta=delta)


def infeasible_huber(
    energies: torch.Tensor,
    infeasible_target: torch.Tensor,
    feasible_mask: torch.Tensor,
    delta: float = 0.1,
) -> torch.Tensor:
    """
    Calibrate infeasible samples' energies to a target (e.g., lambda * violation).
    - infeasible_target should be [N] with values for infeasible entries.
    - Uses feasible_mask to select infeasible subset.
    """
    if energies.shape != infeasible_target.shape:
        raise ValueError("energies and infeasible_target must have same shape [N].")
    if feasible_mask.shape != energies.shape or feasible_mask.dtype != torch.bool:
        raise ValueError("feasible_mask must be BoolTensor [N].")

    mask = ~feasible_mask
    if mask.sum() == 0:
        return energies.new_tensor(0.0)

    return F.huber_loss(energies[mask], infeasible_target[mask], delta=delta)


def infeasible_margin(
    energies: torch.Tensor,
    infeasible_target: torch.Tensor,
    feasible_mask: torch.Tensor,
    anchor: str = "best",
) -> torch.Tensor:
    """
    Enforce infeasible energies to sit above a feasible-energy anchor by a
    margin specified by infeasible_target.

    anchor='best' uses the lowest feasible energy. anchor='worst' uses the
    highest feasible energy, enforcing infeasible candidates above all feasible
    candidates in the current candidate pool.
    """
    if energies.shape != infeasible_target.shape:
        raise ValueError("energies and infeasible_target must have same shape [N].")
    if feasible_mask.shape != energies.shape or feasible_mask.dtype != torch.bool:
        raise ValueError("feasible_mask must be BoolTensor [N].")

    feasible_idx = feasible_mask
    infeasible_idx = (~feasible_mask) & torch.isfinite(infeasible_target)
    if feasible_idx.sum() == 0 or infeasible_idx.sum() == 0:
        return energies.new_tensor(0.0)

    anchor_key = str(anchor).strip().lower()
    feasible_energies = energies[feasible_idx]
    if anchor_key in {"best", "min", "minimum"}:
        feasible_anchor = feasible_energies.min().detach()
    elif anchor_key in {"worst", "max", "maximum", "all", "all_feasible"}:
        feasible_anchor = feasible_energies.max().detach()
    else:
        raise ValueError(
            f"Unsupported infeasible margin anchor '{anchor}'. "
            "Expected 'best' or 'worst'."
        )
    margins = infeasible_target[infeasible_idx]
    infeasible_energies = energies[infeasible_idx]
    return torch.relu(feasible_anchor + margins - infeasible_energies).mean()



