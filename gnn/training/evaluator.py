# gnn/training/evaluator.py
from typing import Dict, List, Iterable
import torch
import math

def _finite_min(x: torch.Tensor) -> float:
    xf = x[torch.isfinite(x)]
    if xf.numel() == 0:
        return float("inf")
    return float(xf.min().item())

def eval_batch_metrics(
    energies: torch.Tensor,
    label_types: List[str],
    feasible_mask: torch.Tensor,
    regret_rel: torch.Tensor,
    topk: Iterable[int] | None = None,
    tie_epsilon: float = 0.0,
) -> Dict[str, float]:
    """
    Per-instance metrics (batch_size=1 listwise):
      - top1_acc: 1 if argmin energy is labeled 'positive'
      - mean_regret@1: regret of argmin; if infeasible/NaN, fallback to best feasible regret; else 1.0
      - auc_feas: simple rank-based AUC separating feasible vs infeasible using -energies as score
      - topk hit + regret@k if topk is provided
    """
    metrics: Dict[str, float] = {}
    feas_idx_all = torch.where(feasible_mask)[0]

    argmin = int(torch.argmin(energies).item())
    is_pos = (label_types[argmin] == "positive")
    metrics["top1_acc"] = 1.0 if is_pos else 0.0

    r1 = float(regret_rel[argmin].item())
    if not math.isfinite(r1):
        r1 = 1.0
    metrics["mean_regret@1"] = r1

    eps_top1 = 0.0
    if tie_epsilon and tie_epsilon > 0 and feas_idx_all.numel() > 0:
        best_feas = _finite_min(regret_rel[feas_idx_all])
        if math.isfinite(best_feas):
            eps_top1 = 1.0 if r1 <= best_feas + tie_epsilon else 0.0
    metrics["eps_top1_acc"] = eps_top1

    feas = feasible_mask.bool().cpu()
    if feas.any() and (~feas).any():
        scores = (-energies).detach().cpu().float()
        pos_scores = scores[feas]
        neg_scores = scores[~feas]
        diff = pos_scores.view(-1, 1) - neg_scores.view(1, -1)
        auc = (diff > 0).float().mean().item() + 0.5 * (diff == 0).float().mean().item()
    else:
        auc = float("nan")
    metrics["auc_feas"] = auc

    if topk:
        order = torch.argsort(energies)
        for k in topk:
            k_eff = min(int(k), energies.numel())
            if k_eff <= 0:
                continue
            top_idx = order[:k_eff]
            metrics[f"topk@{k}"] = 1.0 if any(label_types[int(i)] == "positive" for i in top_idx.tolist()) else 0.0
            top_regrets = regret_rel[top_idx]
            finite = top_regrets[torch.isfinite(top_regrets)]
            if finite.numel():
                metrics[f"mean_regret@{k}"] = float(finite.min().item())
            else:
                metrics[f"mean_regret@{k}"] = float("nan")

    return metrics

def reduce_epoch_metrics(agg: Dict[str, float], topk: Iterable[int] | None = None) -> str:
    cnt = max(int(agg.get("count", 1)), 1)
    parts = []
    top1 = agg["top1_acc"] / cnt
    parts.append(f"top1={top1:.3f}")
    eps_top1 = agg.get("eps_top1_acc", 0.0) / cnt
    parts.append(f"eps_top1={eps_top1:.3f}")
    reg1 = agg["mean_regret@1"] / cnt
    parts.append(f"regret@1={reg1:.3f}")
    if topk:
        for k in topk:
            hit = agg.get(f"topk@{k}", 0.0) / cnt
            parts.append(f"top{k}={hit:.3f}")
            rk_sum = agg.get(f"regret_k_sum@{k}", 0.0)
            rk_count = agg.get(f"regret_k_count@{k}", 0)
            if rk_count:
                rk = rk_sum / rk_count
            else:
                rk = float("nan")
            parts.append(f"regret@{k}={(rk if rk == rk else float('nan')):.3f}")
    auc_den = int(agg.get("auc_count", 0))
    auc = agg["auc_feas"] / auc_den if (auc_den > 0) else float("nan")
    parts.append(f"auc_feas={(auc if auc == auc else float('nan')):.3f}")
    return "[val] " + " ".join(parts)

