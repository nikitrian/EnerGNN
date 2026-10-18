# gnn/losses/conpas.py
"""
ConPaS contrastive loss for factorised per-variable predictions.

Reference:
    Huang et al., "Contrastive Predict-and-Search for Mixed Integer Linear Programs",
    ICML 2024.

Factorised configuration score
-------------------------------
Given per-variable logits f = (f_1, ..., f_n) and a binary configuration
y = (y_1, ..., y_n) in {0,1}^n, the ConPaS score is the log-probability of y
under the independent Bernoulli model q(y | f):

    s(y | f) = sum_i [ y_i * log sigma(f_i) + (1 - y_i) * log sigma(-f_i) ]
             = sum_i [ y_i * log_p_i + (1 - y_i) * log(1 - p_i) ]

where p_i = sigma(f_i).  A higher (less negative) score means the model
assigns higher probability to configuration y.

Contrastive loss (InfoNCE)
--------------------------
For an instance with positive set P+ and negative set P-:

    L = -mean_{y+ in P+} [ s(y+)/tau ]
        + log sum_{y in P+ union P-} [ exp(s(y)/tau) ]

Minimising L encourages high-quality configurations (P+) to receive higher
scores than low-quality or infeasible ones (P-), scaled by temperature tau.
"""

from typing import Dict, Iterable, Optional, Tuple
import torch
import torch.nn.functional as F


def conpas_config_score(
    logits: torch.Tensor,   # [n_vars]
    masks: torch.Tensor,    # [N, n_vars]
) -> torch.Tensor:
    """
    Compute the ConPaS factorised log-probability score for N configurations.

    s(y | f) = sum_i [ y_i * log sigma(f_i) + (1-y_i) * log sigma(-f_i) ]

    Args:
        logits: per-variable logits, shape [n_vars]
        masks:  binary configuration matrix, shape [N, n_vars], values in {0,1}

    Returns:
        scores, shape [N]  (each value <= 0)
    """
    if logits.dim() != 1:
        raise ValueError(f"logits must be 1-D [n_vars], got shape {tuple(logits.shape)}")
    if masks.dim() != 2 or masks.shape[1] != logits.shape[0]:
        raise ValueError(
            f"masks must be [N, n_vars], got shape {tuple(masks.shape)} "
            f"vs n_vars={logits.shape[0]}"
        )

    log_p    = F.logsigmoid(logits)    # [n_vars]  log sigma(f_i)
    log_1mp  = F.logsigmoid(-logits)   # [n_vars]  log(1 - sigma(f_i))

    masks_f  = masks.to(dtype=logits.dtype, device=logits.device)   # [N, n_vars]
    # s(y) = y dot log_p + (1-y) dot log_1mp
    scores   = masks_f @ log_p + (1.0 - masks_f) @ log_1mp          # [N]
    return scores


def conpas_contrastive_loss(
    logits: torch.Tensor,                        # [n_vars]
    masks: torch.Tensor,                         # [N, n_vars]
    label_types: Iterable[str],                  # length-N label strings
    feasible_mask: torch.Tensor,                 # [N] bool
    regret_rel: Optional[torch.Tensor] = None,   # [N], NaN for infeasible/missing
    temperature: float = 0.1,
    positive_selection: str = "label",           # "label" | "regret_threshold"
    regret_pos_threshold: float = 0.05,
    include_infeasible_negatives: bool = True,
    include_suboptimal_negatives: bool = True,
    positives_per_instance: Optional[int] = None,
    negatives_per_instance: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict]:
    """
    ConPaS InfoNCE contrastive loss.

    Positive set P+
    ~~~~~~~~~~~~~~~
    positive_selection="label":
        Candidates whose label_type == "positive" (already identified as
        optimal or near-optimal in the dataset construction step).
    positive_selection="regret_threshold":
        Feasible candidates with regret_rel <= regret_pos_threshold.

    Negative set P-
    ~~~~~~~~~~~~~~~
    All candidates not in P+, subject to:
        include_infeasible_negatives  – whether to include infeasible candidates
        include_suboptimal_negatives  – whether to include suboptimal feasible ones

    Loss
    ~~~~
    L = -mean_{y+ in P+}[ s(y+)/tau ]
        + log sum_{y in P+ union P-}[ exp(s(y)/tau) ]

    Returns:
        (loss scalar, info dict)
    """
    labels_list   = list(label_types)
    N             = len(labels_list)
    if masks.shape[0] != N:
        raise ValueError(
            f"masks.shape[0]={masks.shape[0]} != len(label_types)={N}"
        )
    if feasible_mask.shape[0] != N:
        raise ValueError(
            f"feasible_mask.shape[0]={feasible_mask.shape[0]} != N={N}"
        )

    feas = feasible_mask.to(dtype=torch.bool, device=logits.device)

    # ---- build regret tensor (NaN where unavailable)
    if regret_rel is None:
        rr = logits.new_full((N,), float("nan"))
    else:
        rr = regret_rel.to(dtype=logits.dtype, device=logits.device)

    # ---- select positives
    if positive_selection == "regret_threshold":
        pos_mask = feas & torch.isfinite(rr) & (rr <= float(regret_pos_threshold))
    else:  # "label"
        pos_mask = torch.tensor(
            [lt == "positive" for lt in labels_list],
            dtype=torch.bool, device=logits.device,
        )

    if not pos_mask.any():
        zero = logits.new_zeros(())
        return zero, {"n_pos": 0, "n_neg": 0, "n_pool": 0,
                      "mean_pos_score": float("nan"), "mean_neg_score": float("nan")}

    # ---- select negatives
    neg_mask = ~pos_mask
    if not include_infeasible_negatives:
        neg_mask = neg_mask & feas
    if not include_suboptimal_negatives:
        # keep only infeasible negatives (drop suboptimal-feasible ones)
        neg_mask = neg_mask & (~feas)

    # ---- optional subsampling (deterministic by index order for reproducibility)
    pos_indices = pos_mask.nonzero(as_tuple=False).view(-1)
    neg_indices = neg_mask.nonzero(as_tuple=False).view(-1)

    if positives_per_instance is not None and positives_per_instance < pos_indices.shape[0]:
        pos_indices = pos_indices[:int(positives_per_instance)]
    if negatives_per_instance is not None and negatives_per_instance < neg_indices.shape[0]:
        neg_indices = neg_indices[:int(negatives_per_instance)]

    n_pos  = int(pos_indices.shape[0])
    n_neg  = int(neg_indices.shape[0])
    n_pool = n_pos + n_neg

    if n_pool == 0 or n_pos == 0:
        zero = logits.new_zeros(())
        return zero, {"n_pos": n_pos, "n_neg": n_neg, "n_pool": n_pool,
                      "mean_pos_score": float("nan"), "mean_neg_score": float("nan")}

    pool_indices = torch.cat([pos_indices, neg_indices], dim=0)  # [P]
    pool_masks   = masks[pool_indices]                            # [P, n_vars]

    # ---- ConPaS factorised scores for the pool
    scores = conpas_config_score(logits, pool_masks)   # [P]
    scaled = scores / float(temperature)               # [P]

    # InfoNCE: -mean(pos_scaled) + logsumexp(all_scaled)
    pos_scaled = scaled[:n_pos]
    loss = -pos_scaled.mean() + torch.logsumexp(scaled, dim=0)

    info = {
        "n_pos":           n_pos,
        "n_neg":           n_neg,
        "n_pool":          n_pool,
        "mean_pos_score":  float(scores[:n_pos].mean().detach().item()),
        "mean_neg_score":  float(scores[n_pos:].mean().detach().item()) if n_neg > 0 else float("nan"),
    }
    return loss, info
