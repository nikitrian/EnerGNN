# gnn/models/model.py
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import BipartiteGNNEncoder
from .readout import (
    CADirectYReadout,
    CAPooledOnlyReadout,
    CASumOnlyPlusReadout,
    QKPDirectYReadout,
    QKPReadout,
    QKPSizeAwareReadout,
    SetPoolReadout,
)
from .energy_head import EnergyHead, LinearEnergyHead, ShallowEnergyHead


MODEL_DISPLAY_NAMES = {
    "linear_mip_bce": "Linear+MIP(BCE)",
    "linear_mip_contrastive": "Linear+MIP(contrastive)",
}


def model_display_name(model_type: str) -> str:
    key = str(model_type).strip().lower()
    return MODEL_DISPLAY_NAMES.get(key, str(model_type))


def resolve_ca_readout_variant(problem: str, ca_readout_variant: Optional[str] = None) -> str:
    raw_variant = "auto" if ca_readout_variant is None else str(ca_readout_variant).strip().lower()
    aliases = {
        "auto": "auto",
        "pooled_only": "pooled_only",
        "sum_only_plus": "sum_only_plus",
        "direct_y": "direct_y",
    }
    if raw_variant not in aliases:
        raise ValueError(
            f"Unsupported ca_readout_variant '{ca_readout_variant}'. "
            "Expected 'auto', 'pooled_only', 'sum_only_plus', or 'direct_y'."
        )
    resolved = aliases[raw_variant]
    if resolved == "auto":
        return "sum_only_plus"
    return resolved


def resolve_ca_head_variant(problem: str, ca_head_variant: Optional[str] = None) -> str:
    raw_variant = "auto" if ca_head_variant is None else str(ca_head_variant).strip().lower()
    aliases = {
        "auto": "auto",
        "mlp16": "mlp16",
        "mlp16_deep": "mlp16_deep",
        "mlp32": "mlp32",
        "mlp32_deep": "mlp32_deep",
        "mlp64": "mlp64",
        "mlp64_shallow": "mlp64_shallow",
        "linear": "linear",
    }
    if raw_variant not in aliases:
        raise ValueError(
            f"Unsupported ca_head_variant '{ca_head_variant}'. "
            "Expected 'auto', 'mlp16', 'mlp16_deep', 'mlp32', 'mlp32_deep', 'mlp64', 'mlp64_shallow', or 'linear'."
        )
    resolved = aliases[raw_variant]
    if resolved == "auto":
        return "mlp32"
    return resolved


def resolve_qkp_readout_variant(problem: str, qkp_readout_variant: Optional[str] = None) -> str:
    problem_key = str(problem).strip().lower()
    raw_variant = "auto" if qkp_readout_variant is None else str(qkp_readout_variant).strip().lower()
    aliases = {
        "auto": "auto",
        "sum_only": "sum_only",
        "size_aware": "size_aware",
        "direct_y": "direct_y",
    }
    if raw_variant not in aliases:
        raise ValueError(
            f"Unsupported qkp_readout_variant '{qkp_readout_variant}'. "
            "Expected 'auto', 'sum_only', 'size_aware', or 'direct_y'."
        )
    resolved = aliases[raw_variant]
    if resolved == "auto":
        return "size_aware" if problem_key == "qkp" else "sum_only"
    return resolved


class GraphEBM(nn.Module):
    """
    Full model:
      - Encoder: graph -> node/global embeddings
      - Readout: (embeddings, masks) -> per-mask features
      - Head: features -> scalar energy
    """
    def __init__(self, dc: int, dm: int, de: int,
                 dh: int = 128, nlayers: int = 2, dropout: float = 0.1,
                 problem: str = "cart", head_hidden: int = 64,
                 ca_readout_variant: Optional[str] = "auto",
                 ca_direct_y_dim: Optional[int] = None,
                 qkp_readout_variant: Optional[str] = "auto",
                 qkp_direct_y_dim: Optional[int] = None,
                 ca_head_variant: Optional[str] = "auto"):
        super().__init__()
        self.problem = str(problem)
        self.ca_readout_variant = resolve_ca_readout_variant(self.problem, ca_readout_variant)
        self.ca_direct_y_dim = None if ca_direct_y_dim is None else int(ca_direct_y_dim)
        self.qkp_readout_variant = resolve_qkp_readout_variant(self.problem, qkp_readout_variant)
        self.qkp_direct_y_dim = None if qkp_direct_y_dim is None else int(qkp_direct_y_dim)
        self.ca_head_variant = resolve_ca_head_variant(self.problem, ca_head_variant)
        self.encoder = BipartiteGNNEncoder(dc, dm, de, dh=dh, nlayers=nlayers, dropout=dropout)
        if self.problem == "ca":
            if self.ca_readout_variant == "sum_only_plus":
                self.readout = CASumOnlyPlusReadout(dh)
            elif self.ca_readout_variant == "pooled_only":
                self.readout = CAPooledOnlyReadout(dh)
            elif self.ca_readout_variant == "direct_y":
                if self.ca_direct_y_dim is None:
                    raise ValueError(
                        "ca_direct_y_dim is required when "
                        "ca_readout_variant='direct_y'."
                    )
                self.readout = CADirectYReadout(dh, self.ca_direct_y_dim)
            else:
                raise AssertionError(f"Unhandled CA readout: {self.ca_readout_variant}")
        elif self.problem == "qkp":
            if self.qkp_readout_variant == "size_aware":
                self.readout = QKPSizeAwareReadout(dh)
            elif self.qkp_readout_variant == "direct_y":
                if self.qkp_direct_y_dim is None:
                    raise ValueError(
                        "qkp_direct_y_dim is required when "
                        "qkp_readout_variant='direct_y'."
                    )
                self.readout = QKPDirectYReadout(dh, self.qkp_direct_y_dim)
            else:
                self.readout = QKPReadout(dh)
        else:
            self.readout = SetPoolReadout(dh)

        if self.ca_head_variant == "linear":
            self.head = LinearEnergyHead(self.readout.out_dim)
        elif self.ca_head_variant == "mlp16":
            self.head = ShallowEnergyHead(self.readout.out_dim, hidden=16, dropout=dropout, variant="mlp16")
        elif self.ca_head_variant == "mlp16_deep":
            self.head = EnergyHead(
                self.readout.out_dim,
                hidden=16,
                dropout=dropout,
                variant="mlp16_deep",
            )
        elif self.ca_head_variant == "mlp32":
            self.head = ShallowEnergyHead(self.readout.out_dim, hidden=32, dropout=dropout, variant="mlp32")
        elif self.ca_head_variant == "mlp32_deep":
            self.head = EnergyHead(
                self.readout.out_dim,
                hidden=32,
                dropout=dropout,
                variant="mlp32_deep",
            )
        elif self.ca_head_variant == "mlp64_shallow":
            self.head = ShallowEnergyHead(self.readout.out_dim, hidden=64, dropout=dropout, variant="mlp64_shallow")
        else:
            self.head = EnergyHead(
                self.readout.out_dim,
                hidden=head_hidden,
                dropout=dropout,
                variant="mlp64",
            )

    def forward(self, graph: Dict[str, Any], masks: torch.Tensor):
        """
        graph: dict with tensors from graph_builder
        masks: [N, |M|] in {0,1}
        Returns: energies [N]
        """
        enc_out = self.encoder(graph)
        if self.problem in {"ca", "qkp"}:
            feats = self.readout(enc_out, masks, graph=graph)
        else:
            feats = self.readout(enc_out, masks, graph["edge_index"], graph["nC"])
        energy = self.head(feats)
        return energy


class LinearMIPBCEBaseline(nn.Module):
    """
    Linear+MIP(BCE): encoder-sharing baseline with independent variable scores.

    The variable logits are
        s_i = Linear(h_i)
    where h_i is the encoder's variable/facility embedding. For compatibility
    with the existing lower-is-better energy code, forward(graph, masks) returns
        E(y) = -sum_i s_i y_i.
    Exact MIP inference can therefore minimize E(y), which is equivalent to
    maximizing the linear score sum under the original hard constraints.
    """

    model_type = "linear_mip_bce"
    display_name = "Linear+MIP(BCE)"

    def __init__(
        self,
        dc: int,
        dm: int,
        de: int,
        dh: int = 128,
        nlayers: int = 2,
        dropout: float = 0.1,
        problem: str = "cart",
        **_: Any,
    ):
        super().__init__()
        self.problem = str(problem)
        self.encoder = BipartiteGNNEncoder(dc, dm, de, dh=dh, nlayers=nlayers, dropout=dropout)
        self.score_head = nn.Linear(dh, 1)

    def score_variables(self, graph: Dict[str, Any]) -> torch.Tensor:
        enc_out = self.encoder(graph)
        return self.score_head(enc_out["hm"]).squeeze(-1)

    def forward(self, graph: Dict[str, Any], masks: torch.Tensor) -> torch.Tensor:
        scores = self.score_variables(graph)
        masks_f = masks.to(device=scores.device, dtype=scores.dtype)
        if masks_f.dim() == 1:
            masks_f = masks_f.unsqueeze(0)
        if masks_f.shape[-1] != scores.numel():
            raise RuntimeError(
                f"Mask width {masks_f.shape[-1]} does not match variable scores {scores.numel()}."
            )
        return -(masks_f @ scores)


class LinearMIPContrastiveBaseline(LinearMIPBCEBaseline):
    """
    Linear+MIP(contrastive): additive solution-level energy baseline.

    Shares the encoder and additive score parameterization with the BCE-trained
    baseline, but uses the solution-level listwise contrastive loss.
    """

    model_type = "linear_mip_contrastive"
    display_name = "Linear+MIP(contrastive)"


class GNNFixingMIPBaseline(LinearMIPBCEBaseline):
    """
    Neural Diving / predict-and-search baseline used for variable fixing.

    Shares the encoder with Linear+MIP(BCE) but uses a 2-layer MLP
    head (matching Nair et al. 2020) instead of a single linear layer.
    Inference fixes high-confidence variables and re-solves the original
    benchmark objective on the reduced MIP; stochastic K-dive sampling is
    supported at inference time via n_dives > 1.
    """

    model_type = "gnn_fixing_mip"

    def __init__(
        self,
        dc: int,
        dm: int,
        de: int,
        dh: int = 128,
        nlayers: int = 2,
        dropout: float = 0.1,
        problem: str = "cart",
        **_: Any,
    ):
        super().__init__(dc, dm, de, dh=dh, nlayers=nlayers, dropout=dropout, problem=problem)
        # Replace the single linear head with a 2-layer MLP (Nair et al. style)
        self.score_head = nn.Sequential(
            nn.Linear(dh, dh),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dh, 1),
        )


class PredictAndSearchBaseline(LinearMIPBCEBaseline):
    """
    Original predict-and-search baseline (deterministic, single-dive).

    Identical to Linear+MIP(BCE) in architecture (single Linear(dh,1)
    head) but trained with BCE loss and used at inference with hard variable
    fixing followed by solving the reduced original MIP.  K=1 deterministic
    dive: variables are fixed by confidence threshold, not stochastic sampling.

    Distinct from GNNFixingMIPBaseline (Neural Diving) which uses a 2-layer
    MLP head and K=8 stochastic Bernoulli dives.
    """

    model_type = "predict_and_search"


# Compatibility imports for older code. New code should use the explicit names.
LinearScoreMIPBaseline = LinearMIPBCEBaseline
LinearMIPBaseline = LinearMIPContrastiveBaseline


class ConPaSSharedEncoder(nn.Module):
    """
    ConPaS-style (shared encoder) baseline.

    Display name: ConPaS-style (shared encoder)

    A controlled implementation of the ConPaS paradigm (Huang et al., ICML 2024)
    using the same GNN encoder, graph features, candidate pools and data splits
    as EnerGNN.  The encoder produces variable-node embeddings h_i, which are
    projected through a 3-layer MLP head to a per-variable logit:

        logit_i = Linear(64,1)(ReLU(Linear(64,64)(ReLU(Linear(dh,64)(h_i)))))

    Factorised configuration score (see gnn/losses/conpas.py):
        s(y) = sum_i [ y_i log sigma(logit_i) + (1-y_i) log sigma(-logit_i) ]

    Training uses the ConPaS InfoNCE contrastive loss.
    Inference restricts the original MIP via a Hamming-distance (local branching)
    constraint around the predicted assignment, keeping the original objective and
    all original constraints unchanged.  The neural network is NOT embedded in the
    MIP.

    NOTE: This is not an exact architectural reproduction of the published ConPaS
    model. It deliberately reuses the EnerGNN encoder for a controlled comparison.
    """

    model_type = "conpas_shared_encoder"

    def __init__(
        self,
        dc: int,
        dm: int,
        de: int,
        dh: int = 128,
        nlayers: int = 2,
        dropout: float = 0.1,
        problem: str = "cart",
        var_head_hidden_dims: Tuple[int, int] = (64, 64),
        **_: Any,
    ):
        super().__init__()
        self.problem = str(problem)
        self.encoder = BipartiteGNNEncoder(dc, dm, de, dh=dh, nlayers=nlayers, dropout=dropout)
        h1, h2 = int(var_head_hidden_dims[0]), int(var_head_hidden_dims[1])
        # Per-variable prediction head matching the spec: dh->64->64->1
        self.var_head = nn.Sequential(
            nn.Linear(dh, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 1),
        )

    def predict_variables(self, graph: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (logits [n_vars], probs [n_vars])."""
        enc_out = self.encoder(graph)
        logits  = self.var_head(enc_out["hm"]).squeeze(-1)   # [n_vars]
        probs   = torch.sigmoid(logits)
        return logits, probs

    def score_variables(self, graph: Dict[str, Any]) -> torch.Tensor:
        """Expose logits for compatibility with shared inference helpers."""
        logits, _ = self.predict_variables(graph)
        return logits

    def forward(self, graph: Dict[str, Any], masks: torch.Tensor) -> torch.Tensor:
        """
        Return the negated ConPaS factorised score for each configuration.

        The score s(y) = sum_i[y_i log p_i + (1-y_i) log(1-p_i)] is maximised
        for high-quality configurations.  Returning -s(y) gives a lower-is-better
        energy compatible with the EnerGNN evaluation infrastructure.
        """
        logits, _ = self.predict_variables(graph)
        log_p    = F.logsigmoid(logits)    # [n_vars]
        log_1mp  = F.logsigmoid(-logits)   # [n_vars]
        masks_f  = masks.to(dtype=logits.dtype, device=logits.device)
        scores   = masks_f @ log_p + (1.0 - masks_f) @ log_1mp   # [N]
        return -scores   # lower energy = higher log-prob = better
