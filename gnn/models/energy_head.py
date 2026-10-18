# gnn/models/energy_head.py
import torch.nn as nn


class EnergyHead(nn.Module):
    variant = "mlp64"

    def __init__(
        self,
        in_dim: int,
        hidden: int = 64,
        dropout: float = 0.1,
        variant: str = "mlp64",
    ):
        super().__init__()
        self.variant = str(variant)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.mlp(x).squeeze(-1)  # [N]


class ShallowEnergyHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 32, dropout: float = 0.1, variant: str = "mlp32"):
        super().__init__()
        self.variant = str(variant)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.mlp(x).squeeze(-1)  # [N]


class LinearEnergyHead(nn.Module):
    variant = "linear"

    def __init__(self, in_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)  # [N]
