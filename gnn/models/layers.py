# gnn/models/layers.py
from typing import Literal, Tuple
import torch
import torch.nn as nn

try:
    from torch_scatter import scatter_mean
except Exception:
    # Minimal fallback if torch_scatter isn't installed (slower on CPU/GPU)
    def scatter_mean(src, index, dim=0, dim_size=None):
        if dim != 0:
            raise NotImplementedError("Fallback only supports dim=0")
        if dim_size is None:
            dim_size = int(index.max()) + 1
        out = torch.zeros((dim_size,) + src.shape[1:], device=src.device, dtype=src.dtype)
        cnt = torch.zeros((dim_size, ), device=src.device, dtype=src.dtype)
        out.index_add_(0, index, src)
        cnt.index_add_(0, index, torch.ones_like(index, dtype=src.dtype))
        cnt = torch.clamp(cnt, min=1.0)
        return out / cnt.unsqueeze(-1)

class MLP(nn.Module):
    def __init__(self, dims: Tuple[int, ...], dropout: float = 0.0, act=nn.ReLU):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i+1])]
            if i < len(dims) - 2:
                layers += [act(), nn.Dropout(dropout)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class EdgeMessageLayer(nn.Module):
    """
    Edge-aware message passing on a bipartite graph with edges (c -> m).
    direction = 'c_to_m' or 'm_to_c'
    Inputs:
      hc: [|C|, dh], hm: [|M|, dh]
      edge_index: [2, |E|] where row0 = c_idx, row1 = m_idx_offset (nC + m_idx)
      edge_attr: [|E|, de] (e.g., U1 & RBFs)
    """
    def __init__(self, dh: int, de: int, hidden: int, dropout: float = 0.0,
                 direction: Literal["c_to_m","m_to_c"]="c_to_m"):
        super().__init__()
        self.direction = direction
        self.msg = MLP((dh + de, hidden), dropout)
        self.upd = MLP((dh + hidden, dh), dropout)
        self.norm = nn.LayerNorm(dh)

    def forward(self, hc, hm, edge_index, edge_attr):
        nC = hc.size(0)
        c_idx = edge_index[0]                       # [E]
        m_idx = edge_index[1] - nC                  # [E] (remove offset)
        if self.direction == "c_to_m":
            src = hc[c_idx]                         # [E, dh]
            dst_size = hm.size(0)
            dst_old = hm
        else:  # m_to_c
            src = hm[m_idx]
            dst_size = hc.size(0)
            dst_old = hc

        msg_in = torch.cat([src, edge_attr], dim=-1)      # [E, dh+de]
        m = self.msg(msg_in)                               # [E, hidden]

        # aggregate to destination nodes
        dst_idx = m_idx if self.direction == "c_to_m" else c_idx
        agg = scatter_mean(m, dst_idx, dim=0, dim_size=dst_size)  # [dst_size, hidden]

        upd_in = torch.cat([dst_old, agg], dim=-1)         # [dst_size, dh+hidden]
        dst_new = self.upd(upd_in)                          # [dst_size, dh]
        dst_new = self.norm(dst_old + dst_new)              # residual + norm

        if self.direction == "c_to_m":
            return hc, dst_new
        else:
            return dst_new, hm
