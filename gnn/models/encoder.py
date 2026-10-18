# gnn/models/encoder.py
from typing import Dict, Any
import torch
import torch.nn as nn
from .layers import EdgeMessageLayer

class BipartiteGNNEncoder(nn.Module):
    """
    Encodes one instance graph (C↔M with edge features) into node embeddings.
    Expects a dict container with:
      "Xc": [|C|, dc], "Xm": [|M|, dm],
      "edge_index": [2, |E|] (c->(nC+m)), "edge_attr": [|E|, de]
    """
    def __init__(self, dc: int, dm: int, de: int, dh: int = 128, nlayers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.proj_c = nn.Linear(dc, dh)
        self.proj_m = nn.Linear(dm, dh)

        layers = []
        for _ in range(nlayers):
            layers += [
                EdgeMessageLayer(dh, de, hidden=dh, dropout=dropout, direction="c_to_m"),
                EdgeMessageLayer(dh, de, hidden=dh, dropout=dropout, direction="m_to_c"),
            ]
        self.layers = nn.ModuleList(layers)

    def forward(self, graph: Dict[str, Any]):
        Xc = graph["Xc"]          # [|C|, dc]
        Xm = graph["Xm"]          # [|M|, dm]
        E  = graph["edge_index"]  # [2, |E|]
        Ea = graph["edge_attr"]   # [|E|, de]

        hc = self.proj_c(Xc)      # [|C|, dh]
        hm = self.proj_m(Xm)      # [|M|, dh]

        for layer in self.layers:
            hc, hm = layer(hc, hm, E, Ea)

        # Global context: mean over both types, then fuse
        hg = torch.cat([hc.mean(dim=0), hm.mean(dim=0)], dim=-1)   # [2*dh]
        return {"hc": hc, "hm": hm, "hg": hg, "Xc_raw": Xc, "Xm_raw": Xm, "edge_index": E, "edge_attr": Ea}
