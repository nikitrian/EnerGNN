# gnn/data/schema.py
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

@dataclass
class InstanceMeta:
    instance_id: int
    C: List[str]
    M: List[str]
    CAPMAX: Dict[str, float]               # maximum parallel lines per facility
    U1: Dict[str, Dict[str, float]]        # U1[c][m]
    CIM: Dict[str, float]            
    CVM: Dict[str, float] 
    INC: Dict[str, Dict[str, int]]         # INC[c][t_str]
    T: int                                 # horizon inferred from INC
    globals: Dict[str, Any]                # e.g., TMFE, total_therapies, demand_kwargs

@dataclass
class SampleRow:
    instance_id: int
    label_type: str                        # 'positive' | 'ham_neg_feas' | 'ham_neg_infeas' | ...
    feasible: bool
    objective: Optional[float]             # None or float('inf') if infeasible
    status: str
    e1: Dict[str, int]                     # {'m1':0/1, ...}
    hamming_r: Optional[int]
    regret_rel: Optional[float]            # None for infeasible
    regret_abs: Optional[float]
    seed_demand: int
    seed_solver: Optional[int]
    extras: Dict[str, Any]                 # passthrough field for demand kwargs etc.
