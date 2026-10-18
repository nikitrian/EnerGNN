from typing import Dict, Any
from pyomo.environ import value

def pack_instance_metadata(instance) -> Dict[str,Any]:
    # sets
    c = [str(cc) for cc in instance.c]
    m = [str(mm) for mm in instance.m]

    CAPMAX = {str(mm): float(value(instance.CAPMAX[mm])) for mm in instance.m}

    # 2-D U1 → nested dict: U1[c][m] = val
    U1 = {str(cc): {str(mm): float(value(instance.U1[cc, mm, "j1"])) for mm in instance.m}
          for cc in instance.c}

    return dict(c=c, m=m, CAPMAX=CAPMAX, U1=U1)

def pack_inc(instance) -> Dict[str, Dict[int, int]]:
    # INC[(c,t)] → INC[c][t]
    inc = {str(cc): {int(t): int(sum(value(instance.INC[p, cc, t]) for p in instance.p)) for t in instance.t}
           for cc in instance.c}
    return inc


def record_row(instance, solve_res, label_type, instance_id, seed_demand, seed_solver, hamming_r=None):
    return dict(
        instance_id=int(instance_id),
        label_type=label_type,      # 'positive' | 'ham_neg_feas' | 'ham_neg_infeas' | 'inst_infeas'
        feasible=bool(solve_res['feasible']),
        objective=float(solve_res['obj']),
        mip_gap_rel=float(solve_res['gap_rel']),
        gap_abs=float(solve_res['gap_abs']),
        status=solve_res['status'],
        e1=solve_res['e1'],         # dict m->0/1
        hamming_r=hamming_r,
        seed_demand=int(seed_demand),
        seed_solver=(None if seed_solver is None else int(seed_solver)),
    )
