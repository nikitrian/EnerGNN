import math

from pyomo.environ import SolverFactory, value
from pyomo.opt import TerminationCondition


def solve_instance(instance, timelimit=None, mipgap=None, seed=None, gurobi_opts=None, tee=True):
    opt = SolverFactory('gurobi')
    opts = dict(OutputFlag=1 if tee else 0)
    if timelimit is not None: opts['TimeLimit'] = float(timelimit)
    if mipgap is not None:    opts['MIPGap'] = float(mipgap)
    if seed is not None:      opts['Seed'] = int(seed)
    if gurobi_opts:           opts.update(gurobi_opts)

    res = opt.solve(instance, options=opts, tee=tee)
    term = res.solver.termination_condition

    accepted_term = term in (
        TerminationCondition.optimal,
        TerminationCondition.feasible,
        TerminationCondition.globallyOptimal,
        TerminationCondition.locallyOptimal,
    )
    objective_value = value(instance.obj, exception=False)
    has_incumbent = objective_value is not None and all(
        instance.E1[m].value is not None for m in instance.m
    )
    feasible = bool((accepted_term or has_incumbent) and objective_value is not None)
    obj = float(objective_value) if feasible else float('inf')
    if feasible and not math.isfinite(obj):
        feasible = False
        obj = float('inf')

    # best bound (best effort)
    try:
        bound = float(res.solver.best_bound)
    except Exception:
        try:
            bound = float(opt._solver_model.ObjBound)
        except Exception:
            bound = obj

    gap_abs = abs(obj - bound) if (obj != float('inf')) else float('inf')
    gap_rel = (gap_abs / (abs(obj) + 1e-9)) if (obj not in (0.0, float('inf'))) else float('inf')

    # Safely extract E1 even if infeasible (may be None)
    e1 = {}
    for m in instance.m:
        v = instance.E1[m].value  # NOT value(...): returns None safely
        e1[str(m)] = (int(round(v)) if v is not None else None)

    try:
        solve_time = float(res.solver.time)
    except Exception:
        solve_time = None

    return dict(
        feasible=feasible,
        obj=obj,
        bound=float(bound),
        gap_abs=float(gap_abs),
        gap_rel=float(gap_rel),
        e1=e1,
        status=str(term),
        solve_time=solve_time,
    )
