"""Shared reduced-model solving and candidate labels for detailed CART."""

import math
import time


class IncompleteCandidatePool(RuntimeError):
    """A candidate has neither a certified cost nor proven infeasibility."""


COMPLETION_GAP_TOL = 1e-9


def resolve_formulation(formulation="detailed", turnaround_limit=17):
    if not math.isfinite(float(turnaround_limit)) or float(turnaround_limit) <= 0:
        raise ValueError("Turnaround limit must be finite and positive.")
    if formulation not in ("auto", "detailed"):
        raise ValueError("CART requires the detailed Gurobi formulation.")
    return "detailed"


def is_complete(record, mip_gap):
    if record.get("status") == "infeasible" and record.get("feasible") is False:
        return True
    gap = record.get("mip_gap_rel")
    return bool(
        record.get("proven_optimal") and record.get("feasible") is True
        and gap is not None and float(gap) <= float(mip_gap) + COMPLETION_GAP_TOL
    )


def _finite_attr(model, name):
    try:
        result = float(getattr(model, name))
        return result if math.isfinite(result) else None
    except Exception:
        return None


def solve_mask(data, mask, *, time_horizon, max_open_facilities=None,
               time_limit=3600, threads=1, mip_gap=1e-4, tee=False,
               formulation="detailed", turnaround_limit=17.0,
               min_open_facilities=None, forced_e1_values=None,
               conpas_x0_facilities=None, conpas_x1_facilities=None, conpas_delta=None):
    """Build the reduced direct-gurobi model and retain solver termination data."""
    resolve_formulation(formulation, turnaround_limit)
    if mask is not None and (not any(mask) or
                            (max_open_facilities is not None and sum(mask) > max_open_facilities)):
        return {
            "status": "infeasible", "feasible": False, "proven_optimal": False,
            "objective": None, "best_bound": None, "mip_gap_rel": None,
            "build_time_sec": 0.0, "solver_time_sec": 0.0,
            "source": "empty_mask" if not any(mask) else "cardinality_bound",
        }
    from gurobipy import GRB

    from .model_cart_detailed_gurobi import (
        build_detailed_gurobi_model,
        detailed_data_from_mapping,
        restrict_detailed_data,
    )

    time_limit_value = None
    if time_limit is not None and math.isfinite(float(time_limit)):
        time_limit_value = float(time_limit)

    model = None
    env = None
    started = time.perf_counter()
    cart_data = detailed_data_from_mapping(data, time_horizon=int(time_horizon))
    facilities = list(cart_data.m)
    bits = None if mask is None else list(mask)
    if bits is not None and (len(bits) != len(facilities) or any(bit not in (0, 1) for bit in bits)):
        raise ValueError("Mask must contain one binary value per original facility.")
    # Scenario solves retain all facilities so partial fixes cannot be lost by reduction.
    full_scenario = (mask is None or bool(forced_e1_values) or bool(min_open_facilities)
                     or conpas_delta is not None)
    if full_scenario:
        reduced_data = cart_data
    else:
        selected = [facility for facility, bit in zip(facilities, bits) if bit]
        reduced_data = restrict_detailed_data(cart_data, selected)
    model, artifacts = build_detailed_gurobi_model(
        reduced_data,
        max_open_facilities=max_open_facilities,
        solver_threads=threads,
        time_limit=time_limit_value,
        mip_gap=mip_gap,
        solver_options={"MIPGapAbs": 0.0},
        fix_e1=not full_scenario,
        fixed_e1_values=dict(zip(facilities, bits)) if full_scenario and bits is not None else None,
        min_open_facilities=min_open_facilities,
        forced_e1_values=forced_e1_values,
        conpas_x0_facilities=conpas_x0_facilities,
        conpas_x1_facilities=conpas_x1_facilities,
        conpas_delta=conpas_delta,
        tee=tee,
        turnaround_limit=turnaround_limit,
    )
    build_time = time.perf_counter() - started
    env = artifacts.env
    try:
        solve_start = time.perf_counter()
        model.optimize()
        has_solution = int(model.SolCount) > 0
        status = int(model.Status)
        return {
            "status": (
                "optimal" if status == GRB.OPTIMAL
                else "infeasible" if status == GRB.INFEASIBLE
                else "time_limit" if status == GRB.TIME_LIMIT
                else f"status_{status}"
            ),
            "gurobi_status_code": status,
            "feasible": True if has_solution else False if status == GRB.INFEASIBLE else None,
            "proven_optimal": bool(status == GRB.OPTIMAL and has_solution),
            "objective": float(model.ObjVal) if has_solution else None,
            "best_bound": _finite_attr(model, "ObjBound"),
            "mip_gap_rel": _finite_attr(model, "MIPGap") if has_solution else None,
            "build_time_sec": build_time,
            "solve_wall_time_sec": time.perf_counter() - solve_start,
            "solver_time_sec": float(model.Runtime),
            "model_num_variables": int(model.NumVars),
            "model_num_constraints": int(model.NumConstrs),
            "model_num_binary_variables": int(model.NumBinVars),
            "source": "reduced_detailed_milp",
            "formulation": "detailed", "turnaround_limit": float(turnaround_limit),
            "opened_facilities": (
                [facility for facility in facilities
                 if facility in artifacts.E1 and artifacts.E1[facility].X > 0.5]
                if has_solution else []
            ),
        }
    finally:
        if model is not None:
            model.dispose()
        if env is not None:
            env.dispose()


def label_candidate_results(results, *, mip_gap=1e-4, tie_tolerance=1e-6):
    """Label only completed solves; never turn unknown masks into negatives."""
    for result in results:
        if not is_complete(result, mip_gap):
            raise IncompleteCandidatePool(
                "Candidate solve is incomplete; increase the solve limit or use "
                "dataset_maker_masks on prepared patient inputs before exporting labels."
            )
        if result["feasible"] and not math.isfinite(result["objective"]):
            raise ValueError("A completed feasible candidate must have a finite cost.")
    objectives = [result["objective"] for result in results if result["feasible"]]
    best = min(objectives) if objectives else None
    rows = []
    for result in results:
        feasible = result["feasible"]
        gap = max(0.0, result["objective"] - best) if feasible else None
        positive = feasible and gap <= tie_tolerance * max(abs(best), 1.0)
        rows.append(dict(
            result,
            label_type="positive" if positive else "ham_neg_feas" if feasible else "ham_neg_infeas",
            optimal=bool(positive),
            regret_abs=gap,
            regret_rel=gap / max(abs(best), 1e-9) if feasible else None,
        ))
    return rows
