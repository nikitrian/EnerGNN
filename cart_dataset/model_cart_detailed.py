"""
Detailed CART supply-chain model with patient-level flows.

you can import it (to get an AbstractModel) or execute it directly from the
CLI to instantiate a concrete Pyomo model from a .dat file.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from pyomo.environ import (
    AbstractModel,
    Binary,
    ConcreteModel,
    Constraint,
    NonNegativeIntegers,
    NonNegativeReals,
    Objective,
    Param,
    RangeSet,
    Set,
    Var,
    value,
)
from pyomo.dataportal import DataPortal


HOSPITALS = ("h1", "h2", "h3", "h4")
TRANSPORT_MODES = ("j1", "j2")
TT1_VALUES = {"j1": 1, "j2": 2}
TT3_VALUES = {"j1": 1, "j2": 2}

U1_J2_VALUES = {
    "c1": {"m1": 520, "m2": 520, "m3": 520, "m4": 936, "m5": 1053, "m6": 1053},
    "c2": {"m1": 520, "m2": 100, "m3": 520, "m4": 936, "m5": 1053, "m6": 1053},
    "c3": {"m1": 520, "m2": 520, "m3": 936, "m4": 936, "m5": 1053, "m6": 1053},
    "c4": {"m1": 520, "m2": 520, "m3": 936, "m4": 936, "m5": 1053, "m6": 1053},
}

U3_VALUES = {
    "j1": {
        ("m1", "h1"): 1910, ("m1", "h2"): 1910, ("m1", "h3"): 2910, ("m1", "h4"): 1910,
        ("m2", "h1"): 2910, ("m2", "h2"): 400, ("m2", "h3"): 1910, ("m2", "h4"): 1910,
        ("m3", "h1"): 2910, ("m3", "h2"): 2910, ("m3", "h3"): 2910, ("m3", "h4"): 2910,
        ("m4", "h1"): 2910, ("m4", "h2"): 2910, ("m4", "h3"): 2910, ("m4", "h4"): 2910,
        ("m5", "h1"): 2910, ("m5", "h2"): 2910, ("m5", "h3"): 2910, ("m5", "h4"): 2910,
        ("m6", "h1"): 2910, ("m6", "h2"): 2910, ("m6", "h3"): 2910, ("m6", "h4"): 2910,
    },
    "j2": {
        ("m1", "h1"): 910, ("m1", "h2"): 910, ("m1", "h3"): 910, ("m1", "h4"): 910,
        ("m2", "h1"): 910, ("m2", "h2"): 300, ("m2", "h3"): 910, ("m2", "h4"): 910,
        ("m3", "h1"): 1755, ("m3", "h2"): 1755, ("m3", "h3"): 1755, ("m3", "h4"): 1755,
        ("m4", "h1"): 1755, ("m4", "h2"): 1755, ("m4", "h3"): 1755, ("m4", "h4"): 1755,
        ("m5", "h1"): 1855, ("m5", "h2"): 1855, ("m5", "h3"): 1855, ("m5", "h4"): 1755,
        ("m6", "h1"): 1855, ("m6", "h2"): 1855, ("m6", "h3"): 1855, ("m6", "h4"): 1855,
    },
}


def load_static_network(dat_path: Path | str) -> Tuple[Dict[str, List[str]], Dict[str, Any]]:
    """Load the base static network used to construct CART16 facilities."""
    dat_path = Path(dat_path)
    dat_text = dat_path.read_text(encoding="utf-8")
    if re.search(r"^\s*param\s+CAPMAX\s*:=", dat_text, flags=re.MULTILINE):
        capacity_name = "CAPMAX"
    elif re.search(r"^\s*param\s+FCAP\s*:=", dat_text, flags=re.MULTILINE):
        capacity_name = "FCAP"
    else:
        raise ValueError(f"Missing CAPMAX parameter in {dat_path.as_posix()}.")

    schema = AbstractModel()
    schema.c = Set()
    schema.m = Set()
    schema.CIM = Param(schema.m)
    schema.CVM = Param(schema.m)
    if capacity_name == "CAPMAX":
        schema.CAPMAX = Param(schema.m)
    else:
        schema.FCAP = Param(schema.m)
    schema.U1 = Param(schema.c, schema.m)

    portal = DataPortal()
    portal.load(filename=dat_path.as_posix(), model=schema)
    instance = schema.create_instance(portal)
    capacity = getattr(instance, capacity_name)

    clinics = [str(c) for c in instance.c]
    facilities = [str(m) for m in instance.m]
    static_sets = {"c": clinics, "m": facilities}
    static_params: Dict[str, Any] = {
        "CIM": {str(m): float(value(instance.CIM[m])) for m in instance.m},
        "CVM": {str(m): float(value(instance.CVM[m])) for m in instance.m},
        "CAPMAX": {str(m): float(value(capacity[m])) for m in instance.m},
        "U1": {
            (str(c), str(m)): float(value(instance.U1[c, m]))
            for c in instance.c
            for m in instance.m
        },
    }
    return static_sets, static_params


def _facility_index(name: str) -> Optional[int]:
    text = str(name).strip()
    if not text.startswith("m") or not text[1:].isdigit():
        return None
    return int(text[1:])


def _parent_base_facility(facility: str) -> str:
    index = _facility_index(facility)
    if index is None:
        return facility if facility in {f"m{i}" for i in range(1, 7)} else "m1"
    return f"m{((index - 1) % 6) + 1}"


def _transport_scale(
    static_params: Mapping[str, Any],
    clinics: Sequence[str],
    facility: str,
    parent: str,
) -> float:
    ratios = []
    for clinic in clinics:
        parent_cost = float(static_params["U1"].get((clinic, parent), 0.0))
        facility_cost = float(static_params["U1"].get((clinic, facility), 0.0))
        if parent_cost > 0.0 and facility_cost > 0.0:
            ratios.append(facility_cost / parent_cost)
    if not ratios:
        return 1.0
    return max(0.5, min(1.8, sum(ratios) / len(ratios)))


def _u1_j2_value(
    static_params: Mapping[str, Any],
    clinic: str,
    facility: str,
) -> float:
    if facility in U1_J2_VALUES.get(clinic, {}):
        return float(U1_J2_VALUES[clinic][facility])
    parent = _parent_base_facility(facility)
    parent_j2 = float(U1_J2_VALUES[clinic][parent])
    parent_j1 = float(static_params["U1"].get((clinic, parent), 0.0))
    facility_j1 = float(static_params["U1"].get((clinic, facility), 0.0))
    scale = 1.0 if parent_j1 <= 0.0 else facility_j1 / parent_j1
    return max(50.0, min(3500.0, parent_j2 * scale))


def _u3_value(
    static_params: Mapping[str, Any],
    clinics: Sequence[str],
    facility: str,
    hospital: str,
    mode: str,
) -> float:
    if (facility, hospital) in U3_VALUES[mode]:
        return float(U3_VALUES[mode][(facility, hospital)])
    parent = _parent_base_facility(facility)
    scale = _transport_scale(static_params, clinics, facility, parent)
    return max(50.0, min(5000.0, U3_VALUES[mode][(parent, hospital)] * scale))


def build_detailed_parameters(
    static_sets: Mapping[str, Sequence[str]],
    static_params: Mapping[str, Any],
) -> Dict[str, Any]:
    """Expand the aggregate network parameters into the detailed CART schema."""
    clinics = [str(c) for c in static_sets["c"]]
    facilities = [str(m) for m in static_sets["m"]]
    required_clinics = {"c1", "c2", "c3", "c4"}
    missing = sorted(required_clinics.difference(clinics))
    if missing:
        raise ValueError(f"Detailed CART requires clinics c1-c4; missing {missing}.")

    detailed: Dict[str, Any] = {
        "h": list(HOSPITALS),
        "j": list(TRANSPORT_MODES),
        "TT1": dict(TT1_VALUES),
        "TT3": dict(TT3_VALUES),
        "U1": {},
        "U3": {},
        "CAPMAX": {m: float(static_params["CAPMAX"][m]) for m in facilities},
        "FMAX": 1.0,
        "FMIN": 0.0,
        "TAD": 0.0,
        "TLS": 1.0,
        "TMFE": float(static_params.get("TMFE", 7)),
    }
    for clinic in clinics:
        for facility in facilities:
            detailed["U1"][(clinic, facility, "j1")] = float(
                static_params["U1"][(clinic, facility)]
            )
            detailed["U1"][(clinic, facility, "j2")] = _u1_j2_value(
                static_params, clinic, facility
            )
    for facility in facilities:
        for hospital in HOSPITALS:
            for mode in TRANSPORT_MODES:
                detailed["U3"][(facility, hospital, mode)] = _u3_value(
                    static_params, clinics, facility, hospital, mode
                )
    return detailed


def expand_patient_arrivals(
    clinics: Sequence[str],
    time_horizon: int,
    inc_param: Mapping[Tuple[str, int], int],
) -> Tuple[List[str], Dict[Tuple[str, str, int], int]]:
    """Expand aggregate clinic/day demand into deterministic unit-patient arrivals."""
    patient_ids: List[str] = []
    patient_inc: Dict[Tuple[str, str, int], int] = {}
    patient_index = 1
    for clinic in clinics:
        for day in range(1, int(time_horizon) + 1):
            demand = int(inc_param.get((str(clinic), day), 0))
            if demand < 0:
                raise ValueError(f"Negative demand for ({clinic}, {day}): {demand}.")
            for _ in range(demand):
                patient = f"p{patient_index}"
                patient_ids.append(patient)
                patient_inc[(patient, str(clinic), day)] = 1
                patient_index += 1
    if not patient_ids:
        raise ValueError("Detailed CART requires at least one patient arrival.")
    return patient_ids, patient_inc


def build_detailed_data_from_aggregate(
    static_sets: Mapping[str, Sequence[str]],
    static_params: Mapping[str, Any],
    inc_param: Mapping[Tuple[str, int], int],
    *,
    time_horizon: int,
) -> Dict[str, Any]:
    """Prepare detailed parameters without constructing indexed variables."""
    clinics = [str(c) for c in static_sets["c"]]
    facilities = [str(m) for m in static_sets["m"]]
    detailed = build_detailed_parameters(static_sets, static_params)
    patients, patient_inc = expand_patient_arrivals(clinics, time_horizon, inc_param)

    return {
        "c": {None: clinics},
        "h": {None: detailed["h"]},
        "j": {None: detailed["j"]},
        "m": {None: facilities},
        "p": {None: patients},
        "CIM": {m: float(static_params["CIM"][m]) for m in facilities},
        "CVM": {m: float(static_params["CVM"][m]) for m in facilities},
        "CAPMAX": detailed["CAPMAX"],
        "TT1": detailed["TT1"],
        "TT3": detailed["TT3"],
        "U1": detailed["U1"],
        "U3": detailed["U3"],
        "INC": patient_inc,
        "FMAX": {None: detailed["FMAX"]},
        "FMIN": {None: detailed["FMIN"]},
        "TAD": {None: detailed["TAD"]},
        "TLS": {None: detailed["TLS"]},
        "TMFE": {None: detailed["TMFE"]},
    }


def build_detailed_instance_from_aggregate(
    static_sets: Mapping[str, Sequence[str]],
    static_params: Mapping[str, Any],
    inc_param: Mapping[Tuple[str, int], int],
    *,
    time_horizon: int,
    max_open_facilities: Optional[int] = None,
    turnaround_limit: float = 17.0,
) -> ConcreteModel:
    """Construct the detailed patient-level model from aggregate clinic demand."""
    data = build_detailed_data_from_aggregate(
        static_sets, static_params, inc_param, time_horizon=time_horizon
    )
    model = build_detailed_model(
        time_horizon=int(time_horizon),
        max_open_facilities=max_open_facilities,
        turnaround_limit=turnaround_limit,
    )
    return model.create_instance({None: data})


def build_detailed_model(
    time_horizon: int = 130,
    max_open_facilities: Optional[int] = None,
    turnaround_limit: float = 17.0,
) -> AbstractModel:
    """
    Create the detailed AbstractModel.

    Args:
        time_horizon: number of discrete time periods (days). The default
            matches the original script (130).

    Returns:
        A Pyomo AbstractModel ready to be instantiated with patient-level data.
    """
    if not math.isfinite(float(turnaround_limit)) or float(turnaround_limit) <= 0:
        raise ValueError("Turnaround limit must be finite and positive.")
    model = AbstractModel()

    # --- Sets -----------------------------------------------------------------
    model.c = Set()  # Leukapheresis sites
    model.h = Set()  # Hospitals
    model.j = Set()  # Transport modes
    model.m = Set()  # Manufacturing sites
    model.p = Set()  # Patients
    model.t = RangeSet(1, time_horizon)  # Time periods (days)
    model.tt = Set(initialize=lambda m: list(m.t))  # Alias of set t

    # --- Indexed parameters ---------------------------------------------------
    model.CIM = Param(model.m)  # Capital investment for manufacturing facility
    model.TT1 = Param(model.j)  # Transport time LS -> MS via mode j
    model.TT3 = Param(model.j)  # Transport time MS -> hospital via mode j
    model.U1 = Param(model.c, model.m, model.j)  # Cost LS -> MS via mode j
    model.U3 = Param(model.m, model.h, model.j)  # Cost MS -> hospital via mode j
    model.INC = Param(
        model.p,
        model.c,
        model.t,
        within=NonNegativeIntegers,
        default=0,
    )  # Patient arrival
    model.CVM = Param(model.m)  # Fixed variable costs
    model.CAPMAX = Param(model.m)  # Max parallel lines per facility

    # --- Scalar parameters ----------------------------------------------------
    model.FMAX = Param(within=NonNegativeReals)
    model.FMIN = Param(within=NonNegativeReals)
    model.TAD = Param(within=NonNegativeReals)
    model.TLS = Param(within=NonNegativeReals)
    model.TMFE = Param(within=NonNegativeReals, default=7)

    # --- Binary variables -----------------------------------------------------
    model.E1 = Var(model.m, within=Binary)
    model.X1 = Var(model.c, model.m, within=NonNegativeReals)
    model.X3 = Var(model.m, model.h, within=NonNegativeReals)
    model.Y1 = Var(model.p, model.c, model.m, model.j, model.t, within=Binary)
    model.Y3 = Var(model.p, model.m, model.h, model.j, model.t, within=Binary)

    # --- Integer / non-negative variables ------------------------------------
    model.INH = Var(model.p, model.h, model.t, within=NonNegativeIntegers)
    model.CTM = Var(model.p, within=NonNegativeReals)
    model.FTD = Var(model.p, model.m, model.h, model.j, model.t, within=NonNegativeReals)
    model.TTC = Var(model.p, within=NonNegativeReals)
    model.LSA = Var(model.p, model.c, model.m, model.j, model.t, within=NonNegativeReals)
    model.LSR = Var(model.p, model.c, model.m, model.j, model.t, within=NonNegativeReals)
    model.MSO = Var(model.p, model.m, model.h, model.j, model.t, within=NonNegativeReals)
    model.OUTC = Var(model.p, model.c, model.t, within=NonNegativeReals)
    model.OUTM = Var(model.p, model.m, model.t, within=NonNegativeReals)
    model.INM = Var(model.p, model.m, model.t, within=NonNegativeReals)
    model.DURV = Var(model.p, model.m, model.t, within=NonNegativeReals)
    model.RATIO = Var(model.m, model.t, within=NonNegativeReals)
    model.FCAP = Var(model.m, within=NonNegativeIntegers)
    model.TOTCOST = Var()
    model.CAP = Var(model.m, model.t)
    model.TRT = Var(model.p)
    model.ATRT = Var()
    model.STT = Var(model.p)
    model.CTT = Var(model.p)

    # --- Objective ------------------------------------------------------------
    def obj_rule(m):
        material_qc_cost = 10476 + 9312
        return sum(m.CTM[p] for p in m.p) + sum(m.TTC[p] for p in m.p) + material_qc_cost * len(m.p)

    model.obj = Objective(rule=obj_rule)

    # --- Constraints ---------------------------------------------------------
    def C1_rule(m, p):
        horizon = len(m.t)
        patients = len(m.p)
        return m.CTM[p] == sum(
            (m.FCAP[mm] * m.CIM[mm] / m.CAPMAX[mm]) * horizon / patients for mm in m.m
        ) + sum((m.E1[mm] * m.CVM[mm]) * horizon / patients for mm in m.m)

    model.C1 = Constraint(model.p, rule=C1_rule)

    def RATIOEQ_rule(m, mm, t):
        return m.RATIO[mm, t] == sum(m.DURV[p, mm, t] / m.CAPMAX[mm] for p in m.p)

    model.RATIOEQ = Constraint(model.m, model.t, rule=RATIOEQ_rule)

    def MSBnew_rule(m, p, mm, t):
        return m.DURV[p, mm, t] == sum(
            m.INM[p, mm, tt - 1] - m.OUTM[p, mm, tt] for tt in m.tt if tt <= t and tt > 1
        ) + m.OUTM[p, mm, t]

    model.MSBnew = Constraint(model.p, model.m, model.t, rule=MSBnew_rule)

    def C2_rule(m, p):
        return (
            m.TTC[p]
            == sum(m.Y1[p, c, mm, j, t] * m.U1[c, mm, j] for c in m.c for mm in m.m for j in m.j for t in m.t)
            + sum(m.Y3[p, mm, h, j, t] * m.U3[mm, h, j] for mm in m.m for h in m.h for j in m.j for t in m.t)
        )

    model.C2 = Constraint(model.p, rule=C2_rule)

    def MSB1_rule(m, p, c, t, tt):
        if tt == t + value(m.TLS):
            return m.INC[p, c, t] == m.OUTC[p, c, tt]
        return Constraint.Skip

    model.MSB1 = Constraint(model.p, model.c, model.t, model.tt, rule=MSB1_rule)

    def MSB2_rule(m, p, mm, t, tt):
        if tt == t + value(m.TMFE):
            return m.INM[p, mm, t] == m.OUTM[p, mm, tt]
        return Constraint.Skip

    model.MSB2 = Constraint(model.p, model.m, model.t, model.tt, rule=MSB2_rule)

    def MSB8_rule(m, p, mm, t, tt):
        if tt == t + 7:
            return m.OUTM[p, mm, t] == sum(m.MSO[p, mm, h, j, tt] for h in m.h for j in m.j)
        return Constraint.Skip

    model.MSB8 = Constraint(model.p, model.m, model.t, model.tt, rule=MSB8_rule)

    def MSB3_rule(m, p, c, mm, j, t, tt):
        if tt == t + value(m.TT1[j]):
            return m.LSR[p, c, mm, j, t] == m.LSA[p, c, mm, j, tt]
        return Constraint.Skip

    model.MSB3 = Constraint(model.p, model.c, model.m, model.j, model.t, model.tt, rule=MSB3_rule)

    def MSB4_rule(m, p, mm, h, j, t, tt):
        if tt == t + value(m.TT3[j]):
            return m.MSO[p, mm, h, j, t] == m.FTD[p, mm, h, j, tt]
        return Constraint.Skip

    model.MSB4 = Constraint(model.p, model.m, model.h, model.j, model.t, model.tt, rule=MSB4_rule)

    def MSB5_rule(m, p, mm, t):
        return m.INM[p, mm, t] == sum(m.LSA[p, c, mm, j, t] for c in m.c for j in m.j)

    model.MSB5 = Constraint(model.p, model.m, model.t, rule=MSB5_rule)

    def MSB6_rule(m, p, h, t):
        return m.INH[p, h, t] == sum(m.FTD[p, mm, h, j, t] for mm in m.m for j in m.j)

    model.MSB6 = Constraint(model.p, model.h, model.t, rule=MSB6_rule)

    def MSB7_rule(m, p, c, t):
        return m.OUTC[p, c, t] == sum(m.LSR[p, c, mm, j, t] for mm in m.m for j in m.j)

    model.MSB7 = Constraint(model.p, model.c, model.t, rule=MSB7_rule)

    def CAP1_rule(m, mm, t):
        return m.CAP[mm, t] == m.FCAP[mm] - sum(
            m.INM[p, mm, tt] for p in m.p for tt in m.tt if tt < t and tt >= t - value(m.TMFE)
        )

    model.CAP1 = Constraint(model.m, model.t, rule=CAP1_rule)

    def CAPCON1_rule(m, mm, t):
        return sum(m.INM[p, mm, t] for p in m.p) - sum(m.OUTM[p, mm, t] for p in m.p) <= m.CAP[mm, t]

    model.CAPCON1 = Constraint(model.m, model.t, rule=CAPCON1_rule)

    if max_open_facilities is not None:
        def CON1_rule(m):
            return sum(m.E1[mm] for mm in m.m) <= int(max_open_facilities)

        model.CON1 = Constraint(rule=CON1_rule)

    def CON2_rule(m, c, mm):
        return m.X1[c, mm] <= m.E1[mm]

    model.CON2 = Constraint(model.c, model.m, rule=CON2_rule)

    def CON3_rule(m, mm, h):
        return m.X3[mm, h] <= m.E1[mm]

    model.CON3 = Constraint(model.m, model.h, rule=CON3_rule)

    def CON4_rule(m, p, c, mm, j, t):
        return m.Y1[p, c, mm, j, t] <= m.X1[c, mm]

    model.CON4 = Constraint(model.p, model.c, model.m, model.j, model.t, rule=CON4_rule)

    def CON5_rule(m, p, mm, h, j, t):
        return m.Y3[p, mm, h, j, t] <= m.X3[mm, h]

    model.CON5 = Constraint(model.p, model.m, model.h, model.j, model.t, rule=CON5_rule)

    def CON6_rule(m, p):
        return sum(m.Y1[p, c, mm, j, t] for c in m.c for mm in m.m for j in m.j for t in m.t) == 1

    model.CON6 = Constraint(model.p, rule=CON6_rule)

    def CON7_rule(m, p):
        return sum(m.Y3[p, mm, h, j, t] for mm in m.m for h in m.h for j in m.j for t in m.t) == 1

    model.CON7 = Constraint(model.p, rule=CON7_rule)

    def DEM_rule(m):
        return sum(m.INH[p, h, t] for p in m.p for h in m.h for t in m.t) <= len(m.p)

    model.DEM = Constraint(rule=DEM_rule)

    def CON8_rule(m, p, c, mm, j, t):
        return m.LSR[p, c, mm, j, t] >= m.Y1[p, c, mm, j, t] * m.FMIN

    model.CON8 = Constraint(model.p, model.c, model.m, model.j, model.t, rule=CON8_rule)

    def CON9_rule(m, p, c, mm, j, t):
        return m.LSR[p, c, mm, j, t] <= m.Y1[p, c, mm, j, t] * m.FMAX

    model.CON9 = Constraint(model.p, model.c, model.m, model.j, model.t, rule=CON9_rule)

    def CON10_rule(m, p, mm, h, j, t):
        return m.MSO[p, mm, h, j, t] >= m.Y3[p, mm, h, j, t] * m.FMIN

    model.CON10 = Constraint(model.p, model.m, model.h, model.j, model.t, rule=CON10_rule)

    def CON11_rule(m, p, mm, h, j, t):
        return m.MSO[p, mm, h, j, t] <= m.Y3[p, mm, h, j, t] * m.FMAX

    model.CON11 = Constraint(model.p, model.m, model.h, model.j, model.t, rule=CON11_rule)

    def CON12_rule(m, p):
        return sum(m.Y3[p, mm, "h1", j, t] for mm in m.m for j in m.j for t in m.t) == sum(
            m.INC[p, "c1", t] for t in m.t
        )

    model.CON12 = Constraint(model.p, rule=CON12_rule)

    def CON13_rule(m, p):
        return sum(m.Y3[p, mm, "h2", j, t] for mm in m.m for j in m.j for t in m.t) == sum(
            m.INC[p, "c2", t] for t in m.t
        )

    model.CON13 = Constraint(model.p, rule=CON13_rule)

    def CON14_rule(m, p):
        return sum(m.Y3[p, mm, "h3", j, t] for mm in m.m for j in m.j for t in m.t) == sum(
            m.INC[p, "c3", t] for t in m.t
        )

    model.CON14 = Constraint(model.p, rule=CON14_rule)

    def CON15_rule(m, p):
        return sum(m.Y3[p, mm, "h4", j, t] for mm in m.m for j in m.j for t in m.t) == sum(
            m.INC[p, "c4", t] for t in m.t
        )

    model.CON15 = Constraint(model.p, rule=CON15_rule)

    def TCON_rule(m, p):
        return m.TRT[p] <= float(turnaround_limit)

    model.TCON = Constraint(model.p, rule=TCON_rule)

    def START_rule(m, p):
        return m.STT[p] == sum(m.INC[p, c, t] * t for c in m.c for t in m.t)

    model.START = Constraint(model.p, rule=START_rule)

    def END_rule(m, p):
        return m.CTT[p] == sum(m.INH[p, h, t] * t for h in m.h for t in m.t)

    model.END = Constraint(model.p, rule=END_rule)

    def TIME_rule(m, p):
        return m.TRT[p] == m.CTT[p] - m.STT[p]

    model.TIME = Constraint(model.p, rule=TIME_rule)

    def ATIME_rule(m):
        return m.ATRT == sum(m.TRT[p] for p in m.p) / len(m.p)

    model.ATIME = Constraint(rule=ATIME_rule)

    def TSEQ_rule(m, p):
        return m.STT[p] <= m.CTT[p]

    model.TSEQ = Constraint(model.p, rule=TSEQ_rule)

    def MAXCAP_rule(m, mm):
        return m.FCAP[mm] <= m.CAPMAX[mm] * m.E1[mm]

    model.MAXCAP = Constraint(model.m, rule=MAXCAP_rule)

    return model


def load_detailed_data(dat_path: Path | str, time_horizon: int = 130) -> Dict[str, Any]:
    """Parse patient data without constructing any indexed model variables."""
    schema = build_detailed_model(time_horizon=int(time_horizon))
    portal = DataPortal()
    portal.load(filename=Path(dat_path).as_posix(), model=schema)
    return portal.data()


def restrict_detailed_facilities(
    data: Mapping[str, Any], open_facilities: Sequence[str]
) -> Dict[str, Any]:
    """Remove closed facilities and their parameter entries before construction."""
    facilities = list(data["m"][None])
    requested = list(open_facilities)
    if len(set(requested)) != len(requested):
        raise ValueError("Open facilities must not contain duplicates.")
    unknown = set(requested).difference(facilities)
    if unknown:
        raise ValueError(f"Unknown open facilities: {sorted(unknown)}")
    if not requested:
        raise ValueError("An empty facility mask is infeasible for patient-level demand.")
    keep = set(requested)
    restricted = dict(data)
    restricted["m"] = {None: [m for m in facilities if m in keep]}
    for name in ("CIM", "CVM", "CAPMAX"):
        restricted[name] = {m: v for m, v in data[name].items() if m in keep}
    restricted["U1"] = {key: v for key, v in data["U1"].items() if key[1] in keep}
    restricted["U3"] = {key: v for key, v in data["U3"].items() if key[0] in keep}
    return restricted


def build_fixed_mask_instance(
    data: Mapping[str, Any],
    mask: Sequence[int],
    *,
    time_horizon: int,
    max_open_facilities: Optional[int] = None,
    turnaround_limit: float = 17.0,
) -> ConcreteModel:
    """Build only selected facilities and fix all retained E1 variables open."""
    facilities = list(data["m"][None])
    bits = list(mask)
    if len(bits) != len(facilities) or any(bit not in (0, 1) for bit in bits):
        raise ValueError("Mask must contain one binary value per original facility.")
    selected = [m for m, bit in zip(facilities, bits) if bit]
    if max_open_facilities is not None and len(selected) > max_open_facilities:
        raise ValueError("Mask exceeds max_open_facilities.")
    restricted = restrict_detailed_facilities(data, selected)
    model = build_detailed_model(
        time_horizon=int(time_horizon), max_open_facilities=max_open_facilities,
        turnaround_limit=turnaround_limit,
    )
    instance = model.create_instance({None: restricted})
    for facility in instance.m:
        instance.E1[facility].fix(1)
    return instance


def instantiate_model(
    dat_path: Path,
    time_horizon: int,
    max_open_facilities: Optional[int] = None,
    log_fn=None,
    turnaround_limit: float = 17.0,
) -> AbstractModel:
    """
    Load the detailed model and instantiate it with a .dat file.
    """
    if log_fn:
        log_fn(f"[INFO] Building detailed model with time_horizon={time_horizon}")
    model = build_detailed_model(
        time_horizon=time_horizon,
        max_open_facilities=max_open_facilities,
        turnaround_limit=turnaround_limit,
    )
    if log_fn:
        log_fn(f"[INFO] Loading data from {dat_path}")
    portal = DataPortal()
    portal.load(filename=dat_path.as_posix(), model=model)
    if log_fn:
        log_fn("[INFO] Creating concrete instance...")
    instance = model.create_instance(portal)
    if log_fn:
        log_fn("[INFO] Instance created")
    return instance


def _summarise_instance(instance) -> str:
    npat = len(instance.p)
    nfac = len(instance.m)
    nsites = len(instance.c)
    nhosp = len(instance.h)
    horizon = len(instance.t)
    return (
        f"Instance summary: patients={npat}, LS sites={nsites}, hospitals={nhosp}, "
        f"MS sites={nfac}, horizon={horizon}"
    )


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Instantiate the detailed CART model from a .dat file."
    )
    parser.add_argument("--dat", type=Path, required=True, help="Path to patient-level .dat file.")
    parser.add_argument("--time-horizon", type=int, default=130, help="Number of time periods (default 130).")
    parser.add_argument("--turnaround-limit", type=float, default=17.0,
                        help="Maximum patient turnaround in days (default: 17).")
    parser.add_argument(
        "--max-open-facilities",
        type=int,
        default=-1,
        help="Upper bound on open manufacturing facilities. Default: no explicit cap. "
             "Pass a nonnegative integer to enable one.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only build the instance and print a summary (no solve).",
    )
    parser.add_argument("--solver", type=str, default=None, help="Optional Pyomo solver name to run immediately.")
    parser.add_argument(
        "--tee",
        action="store_true",
        help="Stream solver output (passes tee=True to Pyomo).",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Print progress messages while building the model.",
    )
    parser.add_argument(
        "--fix-e1",
        action="store_true",
        help="Fix all E1 variables to 1 before solving (facilities subset should already be applied in the data).",
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=None,
        help="Optional path to write build/solve metrics as JSON.",
    )
    args = parser.parse_args(argv)

    log_fn = print if args.log else None
    max_open_facilities = None if int(args.max_open_facilities) < 0 else int(args.max_open_facilities)
    metrics = {
        "dat": str(args.dat),
        "time_horizon": int(args.time_horizon),
        "turnaround_limit": float(args.turnaround_limit),
        "max_open_facilities": max_open_facilities,
        "solver": args.solver,
        "tee": bool(args.tee),
        "fix_e1": bool(args.fix_e1),
    }
    build_start = time.perf_counter()
    instance = instantiate_model(
        args.dat,
        time_horizon=args.time_horizon,
        max_open_facilities=max_open_facilities,
        log_fn=log_fn,
        turnaround_limit=args.turnaround_limit,
    )
    build_elapsed = time.perf_counter() - build_start
    summary = _summarise_instance(instance)
    print(summary)
    print(f"[METRIC] build_time_sec={build_elapsed:.3f}")
    metrics["build_time_sec"] = build_elapsed
    metrics["instance"] = {
        "patients": len(instance.p),
        "ls_sites": len(instance.c),
        "hospitals": len(instance.h),
        "ms_sites": len(instance.m),
        "horizon": len(instance.t),
        "n_constraints": sum(1 for _ in instance.component_data_objects(Constraint, active=True)),
        "n_binary_vars": sum(
            1
            for v in instance.component_data_objects(Var, active=True)
            if hasattr(v, "is_binary") and v.is_binary()
        ),
    }

    if args.fix_e1:
        fixed = []
        for m in instance.m:
            instance.E1[m].fix(1)
            fixed.append(str(m))
        if args.log:
            print(f"[INFO] Fixed E1 variables for facilities: {fixed}")
        metrics["instance"]["fixed_e1"] = fixed

    if args.solver:
        from pyomo.opt import SolverFactory

        opt = SolverFactory(args.solver)
        if args.log:
            print(f"[INFO] Solving with {args.solver} (tee={args.tee})")
        solve_start = time.perf_counter()
        res = opt.solve(instance, tee=args.tee)
        solve_elapsed = time.perf_counter() - solve_start
        print(res)
        print(f"[METRIC] solve_time_sec={solve_elapsed:.3f}")
        solve_metrics = {"solve_time_sec": solve_elapsed}
        try:
            obj_val = value(instance.obj)
            print(f"[METRIC] objective_value={obj_val:.6f}")
            solve_metrics["objective_value"] = obj_val
        except Exception:
            pass
        term_cond = res.solver.termination_condition
        print(f"[METRIC] termination_condition={term_cond}")
        solve_metrics["termination_condition"] = str(term_cond)
        solver_status = getattr(res.solver, "status", None)
        if solver_status is not None:
            print(f"[METRIC] solver_status={solver_status}")
            solve_metrics["solver_status"] = str(solver_status)
        gap_value = getattr(res.solver, "gap", None)
        gap_float = None
        try:
            undefined_marker = getattr(res.solver, "UndefinedData", None)
        except Exception:
            undefined_marker = None
        if gap_value not in (None, undefined_marker):
            try:
                gap_float = float(gap_value)
            except (TypeError, ValueError):
                gap_float = None
        if gap_float is not None:
            print(f"[METRIC] solver_gap={gap_float}")
            solve_metrics["solver_gap"] = gap_float
        best_bound = getattr(res.solver, "best_bound", None)
        try:
            solve_metrics["best_bound"] = float(best_bound)
        except (TypeError, ValueError):
            pass
        wall_time = getattr(res.solver, "wallclock_time", None)
        try:
            solve_metrics["wallclock_time"] = float(wall_time)
        except (TypeError, ValueError):
            pass
        metrics["solve"] = solve_metrics
        try:
            e1_solution = {str(m): int(round(instance.E1[m].value)) for m in instance.m}
            print(f"[METRIC] E1_solution={e1_solution}")
            metrics["solve"]["E1_solution"] = e1_solution
        except Exception:
            pass
    if args.metrics_json:
        args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
        with args.metrics_json.open("w") as fh:
            json.dump(metrics, fh, indent=2)
        print(f"[INFO] Metrics written to {args.metrics_json}")
    elif not args.summary_only:
        print("Model instantiated. Pass --solver <name> to solve immediately or --summary-only to skip this message.")


if __name__ == "__main__":
    main()
