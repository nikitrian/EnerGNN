"""
Detailed CART supply-chain model implemented directly in gurobipy.

This file is intended to be a literal algebraic port of
`cart_dataset/model_cart_detailed.py`, not a reduced reformulation.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import gurobipy as gp
from gurobipy import GRB

MATERIAL_QC_COST = 10476.0 + 9312.0

_PARAM_ARITY = {
    "CIM": 1,
    "TT1": 1,
    "TT3": 1,
    "U1": 3,
    "U3": 3,
    "INC": 3,
    "CVM": 1,
    "CAPMAX": 1,
    "FMAX": 0,
    "FMIN": 0,
    "TAD": 0,
    "TLS": 0,
    "TMFE": 0,
}


@dataclass
class DetailedCartData:
    c: List[str]
    h: List[str]
    j: List[str]
    m: List[str]
    p: List[str]
    time_horizon: int
    CIM: Dict[str, float]
    TT1: Dict[str, int]
    TT3: Dict[str, int]
    U1: Dict[Tuple[str, str, str], float]
    U3: Dict[Tuple[str, str, str], float]
    INC: Dict[Tuple[str, str, int], int]
    CVM: Dict[str, float]
    CAPMAX: Dict[str, float]
    FMAX: float
    FMIN: float
    TAD: float
    TLS: int
    TMFE: int


@dataclass
class DetailedCartArtifacts:
    E1: gp.tupledict
    env: gp.Env


def _strip_comment(line: str) -> str:
    return line.split("#", 1)[0].strip()


def _collect_block(lines: List[str], start_idx: int, remainder: str) -> Tuple[str, int]:
    chunks = [remainder]
    idx = start_idx
    while ";" not in chunks[-1]:
        idx += 1
        if idx >= len(lines):
            raise ValueError("Unterminated .dat block while parsing detailed CART data.")
        chunks.append(lines[idx])
    text = " ".join(chunks)
    text = text.split(";", 1)[0].strip()
    return text, idx


def _parse_scalar(text: str) -> float:
    tokens = text.split()
    if len(tokens) != 1:
        raise ValueError(f"Expected scalar parameter, got: {text!r}")
    return float(tokens[0])


def parse_patient_dat(dat_path: Path, time_horizon: int) -> DetailedCartData:
    cleaned = [_strip_comment(line) for line in dat_path.read_text(encoding="utf-8-sig").splitlines()]
    cleaned = [line for line in cleaned if line]

    sets: Dict[str, List[str]] = {}
    params: Dict[str, object] = {}

    idx = 0
    while idx < len(cleaned):
        line = cleaned[idx]
        if line.startswith("set "):
            match = re.match(r"^set\s+(\w+)\s*:=\s*(.*)$", line)
            if not match:
                raise ValueError(f"Could not parse set declaration: {line!r}")
            name, remainder = match.groups()
            block, idx = _collect_block(cleaned, idx, remainder)
            sets[name] = block.split()
        elif line.startswith("param "):
            match = re.match(r"^param\s+(\w+)\s*:=\s*(.*)$", line)
            if not match:
                raise ValueError(f"Could not parse param declaration: {line!r}")
            name, remainder = match.groups()
            if name not in _PARAM_ARITY:
                raise ValueError(f"Unsupported parameter '{name}' in {dat_path.as_posix()}.")
            arity = _PARAM_ARITY[name]
            block, idx = _collect_block(cleaned, idx, remainder)
            if arity == 0:
                params[name] = _parse_scalar(block)
            else:
                tokens = block.split()
                stride = arity + 1
                if len(tokens) % stride != 0:
                    raise ValueError(
                        f"Parameter '{name}' expected token stride {stride}, got {len(tokens)} tokens."
                    )
                parsed: Dict[object, float] = {}
                for pos in range(0, len(tokens), stride):
                    key_tokens = tokens[pos : pos + arity]
                    raw_val = float(tokens[pos + arity])
                    if name == "INC":
                        patient, clinic, t_str = key_tokens
                        parsed[(patient, clinic, int(t_str))] = int(round(raw_val))
                    elif arity == 1:
                        parsed[key_tokens[0]] = raw_val
                    else:
                        parsed[tuple(key_tokens)] = raw_val
                params[name] = parsed
        else:
            raise ValueError(f"Unsupported line in {dat_path.as_posix()}: {line!r}")
        idx += 1

    required_sets = {"c", "h", "j", "m", "p"}
    missing_sets = required_sets - sets.keys()
    if missing_sets:
        raise ValueError(f"Missing sets in {dat_path.as_posix()}: {sorted(missing_sets)}")

    missing_params = set(_PARAM_ARITY) - params.keys()
    missing_params.discard("TMFE")
    if missing_params:
        raise ValueError(f"Missing params in {dat_path.as_posix()}: {sorted(missing_params)}")

    return DetailedCartData(
        c=list(sets["c"]),
        h=list(sets["h"]),
        j=list(sets["j"]),
        m=list(sets["m"]),
        p=list(sets["p"]),
        time_horizon=int(time_horizon),
        CIM={str(k): float(v) for k, v in dict(params["CIM"]).items()},
        TT1={str(k): int(round(float(v))) for k, v in dict(params["TT1"]).items()},
        TT3={str(k): int(round(float(v))) for k, v in dict(params["TT3"]).items()},
        U1={(str(c), str(m), str(j)): float(v) for (c, m, j), v in dict(params["U1"]).items()},
        U3={(str(m), str(h), str(j)): float(v) for (m, h, j), v in dict(params["U3"]).items()},
        INC=dict(params["INC"]),
        CVM={str(k): float(v) for k, v in dict(params["CVM"]).items()},
        CAPMAX={str(k): float(v) for k, v in dict(params["CAPMAX"]).items()},
        FMAX=float(params["FMAX"]),
        FMIN=float(params["FMIN"]),
        TAD=float(params["TAD"]),
        TLS=int(round(float(params["TLS"]))),
        TMFE=int(round(float(params["TMFE"])) if "TMFE" in params else 7.0),
    )


def summarise_data(data: DetailedCartData) -> str:
    return (
        f"Instance summary: patients={len(data.p)}, LS sites={len(data.c)}, hospitals={len(data.h)}, "
        f"MS sites={len(data.m)}, horizon={data.time_horizon}"
    )


def detailed_data_to_mapping(data: DetailedCartData) -> Dict[str, Any]:
    """Return the Pyomo-DataPortal-shaped mapping used by dataset scripts."""
    return {
        "c": {None: list(data.c)},
        "h": {None: list(data.h)},
        "j": {None: list(data.j)},
        "m": {None: list(data.m)},
        "p": {None: list(data.p)},
        "CIM": dict(data.CIM),
        "TT1": dict(data.TT1),
        "TT3": dict(data.TT3),
        "U1": dict(data.U1),
        "U3": dict(data.U3),
        "INC": dict(data.INC),
        "CVM": dict(data.CVM),
        "CAPMAX": dict(data.CAPMAX),
        "FMAX": {None: float(data.FMAX)},
        "FMIN": {None: float(data.FMIN)},
        "TAD": {None: float(data.TAD)},
        "TLS": {None: int(data.TLS)},
        "TMFE": {None: int(data.TMFE)},
    }


def _set_values(data: Mapping[str, Any], name: str) -> List[str]:
    return [str(value) for value in data[name][None]]


def _scalar_value(data: Mapping[str, Any], name: str, default: Any = None) -> Any:
    if name not in data:
        return default
    value = data[name]
    if isinstance(value, MappingABC) and None in value:
        return value[None]
    return value


def detailed_data_from_mapping(
    data: Mapping[str, Any] | DetailedCartData,
    *,
    time_horizon: Optional[int] = None,
) -> DetailedCartData:
    """Build the direct-gurobi data object from the existing dataset mapping."""
    if isinstance(data, DetailedCartData):
        return data
    horizon = int(time_horizon) if time_horizon is not None else max(
        [int(key[2]) for key in data.get("INC", {})] or [0]
    )
    return DetailedCartData(
        c=_set_values(data, "c"),
        h=_set_values(data, "h"),
        j=_set_values(data, "j"),
        m=_set_values(data, "m"),
        p=_set_values(data, "p"),
        time_horizon=horizon,
        CIM={str(k): float(v) for k, v in data["CIM"].items()},
        TT1={str(k): int(round(float(v))) for k, v in data["TT1"].items()},
        TT3={str(k): int(round(float(v))) for k, v in data["TT3"].items()},
        U1={(str(c), str(m), str(j)): float(v) for (c, m, j), v in data["U1"].items()},
        U3={(str(m), str(h), str(j)): float(v) for (m, h, j), v in data["U3"].items()},
        INC={(str(p), str(c), int(t)): int(round(float(v))) for (p, c, t), v in data["INC"].items()},
        CVM={str(k): float(v) for k, v in data["CVM"].items()},
        CAPMAX={str(k): float(v) for k, v in data["CAPMAX"].items()},
        FMAX=float(_scalar_value(data, "FMAX")),
        FMIN=float(_scalar_value(data, "FMIN")),
        TAD=float(_scalar_value(data, "TAD")),
        TLS=int(round(float(_scalar_value(data, "TLS")))),
        TMFE=int(round(float(_scalar_value(data, "TMFE", 7)))),
    )


def load_detailed_data(dat_path: Path | str, time_horizon: int = 130) -> Dict[str, Any]:
    """Parse patient data without importing Pyomo."""
    return detailed_data_to_mapping(parse_patient_dat(Path(dat_path), time_horizon=int(time_horizon)))


def restrict_detailed_data(
    data: DetailedCartData,
    open_facilities: Sequence[str],
) -> DetailedCartData:
    requested = [str(facility) for facility in open_facilities]
    if len(set(requested)) != len(requested):
        raise ValueError("Open facilities must not contain duplicates.")
    if not requested:
        raise ValueError("An empty facility mask is infeasible for patient-level demand.")
    return _restrict_data_to_facilities(data, requested)


def _parse_facility_list(raw: Optional[str]) -> Optional[List[str]]:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def _restrict_data_to_facilities(
    data: DetailedCartData,
    open_facilities: List[str],
) -> DetailedCartData:
    keep = list(open_facilities)
    keep_set = set(keep)
    unknown = [mm for mm in keep if mm not in data.m]
    if unknown:
        raise ValueError(f"Unknown facilities in fixed subset: {unknown}")

    return DetailedCartData(
        c=list(data.c),
        h=list(data.h),
        j=list(data.j),
        m=keep,
        p=list(data.p),
        time_horizon=int(data.time_horizon),
        CIM={k: v for k, v in data.CIM.items() if k in keep_set},
        TT1=dict(data.TT1),
        TT3=dict(data.TT3),
        U1={(c, m, j): v for (c, m, j), v in data.U1.items() if m in keep_set},
        U3={(m, h, j): v for (m, h, j), v in data.U3.items() if m in keep_set},
        INC=dict(data.INC),
        CVM={k: v for k, v in data.CVM.items() if k in keep_set},
        CAPMAX={k: v for k, v in data.CAPMAX.items() if k in keep_set},
        FMAX=float(data.FMAX),
        FMIN=float(data.FMIN),
        TAD=float(data.TAD),
        TLS=int(data.TLS),
        TMFE=int(data.TMFE),
    )


def build_detailed_gurobi_model(
    data: DetailedCartData,
    *,
    max_open_facilities: Optional[int] = None,
    turnaround_limit: float = 17.0,
    solver_threads: Optional[int] = None,
    time_limit: Optional[float] = None,
    mip_gap: Optional[float] = None,
    solver_options: Optional[Dict[str, object]] = None,
    fix_e1: bool = False,
    fixed_e1_values: Optional[Dict[str, int]] = None,
    e1_upper_bounds: Optional[Dict[str, int]] = None,
    hamming_center_values: Optional[Dict[str, int]] = None,
    hamming_radius: Optional[int] = None,
    conpas_x0_facilities: Optional[List[str]] = None,
    conpas_x1_facilities: Optional[List[str]] = None,
    conpas_delta: Optional[int] = None,
    min_open_facilities: Optional[int] = None,
    forced_e1_values: Optional[Dict[str, int]] = None,
    tee: bool = False,
) -> Tuple[gp.Model, DetailedCartArtifacts]:
    if not math.isfinite(float(turnaround_limit)) or float(turnaround_limit) <= 0:
        raise ValueError("Turnaround limit must be finite and positive.")
    forced = dict(forced_e1_values or {})
    if not set(forced).issubset(data.m) or any(value not in (0, 1) for value in forced.values()):
        raise ValueError("Forced E1 values must map known facilities to binary values.")
    options = dict(solver_options or {})
    T = list(range(1, int(data.time_horizon) + 1))

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 1 if tee else 0)
    env.start()

    model = gp.Model("Detailed_CART_Gurobi", env=env)
    if time_limit is not None:
        model.setParam("TimeLimit", float(time_limit))
    if solver_threads is not None:
        model.setParam("Threads", int(solver_threads))
    if mip_gap is not None:
        model.setParam("MIPGap", float(mip_gap))
    for key, value in options.items():
        model.setParam(str(key), value)

    E1 = model.addVars(data.m, vtype=GRB.BINARY, name="E1")
    if min_open_facilities is not None and int(min_open_facilities) > 0:
        model.addConstr(E1.sum() >= int(min_open_facilities), name="CON1_MinOpenFacilities")
    for facility, bit in forced.items():
        model.addConstr(E1[facility] == int(bit), name=f"FORCED_E1[{facility}]")
    X1 = model.addVars(data.c, data.m, lb=0.0, vtype=GRB.CONTINUOUS, name="X1")
    X3 = model.addVars(data.m, data.h, lb=0.0, vtype=GRB.CONTINUOUS, name="X3")
    Y1 = model.addVars(data.p, data.c, data.m, data.j, T, vtype=GRB.BINARY, name="Y1")
    Y3 = model.addVars(data.p, data.m, data.h, data.j, T, vtype=GRB.BINARY, name="Y3")

    INH = model.addVars(data.p, data.h, T, lb=0.0, vtype=GRB.INTEGER, name="INH")
    CTM = model.addVars(data.p, lb=0.0, vtype=GRB.CONTINUOUS, name="CTM")
    FTD = model.addVars(data.p, data.m, data.h, data.j, T, lb=0.0, vtype=GRB.CONTINUOUS, name="FTD")
    TTC = model.addVars(data.p, lb=0.0, vtype=GRB.CONTINUOUS, name="TTC")
    LSA = model.addVars(data.p, data.c, data.m, data.j, T, lb=0.0, vtype=GRB.CONTINUOUS, name="LSA")
    LSR = model.addVars(data.p, data.c, data.m, data.j, T, lb=0.0, vtype=GRB.CONTINUOUS, name="LSR")
    MSO = model.addVars(data.p, data.m, data.h, data.j, T, lb=0.0, vtype=GRB.CONTINUOUS, name="MSO")
    OUTC = model.addVars(data.p, data.c, T, lb=0.0, vtype=GRB.CONTINUOUS, name="OUTC")
    OUTM = model.addVars(data.p, data.m, T, lb=0.0, vtype=GRB.CONTINUOUS, name="OUTM")
    INM = model.addVars(data.p, data.m, T, lb=0.0, vtype=GRB.CONTINUOUS, name="INM")
    DURV = model.addVars(data.p, data.m, T, lb=0.0, vtype=GRB.CONTINUOUS, name="DURV")
    RATIO = model.addVars(data.m, T, lb=0.0, vtype=GRB.CONTINUOUS, name="RATIO")
    FCAP = model.addVars(data.m, lb=0.0, vtype=GRB.INTEGER, name="FCAP")

    TOTCOST = model.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="TOTCOST")
    CAP = model.addVars(data.m, T, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="CAP")
    TRT = model.addVars(data.p, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="TRT")
    ATRT = model.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="ATRT")
    STT = model.addVars(data.p, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="STT")
    CTT = model.addVars(data.p, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="CTT")

    horizon = float(len(T))
    patients = float(len(data.p))
    model.setObjective(
        gp.quicksum(CTM[p] for p in data.p)
        + gp.quicksum(TTC[p] for p in data.p)
        + float(MATERIAL_QC_COST) * len(data.p),
        GRB.MINIMIZE,
    )

    if max_open_facilities is not None and int(max_open_facilities) >= 0:
        model.addConstr(
            gp.quicksum(E1[mm] for mm in data.m) <= int(max_open_facilities),
            name="CON1_MaxOpenFacilities",
        )

    for p in data.p:
        model.addConstr(
            CTM[p]
            == gp.quicksum((FCAP[mm] * float(data.CIM[mm]) / float(data.CAPMAX[mm])) * horizon / patients for mm in data.m)
            + gp.quicksum((E1[mm] * float(data.CVM[mm])) * horizon / patients for mm in data.m),
            name=f"C1[{p}]",
        )

    for mm in data.m:
        capmax = float(data.CAPMAX[mm])
        for t in T:
            model.addConstr(
                RATIO[mm, t] == gp.quicksum(DURV[p, mm, t] / capmax for p in data.p),
                name=f"RATIOEQ[{mm},{t}]",
            )

    for p in data.p:
        for mm in data.m:
            for t in T:
                model.addConstr(
                    DURV[p, mm, t]
                    == gp.quicksum(INM[p, mm, tt - 1] - OUTM[p, mm, tt] for tt in T if tt <= t and tt > 1)
                    + OUTM[p, mm, t],
                    name=f"MSBnew[{p},{mm},{t}]",
                )

    for p in data.p:
        model.addConstr(
            TTC[p]
            == gp.quicksum(Y1[p, c, mm, j, t] * float(data.U1[(c, mm, j)]) for c in data.c for mm in data.m for j in data.j for t in T)
            + gp.quicksum(Y3[p, mm, h, j, t] * float(data.U3[(mm, h, j)]) for mm in data.m for h in data.h for j in data.j for t in T),
            name=f"C2[{p}]",
        )

    tls = int(data.TLS)
    tmfe = int(data.TMFE)
    for p in data.p:
        for c in data.c:
            for t in T:
                tt = t + tls
                if tt in T:
                    rhs = int(data.INC.get((p, c, t), 0))
                    model.addConstr(OUTC[p, c, tt] == rhs, name=f"MSB1[{p},{c},{t},{tt}]")

    for p in data.p:
        for mm in data.m:
            for t in T:
                tt = t + tmfe
                if tt in T:
                    model.addConstr(INM[p, mm, t] == OUTM[p, mm, tt], name=f"MSB2[{p},{mm},{t},{tt}]")

    for p in data.p:
        for mm in data.m:
            for t in T:
                tt = t + 7
                if tt in T:
                    model.addConstr(
                        OUTM[p, mm, t] == gp.quicksum(MSO[p, mm, h, j, tt] for h in data.h for j in data.j),
                        name=f"MSB8[{p},{mm},{t},{tt}]",
                    )

    for p in data.p:
        for c in data.c:
            for mm in data.m:
                for j in data.j:
                    shift = int(data.TT1[j])
                    for t in T:
                        tt = t + shift
                        if tt in T:
                            model.addConstr(
                                LSR[p, c, mm, j, t] == LSA[p, c, mm, j, tt],
                                name=f"MSB3[{p},{c},{mm},{j},{t},{tt}]",
                            )

    for p in data.p:
        for mm in data.m:
            for h in data.h:
                for j in data.j:
                    shift = int(data.TT3[j])
                    for t in T:
                        tt = t + shift
                        if tt in T:
                            model.addConstr(
                                MSO[p, mm, h, j, t] == FTD[p, mm, h, j, tt],
                                name=f"MSB4[{p},{mm},{h},{j},{t},{tt}]",
                            )

    for p in data.p:
        for mm in data.m:
            for t in T:
                model.addConstr(
                    INM[p, mm, t] == gp.quicksum(LSA[p, c, mm, j, t] for c in data.c for j in data.j),
                    name=f"MSB5[{p},{mm},{t}]",
                )

    for p in data.p:
        for h in data.h:
            for t in T:
                model.addConstr(
                    INH[p, h, t] == gp.quicksum(FTD[p, mm, h, j, t] for mm in data.m for j in data.j),
                    name=f"MSB6[{p},{h},{t}]",
                )

    for p in data.p:
        for c in data.c:
            for t in T:
                model.addConstr(
                    OUTC[p, c, t] == gp.quicksum(LSR[p, c, mm, j, t] for mm in data.m for j in data.j),
                    name=f"MSB7[{p},{c},{t}]",
                )

    for mm in data.m:
        for t in T:
            model.addConstr(
                CAP[mm, t]
                == FCAP[mm]
                - gp.quicksum(INM[p, mm, tt] for p in data.p for tt in T if tt < t and tt >= t - tmfe),
                name=f"CAP1[{mm},{t}]",
            )
            model.addConstr(
                gp.quicksum(INM[p, mm, t] for p in data.p) - gp.quicksum(OUTM[p, mm, t] for p in data.p) <= CAP[mm, t],
                name=f"CAPCON1[{mm},{t}]",
            )

    for c in data.c:
        for mm in data.m:
            model.addConstr(X1[c, mm] <= E1[mm], name=f"CON2[{c},{mm}]")

    for mm in data.m:
        for h in data.h:
            model.addConstr(X3[mm, h] <= E1[mm], name=f"CON3[{mm},{h}]")

    for p in data.p:
        for c in data.c:
            for mm in data.m:
                for j in data.j:
                    for t in T:
                        model.addConstr(Y1[p, c, mm, j, t] <= X1[c, mm], name=f"CON4[{p},{c},{mm},{j},{t}]")

    for p in data.p:
        for mm in data.m:
            for h in data.h:
                for j in data.j:
                    for t in T:
                        model.addConstr(Y3[p, mm, h, j, t] <= X3[mm, h], name=f"CON5[{p},{mm},{h},{j},{t}]")

    for p in data.p:
        model.addConstr(
            gp.quicksum(Y1[p, c, mm, j, t] for c in data.c for mm in data.m for j in data.j for t in T) == 1.0,
            name=f"CON6[{p}]",
        )
        model.addConstr(
            gp.quicksum(Y3[p, mm, h, j, t] for mm in data.m for h in data.h for j in data.j for t in T) == 1.0,
            name=f"CON7[{p}]",
        )

    model.addConstr(
        gp.quicksum(INH[p, h, t] for p in data.p for h in data.h for t in T) <= len(data.p),
        name="DEM",
    )

    fmin = float(data.FMIN)
    fmax = float(data.FMAX)
    for p in data.p:
        for c in data.c:
            for mm in data.m:
                for j in data.j:
                    for t in T:
                        model.addConstr(LSR[p, c, mm, j, t] >= Y1[p, c, mm, j, t] * fmin, name=f"CON8[{p},{c},{mm},{j},{t}]")
                        model.addConstr(LSR[p, c, mm, j, t] <= Y1[p, c, mm, j, t] * fmax, name=f"CON9[{p},{c},{mm},{j},{t}]")

    for p in data.p:
        for mm in data.m:
            for h in data.h:
                for j in data.j:
                    for t in T:
                        model.addConstr(MSO[p, mm, h, j, t] >= Y3[p, mm, h, j, t] * fmin, name=f"CON10[{p},{mm},{h},{j},{t}]")
                        model.addConstr(MSO[p, mm, h, j, t] <= Y3[p, mm, h, j, t] * fmax, name=f"CON11[{p},{mm},{h},{j},{t}]")

    clinic_hospital_pairs = [("c1", "h1"), ("c2", "h2"), ("c3", "h3"), ("c4", "h4")]
    for p in data.p:
        for idx, (clinic, hospital) in enumerate(clinic_hospital_pairs, start=12):
            if clinic in data.c and hospital in data.h:
                model.addConstr(
                    gp.quicksum(Y3[p, mm, hospital, j, t] for mm in data.m for j in data.j for t in T)
                    == gp.quicksum(int(data.INC.get((p, clinic, t), 0)) for t in T),
                    name=f"CON{idx}[{p}]",
                )

    for p in data.p:
        model.addConstr(TRT[p] <= float(turnaround_limit), name=f"TCON[{p}]")
        model.addConstr(
            STT[p] == gp.quicksum(int(data.INC.get((p, c, t), 0)) * t for c in data.c for t in T),
            name=f"START[{p}]",
        )
        model.addConstr(
            CTT[p] == gp.quicksum(INH[p, h, t] * t for h in data.h for t in T),
            name=f"END[{p}]",
        )
        model.addConstr(TRT[p] == CTT[p] - STT[p], name=f"TIME[{p}]")
        model.addConstr(STT[p] <= CTT[p], name=f"TSEQ[{p}]")

    model.addConstr(ATRT == gp.quicksum(TRT[p] for p in data.p) / float(len(data.p)), name="ATIME")

    for mm in data.m:
        model.addConstr(FCAP[mm] <= float(data.CAPMAX[mm]) * E1[mm], name=f"MAXCAP[{mm}]")

    if fix_e1 and fixed_e1_values is not None:
        raise ValueError("Pass only one of fix_e1=True or fixed_e1_values.")
    if fix_e1 and e1_upper_bounds is not None:
        raise ValueError("Pass only one of fix_e1=True or e1_upper_bounds.")
    if fixed_e1_values is not None and e1_upper_bounds is not None:
        raise ValueError("Pass only one of fixed_e1_values or e1_upper_bounds.")
    if hamming_center_values is not None and hamming_radius is None:
        raise ValueError("hamming_radius is required when hamming_center_values is provided.")
    has_rank_neighbourhood = (
        bool(conpas_x0_facilities)
        or bool(conpas_x1_facilities)
        or conpas_delta is not None
    )
    if hamming_center_values is not None and has_rank_neighbourhood:
        raise ValueError("Pass only one of hamming_center_values or ConPaS rank neighbourhood.")
    if (bool(conpas_x0_facilities) or bool(conpas_x1_facilities)) and conpas_delta is None:
        raise ValueError("conpas_delta is required when ConPaS rank facilities are provided.")

    if fixed_e1_values is not None:
        unknown = [mm for mm in fixed_e1_values if mm not in data.m]
        if unknown:
            raise ValueError(f"Unknown facilities in fixed_e1_values: {unknown}")
        for mm in data.m:
            value = int(fixed_e1_values.get(mm, 0))
            if value not in {0, 1}:
                raise ValueError(f"Fixed E1 value for {mm} must be 0 or 1, got {value}.")
            E1[mm].lb = float(value)
            E1[mm].ub = float(value)
    elif e1_upper_bounds is not None:
        unknown = [mm for mm in e1_upper_bounds if mm not in data.m]
        if unknown:
            raise ValueError(f"Unknown facilities in e1_upper_bounds: {unknown}")
        for mm in data.m:
            value = int(e1_upper_bounds.get(mm, 0))
            if value not in {0, 1}:
                raise ValueError(f"E1 upper bound for {mm} must be 0 or 1, got {value}.")
            E1[mm].ub = float(value)
    elif fix_e1:
        for mm in data.m:
            E1[mm].lb = 1.0
            E1[mm].ub = 1.0

    if hamming_center_values is not None:
        unknown = [mm for mm in hamming_center_values if mm not in data.m]
        missing = [mm for mm in data.m if mm not in hamming_center_values]
        if unknown:
            raise ValueError(f"Unknown facilities in hamming_center_values: {unknown}")
        if missing:
            raise ValueError(f"Missing facilities in hamming_center_values: {missing}")
        radius = int(hamming_radius)
        if radius < 0:
            raise ValueError(f"hamming_radius must be nonnegative, got {radius}.")
        lhs = gp.LinExpr()
        for mm in data.m:
            value = int(hamming_center_values[mm])
            if value not in {0, 1}:
                raise ValueError(f"Hamming center value for {mm} must be 0 or 1, got {value}.")
            lhs += (1.0 - E1[mm]) if value == 1 else E1[mm]
        model.addConstr(lhs <= float(radius), name="CONPAS_Hamming_E1")

    if has_rank_neighbourhood:
        x0 = [str(mm) for mm in (conpas_x0_facilities or [])]
        x1 = [str(mm) for mm in (conpas_x1_facilities or [])]
        unknown = [mm for mm in [*x0, *x1] if mm not in data.m]
        if unknown:
            raise ValueError(f"Unknown facilities in ConPaS rank neighbourhood: {unknown}")
        overlap = sorted(set(x0) & set(x1))
        if overlap:
            raise ValueError(f"ConPaS X0 and X1 overlap: {overlap}")
        delta = int(conpas_delta or 0)
        if delta < 0:
            raise ValueError(f"conpas_delta must be nonnegative, got {delta}.")
        if delta > len(x0) + len(x1):
            # The constraint is redundant, but keeping it is harmless and auditable.
            pass
        if x0 or x1:
            lhs = gp.LinExpr()
            for mm in x0:
                lhs += E1[mm]
            for mm in x1:
                lhs += 1.0 - E1[mm]
            model.addConstr(lhs <= float(delta), name="CONPAS_Rank_E1")

    model.update()
    _ = TOTCOST  # preserve exact variable inventory even though Pyomo never uses it
    return model, DetailedCartArtifacts(E1=E1, env=env)


def _status_strings(model: gp.Model) -> Tuple[str, str]:
    status = int(model.Status)
    if status == GRB.OPTIMAL:
        return "optimal", "ok"
    if status == GRB.INFEASIBLE:
        return "infeasible", "warning"
    if status == GRB.TIME_LIMIT:
        return "time_limit", "warning"
    if status == GRB.INTERRUPTED:
        return "interrupted", "warning"
    if status == GRB.NODE_LIMIT:
        return "node_limit", "warning"
    if status == GRB.MEM_LIMIT:
        return "memory_limit", "warning"
    if status == GRB.INF_OR_UNBD:
        return "inf_or_unbd", "warning"
    return f"status_{status}", "warning"


def solve_detailed_gurobi_dat(
    dat_path: Path,
    *,
    time_horizon: int = 130,
    max_open_facilities: Optional[int] = None,
    turnaround_limit: float = 17.0,
    solver_threads: Optional[int] = None,
    time_limit: Optional[float] = None,
    mip_gap: Optional[float] = None,
    solver_options: Optional[Dict[str, object]] = None,
    fix_e1: bool = False,
    fixed_open_facilities: Optional[List[str]] = None,
    fixed_e1_mask: Optional[List[int]] = None,
    prune_fixed_facilities: bool = False,
    allow_drop_fixed_facilities: bool = False,
    hamming_center_mask: Optional[List[int]] = None,
    hamming_radius: Optional[int] = None,
    conpas_x0_facilities: Optional[List[str]] = None,
    conpas_x1_facilities: Optional[List[str]] = None,
    conpas_delta: Optional[int] = None,
    tee: bool = False,
) -> Dict[str, object]:
    result: Dict[str, object] = {
        "dat": dat_path.as_posix(),
        "dat_name": dat_path.name,
        "turnaround_limit": float(turnaround_limit),
        "ok": False,
    }
    total_start = time.perf_counter()
    build_start = time.perf_counter()
    data = parse_patient_dat(dat_path, time_horizon=int(time_horizon))

    if fixed_open_facilities is not None and fixed_e1_mask is not None:
        raise ValueError("Pass only one of fixed_open_facilities or fixed_e1_mask.")
    if prune_fixed_facilities and allow_drop_fixed_facilities:
        raise ValueError("Pass only one of prune_fixed_facilities or allow_drop_fixed_facilities.")
    if prune_fixed_facilities and hamming_center_mask is not None:
        raise ValueError("hamming_center_mask cannot be combined with prune_fixed_facilities.")
    has_rank_neighbourhood = (
        bool(conpas_x0_facilities)
        or bool(conpas_x1_facilities)
        or conpas_delta is not None
    )
    if prune_fixed_facilities and has_rank_neighbourhood:
        raise ValueError("ConPaS rank neighbourhood cannot be combined with prune_fixed_facilities.")
    if hamming_center_mask is not None and has_rank_neighbourhood:
        raise ValueError("Pass only one of hamming_center_mask or ConPaS rank neighbourhood.")
    if hamming_center_mask is not None and hamming_radius is None:
        raise ValueError("hamming_radius is required when hamming_center_mask is provided.")
    if hamming_radius is not None and int(hamming_radius) < 0:
        raise ValueError(f"hamming_radius must be nonnegative, got {hamming_radius}.")
    if (bool(conpas_x0_facilities) or bool(conpas_x1_facilities)) and conpas_delta is None:
        raise ValueError("conpas_delta is required when ConPaS rank facilities are provided.")
    if conpas_delta is not None and int(conpas_delta) < 0:
        raise ValueError(f"conpas_delta must be nonnegative, got {conpas_delta}.")

    original_facilities = list(data.m)
    requested_open_facilities: Optional[List[str]] = None
    fixed_e1_values: Optional[Dict[str, int]] = None
    e1_upper_bounds: Optional[Dict[str, int]] = None
    hamming_center_values: Optional[Dict[str, int]] = None
    raw_hamming_mask: Optional[List[int]] = None
    if hamming_center_mask is not None:
        raw_hamming_mask = [int(v) for v in hamming_center_mask]
        if len(raw_hamming_mask) != len(data.m):
            raise ValueError(
                f"Hamming center mask length {len(raw_hamming_mask)} does not match "
                f"number of facilities {len(data.m)}."
            )
        bad_values = [v for v in raw_hamming_mask if int(v) not in {0, 1}]
        if bad_values:
            raise ValueError(f"Hamming center mask must contain only 0/1 values, got {bad_values}.")
        hamming_center_values = {
            mm: int(bit)
            for mm, bit in zip(data.m, raw_hamming_mask)
        }
    if fixed_e1_mask is not None:
        raw_mask = [int(v) for v in fixed_e1_mask]
        if len(raw_mask) != len(data.m):
            raise ValueError(
                f"Fixed E1 mask length {len(raw_mask)} does not match number of facilities {len(data.m)}."
            )
        bad_values = [v for v in raw_mask if int(v) not in {0, 1}]
        if bad_values:
            raise ValueError(f"Fixed E1 mask must contain only 0/1 values, got {bad_values}.")
        requested_open_facilities = [
            mm for mm, bit in zip(data.m, raw_mask) if int(bit) == 1
        ]
    elif fixed_open_facilities is not None:
        requested_open_facilities = [str(mm) for mm in fixed_open_facilities]

    fixed_mode = None
    if requested_open_facilities is not None:
        if bool(prune_fixed_facilities):
            fixed_mode = "pruned_exact"
        elif bool(allow_drop_fixed_facilities):
            fixed_mode = "availability_mask"
        else:
            fixed_mode = "full_exact_mask"

    if requested_open_facilities is not None and not requested_open_facilities:
        build_time = time.perf_counter() - build_start
        result["build_time_sec"] = float(build_time)
        result["summary"] = summarise_data(data)
        result["instance"] = {
            "patients": len(data.p),
            "ls_sites": len(data.c),
            "hospitals": len(data.h),
            "ms_sites": len(data.m),
            "original_ms_sites": len(original_facilities),
            "horizon": int(data.time_horizon),
            "max_open_facilities": None if max_open_facilities is None else int(max_open_facilities),
            "requested_fixed_open_facilities": [],
            "candidate_facilities": [],
            "fixed_e1_mode": fixed_mode,
            "n_constraints": 0,
            "n_binary_vars": 0,
        }
        empty_solve = {
            "solve_time_sec": 0.0,
            "termination_condition": "infeasible",
            "solver_status": "warning",
            "applied_solver_options": {
                **({"TimeLimit": float(time_limit)} if time_limit is not None else {}),
                **({"Threads": int(solver_threads)} if solver_threads is not None else {}),
                **({"MIPGap": float(mip_gap)} if mip_gap is not None else {}),
                **({"MaxOpenFacilities": int(max_open_facilities)} if max_open_facilities is not None else {}),
                **({"HammingRadius": int(hamming_radius)} if hamming_radius is not None else {}),
                **({"ConPaSDelta": int(conpas_delta)} if conpas_delta is not None else {}),
                **dict(solver_options or {}),
            },
        }
        if bool(allow_drop_fixed_facilities):
            empty_solve["available_facilities"] = []
        else:
            empty_solve["fixed_open_facilities"] = []
        result["solve"] = empty_solve
        result["total_time_sec"] = float(time.perf_counter() - total_start)
        result["ok"] = True
        return result

    if requested_open_facilities is not None and bool(prune_fixed_facilities):
        data = _restrict_data_to_facilities(data, requested_open_facilities)
        fix_e1 = True
    elif requested_open_facilities is not None:
        requested_set = set(requested_open_facilities)
        unknown = [mm for mm in requested_open_facilities if mm not in data.m]
        if unknown:
            raise ValueError(f"Unknown facilities in fixed subset: {unknown}")
        if bool(allow_drop_fixed_facilities):
            e1_upper_bounds = {mm: (1 if mm in requested_set else 0) for mm in data.m}
        else:
            fixed_e1_values = {mm: (1 if mm in requested_set else 0) for mm in data.m}

    model, artifacts = build_detailed_gurobi_model(
        data,
        max_open_facilities=max_open_facilities,
        turnaround_limit=turnaround_limit,
        solver_threads=solver_threads,
        time_limit=time_limit,
        mip_gap=mip_gap,
        solver_options=solver_options,
        fix_e1=fix_e1,
        fixed_e1_values=fixed_e1_values,
        e1_upper_bounds=e1_upper_bounds,
        hamming_center_values=hamming_center_values,
        hamming_radius=hamming_radius,
        conpas_x0_facilities=conpas_x0_facilities,
        conpas_x1_facilities=conpas_x1_facilities,
        conpas_delta=conpas_delta,
        tee=tee,
    )
    build_time = time.perf_counter() - build_start

    result["build_time_sec"] = float(build_time)
    result["summary"] = summarise_data(data)
    result["instance"] = {
        "patients": len(data.p),
        "ls_sites": len(data.c),
        "hospitals": len(data.h),
        "ms_sites": len(data.m),
        "original_ms_sites": len(original_facilities),
        "horizon": int(data.time_horizon),
        "max_open_facilities": None if max_open_facilities is None else int(max_open_facilities),
        "n_constraints": int(model.NumConstrs),
        "n_binary_vars": int(model.NumBinVars),
    }
    if hamming_center_values is not None:
        result["instance"]["conpas_hamming_constraint"] = True
        result["instance"]["hamming_radius"] = int(hamming_radius)
        result["instance"]["hamming_center_mask"] = list(raw_hamming_mask or [])
        result["instance"]["hamming_center_open_facilities"] = [
            mm for mm in data.m if int(hamming_center_values[mm]) == 1
        ]
    if has_rank_neighbourhood:
        x0 = [str(mm) for mm in (conpas_x0_facilities or [])]
        x1 = [str(mm) for mm in (conpas_x1_facilities or [])]
        result["instance"]["conpas_rank_neighbourhood"] = True
        result["instance"]["conpas_x0_facilities"] = x0
        result["instance"]["conpas_x1_facilities"] = x1
        result["instance"]["conpas_delta"] = int(conpas_delta or 0)
        result["instance"]["selected_zero_variables"] = len(x0)
        result["instance"]["selected_one_variables"] = len(x1)
        result["instance"]["allowed_disagreements"] = int(conpas_delta or 0)
        result["instance"]["effective_fixed_variables"] = len(x0) + len(x1) if int(conpas_delta or 0) == 0 else 0
    if fix_e1:
        result["instance"]["fixed_e1"] = list(data.m)
    if requested_open_facilities is not None:
        result["instance"]["requested_fixed_open_facilities"] = list(requested_open_facilities)
        result["instance"]["candidate_facilities"] = list(requested_open_facilities)
        result["instance"]["fixed_e1_mode"] = fixed_mode
        if fixed_e1_values is not None:
            result["instance"]["fixed_e1"] = dict(fixed_e1_values)
        if e1_upper_bounds is not None:
            result["instance"]["e1_upper_bounds"] = dict(e1_upper_bounds)

    solve_start = time.perf_counter()
    model.optimize()
    solve_time = time.perf_counter() - solve_start

    term, solver_status = _status_strings(model)
    solve_metrics: Dict[str, object] = {
        "solve_time_sec": float(solve_time),
        "termination_condition": term,
        "solver_status": solver_status,
        "applied_solver_options": {
            **({"TimeLimit": float(time_limit)} if time_limit is not None else {}),
            **({"Threads": int(solver_threads)} if solver_threads is not None else {}),
            **({"MIPGap": float(mip_gap)} if mip_gap is not None else {}),
            **({"MaxOpenFacilities": int(max_open_facilities)} if max_open_facilities is not None else {}),
            **({"HammingRadius": int(hamming_radius)} if hamming_radius is not None else {}),
            **({"ConPaSDelta": int(conpas_delta)} if conpas_delta is not None else {}),
            **dict(solver_options or {}),
        },
    }
    if hamming_center_values is not None:
        solve_metrics["hamming_radius"] = int(hamming_radius)
        solve_metrics["hamming_center_open_facilities"] = [
            mm for mm in data.m if int(hamming_center_values[mm]) == 1
        ]
    if has_rank_neighbourhood:
        x0 = [str(mm) for mm in (conpas_x0_facilities or [])]
        x1 = [str(mm) for mm in (conpas_x1_facilities or [])]
        solve_metrics["conpas_delta"] = int(conpas_delta or 0)
        solve_metrics["conpas_x0_facilities"] = x0
        solve_metrics["conpas_x1_facilities"] = x1
        solve_metrics["selected_zero_variables"] = len(x0)
        solve_metrics["selected_one_variables"] = len(x1)
        solve_metrics["allowed_disagreements"] = int(conpas_delta or 0)
        solve_metrics["effective_fixed_variables"] = len(x0) + len(x1) if int(conpas_delta or 0) == 0 else 0
    if model.SolCount > 0:
        solve_metrics["objective_value"] = float(model.ObjVal)
        solve_metrics["E1_solution"] = {
            facility: int(round(artifacts.E1[facility].X))
            for facility in data.m
        }
    if requested_open_facilities is not None and bool(allow_drop_fixed_facilities):
        solve_metrics["available_facilities"] = list(requested_open_facilities)
    elif requested_open_facilities is not None:
        solve_metrics["fixed_open_facilities"] = list(requested_open_facilities)
    try:
        solve_metrics["solver_gap"] = float(model.MIPGap)
    except Exception:
        pass
    try:
        solve_metrics["best_bound"] = float(model.ObjBound)
    except Exception:
        pass
    try:
        solve_metrics["wallclock_time"] = float(model.Runtime)
    except Exception:
        pass
    try:
        solve_metrics["node_count"] = float(model.NodeCount)
    except Exception:
        pass

    result["solve"] = solve_metrics
    result["total_time_sec"] = float(time.perf_counter() - total_start)
    result["ok"] = True
    return result


def _coerce_option_value(raw: str):
    text = str(raw).strip()
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if any(ch in text for ch in [".", "e", "E"]):
            return float(text)
        return int(text)
    except ValueError:
        return text


def _parse_solver_options(option_args: Iterable[str]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for raw in option_args:
        item = str(raw).strip()
        if "=" not in item:
            raise ValueError(f"Invalid --solver-option '{raw}'. Expected KEY=VALUE.")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --solver-option '{raw}': empty key.")
        out[key] = _coerce_option_value(value)
    return out


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build and solve the detailed CART model directly in gurobipy."
    )
    parser.add_argument("--dat", type=Path, required=True, help="Path to patient-level .dat file.")
    parser.add_argument("--time-horizon", type=int, default=130)
    parser.add_argument("--turnaround-limit", type=float, default=17.0,
                        help="Maximum patient turnaround in days (default: 17).")
    parser.add_argument(
        "--max-open-facilities",
        type=int,
        default=-1,
        help="Upper bound on open manufacturing facilities. Default: no explicit cap. "
             "Pass a nonnegative integer to enable one.",
    )
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--time-limit", type=float, default=None)
    parser.add_argument("--mip-gap", type=float, default=None)
    parser.add_argument("--solver-option", action="append", default=[],
                        help="Extra Gurobi parameter KEY=VALUE. May be repeated.")
    parser.add_argument("--fix-e1", action="store_true")
    parser.add_argument(
        "--open-facilities",
        type=str,
        default=None,
        help="Comma-separated manufacturing facilities. By default all facilities remain in the model "
             "and E1 is fixed exactly to this set; use --allow-drop-fixed-facilities to treat this "
             "as an availability set, or --prune-fixed-facilities to build the reduced exact model.",
    )
    parser.add_argument(
        "--e1-mask",
        type=str,
        default=None,
        help="Comma-separated 0/1 mask over the original facility order in the .dat file. "
             "By default all facilities remain in the model and E1 is fixed exactly to this mask; "
             "use --allow-drop-fixed-facilities to treat 1s as available facilities, or "
             "--prune-fixed-facilities to build the reduced exact model.",
    )
    parser.add_argument(
        "--allow-drop-fixed-facilities",
        action="store_true",
        help="When --open-facilities or --e1-mask is used, apply E1 <= mask instead of E1 == mask. "
             "Predicted-open facilities may then be closed by the detailed model.",
    )
    parser.add_argument(
        "--prune-fixed-facilities",
        action="store_true",
        help="When --open-facilities or --e1-mask is used, remove closed facilities before model build. "
             "The remaining facilities are fixed open exactly.",
    )
    parser.add_argument("--tee", action="store_true")
    parser.add_argument("--metrics-json", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.summary_only:
        data = parse_patient_dat(args.dat, time_horizon=args.time_horizon)
        print(summarise_data(data))
        return

    max_open_facilities = None if int(args.max_open_facilities) < 0 else int(args.max_open_facilities)
    metrics = solve_detailed_gurobi_dat(
        args.dat,
        time_horizon=args.time_horizon,
        turnaround_limit=args.turnaround_limit,
        max_open_facilities=max_open_facilities,
        solver_threads=args.threads,
        time_limit=args.time_limit,
        mip_gap=args.mip_gap,
        solver_options=_parse_solver_options(args.solver_option),
        fix_e1=args.fix_e1,
        fixed_open_facilities=_parse_facility_list(args.open_facilities),
        fixed_e1_mask=(
            [int(tok.strip()) for tok in str(args.e1_mask).split(",") if tok.strip()]
            if args.e1_mask is not None
            else None
        ),
        prune_fixed_facilities=bool(args.prune_fixed_facilities),
        allow_drop_fixed_facilities=bool(args.allow_drop_fixed_facilities),
        tee=args.tee,
    )
    print(metrics["summary"])
    print(f"[METRIC] build_time_sec={metrics['build_time_sec']:.3f}")
    solve = metrics.get("solve", {})
    print(f"[METRIC] solve_time_sec={float(solve.get('solve_time_sec', 0.0)):.3f}")
    print(f"[METRIC] termination_condition={solve.get('termination_condition')}")
    if "objective_value" in solve:
        print(f"[METRIC] objective_value={float(solve['objective_value']):.6f}")
    if "solver_gap" in solve:
        print(f"[METRIC] solver_gap={float(solve['solver_gap'])}")
    if "E1_solution" in solve:
        print(f"[METRIC] E1_solution={solve['E1_solution']}")
    if args.metrics_json is not None:
        args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
        with args.metrics_json.open("w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)
            fh.write("\n")
        print(f"[INFO] Metrics written to {args.metrics_json.as_posix()}")


if __name__ == "__main__":
    main()
