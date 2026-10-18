"""Seeded synthetic facility expansion for sampled CART generation."""

from __future__ import annotations

from typing import Dict

import numpy as np

from .config import C_MATERIAL, C_QC, FACILITY_UTIL_CAP, TMFE


def _expand_facilities(
    static_sets: Dict[str, list],
    static_params: Dict[str, dict],
    target_facilities: int,
    seed: int,
) -> tuple[Dict[str, list], Dict[str, dict], Dict[str, object]]:
    clinics = [str(c) for c in static_sets["c"]]
    base_facilities = [str(m) for m in static_sets["m"]]
    target = int(target_facilities)
    if target < len(base_facilities):
        raise ValueError(
            f"target_facilities={target} is smaller than base facility count {len(base_facilities)}."
        )

    capacities = static_params.get("CAPMAX", static_params.get("FCAP"))
    if capacities is None:
        raise ValueError("Static parameters must define CAPMAX (or legacy FCAP).")
    rng = np.random.default_rng(int(seed) + 1009 * target)
    facilities = list(base_facilities)
    cim = {m: float(static_params["CIM"][m]) for m in base_facilities}
    cvm = {m: float(static_params["CVM"][m]) for m in base_facilities}
    fcap = {m: float(capacities[m]) for m in base_facilities}
    u1 = {(c, m): float(static_params["U1"][(c, m)]) for c in clinics for m in base_facilities}

    base_u1_by_facility = {
        m: np.asarray([float(static_params["U1"][(c, m)]) for c in clinics], dtype=float)
        for m in base_facilities
    }
    min_u1_by_clinic = {
        c: min(float(static_params["U1"][(c, m)]) for m in base_facilities)
        for c in clinics
    }

    for idx in range(len(base_facilities), target):
        base_m = base_facilities[idx % len(base_facilities)]
        focus_c = clinics[idx % len(clinics)]
        new_m = f"m{idx + 1}"
        facilities.append(new_m)

        cap_jitter = float(rng.uniform(0.85, 1.25))
        new_cap = max(1, int(round(float(capacities[base_m]) * cap_jitter)))
        fcap[new_m] = float(new_cap)

        cost_scale = (new_cap / max(float(capacities[base_m]), 1.0)) * float(rng.uniform(0.9, 1.15))
        cim[new_m] = float(static_params["CIM"][base_m]) * cost_scale
        cvm[new_m] = float(static_params["CVM"][base_m]) * cost_scale

        base_profile = base_u1_by_facility[base_m].copy()
        blend_profile = base_u1_by_facility[base_facilities[int(rng.integers(0, len(base_facilities)))]] * 0.35
        profile = 0.65 * base_profile + blend_profile
        profile *= rng.lognormal(mean=0.0, sigma=0.12, size=len(clinics))

        for c_idx, c in enumerate(clinics):
            if c == focus_c:
                profile[c_idx] = min(profile[c_idx], min_u1_by_clinic[c] * float(rng.uniform(0.75, 1.05)))
            # Keep the synthetic test in the original CART transport-cost range
            # so the primary shift is facility count, not a new distance scale.
            u1[(c, new_m)] = float(np.clip(profile[c_idx], 150.0, 3500.0))

    expanded_sets = {"c": clinics, "m": facilities}
    expanded_params = {"CIM": cim, "CVM": cvm, "FCAP": fcap, "U1": u1}
    metadata = {
        "base_facilities": len(base_facilities),
        "target_facilities": target,
        "synthetic_facilities": max(0, target - len(base_facilities)),
        "seed": int(seed),
    }
    return expanded_sets, expanded_params, metadata


def _static_config(static_sets: Dict[str, list], static_params: Dict[str, dict], metadata: Dict[str, object]) -> Dict[str, object]:
    clinics = [str(c) for c in static_sets["c"]]
    facilities = [str(m) for m in static_sets["m"]]
    capacities = static_params.get("CAPMAX", static_params.get("FCAP"))
    if capacities is None:
        raise ValueError("Static parameters must define CAPMAX (or legacy FCAP).")
    return {
        "C": clinics,
        "M": facilities,
        "TMFE": int(TMFE),
        "util_cap": float(FACILITY_UTIL_CAP),
        "material_qc_cost": float(C_MATERIAL + C_QC),
        "max_facilities": None,
        "synthetic_facility_metadata": metadata,
        "CIM": {m: float(static_params["CIM"][m]) for m in facilities},
        "CVM": {m: float(static_params["CVM"][m]) for m in facilities},
        "FCAP": {m: int(round(float(capacities[m]))) for m in facilities},
        "U1": {
            c: {m: float(static_params["U1"][(c, m)]) for m in facilities}
            for c in clinics
        },
    }
