"""Create CART16 sampled mask labels from existing detailed patient inputs."""

import argparse
import hashlib
import json
import math
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

from cart_dataset.model_cart_detailed_gurobi import load_detailed_data
from cart_dataset.candidate_scoring import is_complete, label_candidate_results, solve_mask


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_records(path, records, jsonl=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            if jsonl:
                for row in records:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
            else:
                json.dump(records, stream, indent=2, allow_nan=False)
                stream.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _instance_id(path):
    return int(Path(path).stem.removeprefix("instance_"))


def load_references(path):
    """Accept imported detailed summaries or unrestricted cluster metrics."""
    if path is None:
        return {}
    path = Path(path)
    paths = sorted(path.rglob("instance_*.json")) if path.is_dir() else [path]
    if not paths:
        raise ValueError(f"No instance metrics found in {path}")
    references = {}
    for record_path in paths:
        record = _read_json(record_path)
        if "changes" in record:
            entries = [dict(change, max_open_facilities=None) for change in record["changes"]]
        elif "solve" in record and "instance" in record:
            solve = record["solve"]
            instance = record["instance"]
            if solve.get("termination_condition") != "optimal":
                if path.is_dir():
                    continue
                raise ValueError(f"Reference is not optimal: {record_path}")
            if (
                instance.get("fixed_e1")
                or instance.get("conpas_hamming_constraint")
                or instance.get("conpas_rank_neighbourhood")
                or instance.get("e1_upper_bounds")
                or solve.get("fixed_open_facilities") is not None
                or solve.get("available_facilities") is not None
            ):
                raise ValueError(f"Expected an unrestricted reference: {record_path}")
            entries = [{
                "instance_id": _instance_id(record.get("dat_name", record_path.stem)),
                "objective": solve.get("objective_value"),
                "e1": solve.get("E1_solution"),
                "best_bound": solve.get("best_bound"),
                "solver_gap": solve.get("solver_gap"),
                "horizon": instance.get("horizon"),
                "turnaround_limit": record.get("turnaround_limit", instance.get("turnaround_limit", 18)),
                "patients": instance.get("patients"),
                "max_open_facilities": instance.get("max_open_facilities"),
                "input_sha256": record.get("input_sha256"),
                "dat": record.get("dat"),
            }]
        else:
            raise ValueError(f"Unrecognized detailed reference: {record_path}")
        for entry in entries:
            instance_id = int(entry["instance_id"])
            if instance_id in references:
                raise ValueError(f"Duplicate detailed reference for instance {instance_id}")
            references[instance_id] = dict(entry, source=str(record_path))
    if not references and not path.is_dir():
        raise ValueError(f"No detailed references found in {path}")
    return references


def validate_reference(reference, data, input_digest, time_horizon, max_open_facilities, turnaround_limit=17):
    if reference is None:
        return None
    if float(reference.get("turnaround_limit", 18)) != float(turnaround_limit):
        raise ValueError("Reference turnaround differs from --turnaround-limit.")
    facilities = list(data["m"][None])
    if int(reference["horizon"]) != time_horizon:
        raise ValueError("Reference horizon differs from --time-horizon.")
    if reference.get("max_open_facilities") != max_open_facilities:
        raise ValueError("Reference facility-count bound differs from the requested model.")
    if int(reference["patients"]) != len(data["p"][None]):
        raise ValueError("Reference patient count differs from the input.")
    e1 = reference.get("e1") or {}
    if set(e1) != set(facilities) or not any(e1.values()) or any(bit not in (0, 1) for bit in e1.values()):
        raise ValueError("Reference has an invalid facility mask.")
    expected_digest = reference.get("input_sha256")
    if not expected_digest:
        original_dat = Path(reference.get("dat") or "")
        if not original_dat.is_file():
            raise ValueError(
                "Reference needs input_sha256 or a readable original dat path "
                "to verify that the patient input is identical."
            )
        expected_digest = _digest(original_dat)
    if expected_digest != input_digest:
        raise ValueError("Reference was solved on different patient input data.")
    objective = float(reference["objective"])
    bound = float(reference["best_bound"])
    if not math.isfinite(objective) or not math.isfinite(bound):
        raise ValueError("Reference objective and bound must be finite.")
    if bound > objective + 1e-7 * max(abs(objective), 1.0):
        raise ValueError("Reference minimization bound exceeds its objective.")
    gap = abs(objective - bound) / max(abs(objective), 1e-9)
    reported_gap = reference.get("solver_gap")
    if reported_gap is not None and not math.isclose(
        float(reported_gap), gap, rel_tol=1e-5, abs_tol=1e-9
    ):
        raise ValueError("Reference solver gap disagrees with its objective and bound.")
    return dict(reference, objective=objective, best_bound=bound, solver_gap=gap)


def _result_path(out_dir, instance_id, mask):
    bits = "".join(str(int(bit)) for bit in mask)
    return Path(out_dir) / f"instance_{instance_id}" / f"mask_{bits}.json"


def _signature(data, input_digest, time_horizon, max_open_facilities, mask, turnaround_limit=17):
    return {
        "input_sha256": input_digest,
        "time_horizon": time_horizon,
        "turnaround_limit": float(turnaround_limit),
        "max_open_facilities": max_open_facilities,
        "facility_order": list(data["m"][None]),
        "mask": list(mask),
        "objective_model": "cart_detailed",
        "model_sha256": _digest(Path(__file__).resolve().parents[1] / "cart_dataset/model_cart_detailed_gurobi.py"),
    }


def _check_cached(record, signature):
    if record.get("signature") != signature:
        raise ValueError("Cached mask belongs to different input/model settings; use a new output directory.")


def _reference_result(reference):
    return {
        "status": "optimal", "feasible": True, "proven_optimal": True,
        "objective": reference["objective"], "best_bound": reference["best_bound"],
        "mip_gap_rel": reference["solver_gap"], "source": "detailed_reference",
        "reference_source": reference["source"],
        "build_time_sec": 0.0, "solver_time_sec": 0.0,
    }


def _cart16_context(data, instance_id, args):
    from .config import FACILITY_UTIL_CAP
    from .dataset_maker_sampled import _compute_peak_active_demand, _split_seed_offset
    facilities, clinics = list(data["m"][None]), list(data["c"][None])
    if len(facilities) != 16:
        raise ValueError("CART16 patient inputs must contain exactly 16 facilities.")
    source = args.input_source or args.dat_dir.parent / args.dat_dir.name.removesuffix("_patient_dats")
    static = _read_json(source / "static_config.json")
    metadata = _load_metadata(args.dat_dir, source).get(instance_id)
    if metadata is None:
        raise ValueError(f"Missing input metadata for instance {instance_id} in {source}.")
    if static["M"] != facilities or static["C"] != clinics:
        raise ValueError("Dataset metadata has different clinic/facility ordering.")
    capacities = static.get("CAPMAX", static.get("FCAP"))
    for name, values in (("CIM", static["CIM"]), ("CVM", static["CVM"]), ("CAPMAX", capacities)):
        if values is None or any(float(values[m]) != float(data[name][m]) for m in facilities):
            raise ValueError(f"Dataset static parameter {name} differs from patient inputs.")
    if any(float(static["U1"][c][m]) != float(data["U1"][(c, m, "j1")]) for c in clinics for m in facilities):
        raise ValueError("Dataset transport costs differ from patient inputs.")
    inc = {c: {} for c in clinics}
    for (_, clinic, day), value in data["INC"].items():
        if value:
            key = str(int(day))
            inc[clinic][key] = inc[clinic].get(key, 0) + int(value)
    recorded = {c: {str(int(day)): int(value) for day, value in days.items() if int(value)}
                for c, days in metadata["inc"].items()}
    if inc != recorded or int(metadata["total_therapies"]) != len(data["p"][None]):
        raise ValueError("Dataset arrival metadata differs from patient inputs.")
    tmfe = int(static.get("TMFE", 7))
    if tmfe != int(data["TMFE"][None]):
        raise ValueError("Dataset manufacturing duration differs from patient inputs.")
    static.update(FCAP=capacities, time_horizon=args.time_horizon,
                  turnaround_limit=args.turnaround_limit, formulation="detailed")
    static.setdefault("util_cap", FACILITY_UTIL_CAP)
    split = args.split or args.dat_dir.name.removesuffix("_patient_dats")
    seed = args.sampling_seed + len(facilities) * 100_000 + _split_seed_offset(split) + instance_id
    peak = _compute_peak_active_demand(inc, args.time_horizon, tmfe)
    return static, metadata, inc, seed, peak


def _cart16_plan(data, instance_id, input_digest, args):
    from .config import FACILITY_UTIL_CAP, TIME_HORIZON
    from .dataset_maker_sampled import DetailedCartScorer, _propose_candidates
    static, metadata, inc, seed, peak = _cart16_context(data, instance_id, args)
    signature = _signature(data, input_digest, args.time_horizon, None, [], args.turnaround_limit)
    signature.update(strategy="cart16_generator", source_policy="original_cart16", seed=seed,
                     candidates_per_instance=args.candidates_per_instance,
                     oversample_factor=args.oversample_factor,
                     util_cap=float(static.get("util_cap", FACILITY_UTIL_CAP)),
                     demand_horizon=int(static.get("demand_horizon", TIME_HORIZON)),
                     max_positives=args.max_positives, infeasible_fraction=args.infeasible_fraction,
                     tie_tolerance=args.tie_tolerance,
                     sampling_model_sha256=_digest(Path(__file__).with_name("dataset_maker_sampled.py")))
    path = args.out_dir / f"instance_{instance_id}/candidate_plan.json"
    if path.exists():
        plan = _read_json(path)
        _check_cached(plan, signature)
        if not is_complete(plan["reference"], args.mip_gap):
            raise ValueError("Saved reference does not satisfy --mip-gap; use a new output directory.")
        return plan
    if not args.prepare_plans_only:
        raise ValueError(f"Missing candidate plan {path}; run --prepare-plans-only first.")
    scorer = DetailedCartScorer(static, inc, seed=seed, threads=args.threads,
                               full_time_limit=args.full_time_limit,
                               fixed_time_limit=args.time_limit, mip_gap=args.mip_gap)
    # Preserve the exact patient inputs, including all detailed transport parameters.
    scorer.data = data
    best, _, _, full_optimal, runtime, incumbents = scorer.solve_full()
    if not full_optimal:
        raise ValueError("Detailed reference is unresolved; increase --full-time-limit and retry preparation.")
    target = max(args.candidates_per_instance * args.oversample_factor,
                 args.candidates_per_instance + 64)
    proposed = _propose_candidates(static, inc, best, incumbents, peak, target, seed + 17)
    plan = {"signature": signature, "reference_mask": best.tolist(),
            "reference": scorer.reference_result, "full_runtime": runtime,
            "masks": [list(mask) for mask in proposed],
            "sources": ["optimal" if source == "reference" else source for source in proposed.values()],
            "peak_active_demand": peak, "metadata": metadata}
    _write_records(path, plan)
    return plan


def candidate_masks(data, instance_id, input_digest, args, reference=None):
    """Persist sampled proposals so solving and collection use identical masks."""
    count = getattr(args, "sampled_candidates", None)
    facilities = list(data["m"][None])
    if getattr(args, "cart16_pool", False):
        return _cart16_plan(data, instance_id, input_digest, args)["masks"]
    if count is None:
        raise ValueError("Use the CART16 candidate pool or supply an explicit sampled pool size.")
    if count > 2 ** len(facilities):
        raise ValueError("--sampled-candidates exceeds the number of binary masks.")
    signature = _signature(data, input_digest, args.time_horizon,
                           args.max_open_facilities, [], args.turnaround_limit)
    signature.update(sampled_candidates=count, sampling_seed=args.sampling_seed)
    path = args.out_dir / f"instance_{instance_id}/candidate_plan.json"
    if path.exists():
        plan = _read_json(path)
        _check_cached(plan, signature)
        if reference is not None and plan["reference_mask"] != [reference["e1"][m] for m in facilities]:
            raise ValueError("Reference mask differs from the saved candidate plan.")
        return plan["masks"]
    if args.collect_only:
        raise ValueError(f"Missing sampled candidate plan: {path}")
    if reference is None or reference["solver_gap"] > args.mip_gap + 1e-12:
        raise ValueError("Sampled pools require a matching completed reference via --references.")
    import numpy as np
    from .dataset_maker_sampled import _compute_peak_active_demand, _propose_candidates

    clinics = list(data["c"][None])
    static = {"M": facilities, "C": clinics, "FCAP": data["CAPMAX"],
              "CIM": data["CIM"], "CVM": data["CVM"], "util_cap": 1.0,
              "U1": {c: {m: data["U1"][(c, m, "j1")] for m in facilities} for c in clinics}}
    inc = {c: {} for c in clinics}
    for (_, clinic, day), value in data["INC"].items():
        if value:
            inc[clinic][int(day)] = inc[clinic].get(int(day), 0) + int(value)
    best = [reference["e1"][m] for m in facilities]
    peak = _compute_peak_active_demand(inc, args.time_horizon, int(data["TMFE"][None]))
    proposed = _propose_candidates(static, inc, np.asarray(best, dtype=np.int8), [],
                                   peak, count, args.sampling_seed + instance_id)
    # The existing proposer may exceed its target while adding mandatory heuristics.
    masks = [list(mask) for mask in proposed]
    _write_records(path, {"signature": signature, "reference_mask": best, "masks": masks,
                          "sources": list(proposed.values())})
    return masks


def collect_instance(data, instance_id, out_dir, input_digest, args, metadata=None):
    facilities = list(data["m"][None])
    masks = candidate_masks(data, instance_id, input_digest, args)
    results = []
    missing = []
    unresolved = []
    for mask in masks:
        path = _result_path(out_dir, instance_id, mask)
        bits = "".join(str(bit) for bit in mask)
        if not path.exists():
            missing.append(bits)
            continue
        record = _read_json(path)
        _check_cached(record, _signature(data, input_digest, args.time_horizon, args.max_open_facilities, mask, args.turnaround_limit))
        if not is_complete(record, args.mip_gap):
            unresolved.append(bits)
        results.append(record)
    complete = not missing and not unresolved
    summary = {
        "instance_id": instance_id, "candidate_count": len(masks),
        "recorded_masks": len(results), "complete": complete,
        "missing_masks": missing, "unresolved_masks": unresolved,
        "objective_model": "cart_detailed",
        "turnaround_limit": float(args.turnaround_limit), "formulation": "detailed",
    }
    rows = []
    if complete:
        objectives = [r["objective"] for r in results if r["feasible"]]
        best = min(objectives) if objectives else None
        summary["best_objective"] = best
        metadata = metadata or {}
        labeled = label_candidate_results(results, mip_gap=args.mip_gap, tie_tolerance=args.tie_tolerance)
        for result in labeled:
            feasible = result["feasible"]
            objective = result["objective"]
            rows.append({
                "instance_id": instance_id,
                "label_type": result["label_type"],
                "feasible": feasible, "objective": objective,
                "objective_model": "cart_detailed", "status": result["status"],
                "turnaround_limit": float(args.turnaround_limit), "formulation": "detailed",
                "e1": dict(zip(facilities, result["signature"]["mask"])),
                "optimal": result["optimal"], "fixed_mask_optimal": result["proven_optimal"],
                "regret_abs": result["regret_abs"],
                "regret_rel": result["regret_rel"],
                "mip_gap_rel": result["mip_gap_rel"], "best_bound": result["best_bound"],
                "seed_demand": metadata.get("seed_demand", 0), "seed_solver": None,
                "total_therapies": len(data["p"][None]),
                "demand_kwargs": metadata.get("demand_kwargs", {}),
                "extras": {"input_sha256": input_digest, "time_horizon": args.time_horizon},
            })
        if args.cart16_pool:
            import numpy as np
            from .config import FACILITY_UTIL_CAP, TIME_HORIZON
            from .dataset_maker_sampled import _sample_to_row, _select_training_pool
            plan = _cart16_plan(data, instance_id, input_digest, args)
            static, metadata, _, _, peak = _cart16_context(data, instance_id, args)
            best_result = min((r for r in labeled if r["feasible"]), key=lambda r: r["objective"])
            best_mask = np.asarray(best_result["signature"]["mask"], dtype=np.int8)
            rows = []
            for result, mask, source in zip(labeled, plan["masks"], plan["sources"]):
                capacities = np.asarray([float(static["FCAP"][m]) for m in facilities])
                capacity = float(np.dot(capacities * float(static.get("util_cap", FACILITY_UTIL_CAP)), mask))
                scored = dict(result, mask=mask, solver_source=result["source"], source=source,
                              capacity=capacity, capacity_heuristic_violation=max(0, peak - capacity) / max(peak, 1))
                row = _sample_to_row(instance_id, scored, best_mask, facilities, metadata,
                                     plan["reference"], True, args.time_horizon,
                                     int(static.get("demand_horizon", TIME_HORIZON)))
                row.update(formulation="detailed", turnaround_limit=float(args.turnaround_limit))
                row["extras"]["peak_active_demand"] = peak
                rows.append(row)
            rows = _select_training_pool(rows, args.candidates_per_instance,
                                         args.max_positives, args.infeasible_fraction)
            summary.update(samples=len(rows), positive=sum(r["label_type"] == "positive" for r in rows),
                           feasible=sum(r["feasible"] for r in rows))
        _write_records(Path(out_dir) / f"instance_{instance_id}/samples.jsonl", rows, jsonl=True)
    _write_records(Path(out_dir) / f"instance_{instance_id}/summary.json", summary)
    return summary, rows


def _load_metadata(dat_dir, dataset_source=None):
    name = dat_dir.name.removesuffix("_patient_dats")
    path = (Path(dataset_source) if dataset_source is not None else dat_dir.parent / name) / "instances_meta.jsonl"
    if not path.exists():
        return {}
    return {int(row["instance_id"]): row for row in
            (json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip())}


def export_dataset_metadata(source_dir, dat_files, args):
    """Verify source metadata before publishing a directly trainable pool."""
    source_dir = Path(source_dir)
    config = _read_json(source_dir / "static_config.json")
    metadata = [json.loads(line) for line in
                (source_dir / "instances_meta.jsonl").read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    by_id = {int(row["instance_id"]): row for row in metadata}
    if len(by_id) != len(metadata) or set(by_id) != {_instance_id(path) for path in dat_files}:
        raise ValueError("Dataset metadata IDs differ from the complete patient input pool.")
    capacities = config.get("CAPMAX", config.get("FCAP"))
    for path in dat_files:
        data = load_detailed_data(path, args.time_horizon)
        facilities, clinics = list(data["m"][None]), list(data["c"][None])
        if config["M"] != facilities or config["C"] != clinics:
            raise ValueError("Dataset metadata has different clinic/facility ordering.")
        for name, values in (("CIM", config["CIM"]), ("CVM", config["CVM"]), ("CAPMAX", capacities)):
            if values is None or any(float(values[m]) != float(data[name][m]) for m in facilities):
                raise ValueError(f"Dataset static parameter {name} differs from patient inputs.")
        if any(float(config["U1"][c][m]) != float(data["U1"][(c, m, "j1")]) for c in clinics for m in facilities):
            raise ValueError("Dataset transport costs differ from patient inputs.")
        if float(config.get("TMFE", 7)) != float(data["TMFE"][None]):
            raise ValueError("Dataset manufacturing duration differs from patient inputs.")
        actual = {}
        for (_, clinic, day), value in data["INC"].items():
            if value:
                key = (str(clinic), int(day))
                actual[key] = actual.get(key, 0) + int(value)
        recorded = {(str(c), int(day)): int(value) for c, days in by_id[_instance_id(path)]["inc"].items()
                    for day, value in days.items() if int(value)}
        if actual != recorded:
            raise ValueError("Dataset arrival metadata differs from patient inputs.")
    config.update(objective_model="cart_detailed", time_horizon=args.time_horizon,
                  max_facilities=args.max_open_facilities,
                  turnaround_limit=float(args.turnaround_limit), formulation="detailed")
    if args.cart16_pool:
        config["candidate_generation"] = {
            "strategy": "sampled_detailed_cart_pool", "candidates_per_instance": args.candidates_per_instance,
            "oversample_factor": args.oversample_factor, "max_positives": args.max_positives,
            "pos_threshold": args.tie_tolerance, "infeasible_fraction": args.infeasible_fraction,
            "seed": args.sampling_seed,
        }
    for row in metadata:
        row.update(objective_model="cart_detailed", time_horizon=args.time_horizon,
                   turnaround_limit=float(args.turnaround_limit), formulation="detailed")
    _write_records(args.out_dir / "static_config.json", config)
    _write_records(args.out_dir / "instances_meta.jsonl", metadata, jsonl=True)


def _process_instance(task):
    dat_path, args, raw_reference, metadata, expected_facilities = task
    instance_id = _instance_id(dat_path)
    input_digest = _digest(dat_path)
    data = load_detailed_data(dat_path, args.time_horizon)
    facilities = list(data["m"][None])
    if not data["p"][None]:
        raise ValueError("Patient inputs must contain at least one patient.")
    if expected_facilities is not None and facilities != expected_facilities:
        raise ValueError("Flattened tasks require identical facility orders across inputs.")
    if any(not 1 <= int(key[2]) <= args.time_horizon for key in data["INC"]):
        raise ValueError("Patient arrival lies outside the requested horizon.")
    reference = validate_reference(
        raw_reference, data, input_digest, args.time_horizon, args.max_open_facilities, args.turnaround_limit
    )
    masks = candidate_masks(data, instance_id, input_digest, args, reference)
    if args.cart16_pool:
        plan = _cart16_plan(data, instance_id, input_digest, args)
        if args.prepare_plans_only:
            return {"instance_id": instance_id, "complete": True, "candidate_count": len(masks),
                    "stage": "prepared"}, []
        reference = dict(plan["reference"], e1=dict(zip(facilities, plan["reference_mask"])),
                         solver_gap=plan["reference"]["mip_gap_rel"], source="candidate_plan")
    if args.mask_index is not None and not 0 <= args.mask_index < len(masks):
        raise ValueError("--mask-index is outside the candidate list.")
    chosen = masks if args.mask_index is None else [masks[args.mask_index]]
    chosen = chosen[args.mask_shard_index::args.num_mask_shards]
    selected_results = []
    if not args.collect_only:
        for mask in chosen:
            path = _result_path(args.out_dir, instance_id, mask)
            signature = _signature(data, input_digest, args.time_horizon, args.max_open_facilities, mask, args.turnaround_limit)
            if path.exists():
                cached = _read_json(path)
                _check_cached(cached, signature)
                if is_complete(cached, args.mip_gap):
                    print(f"instance={instance_id} mask={mask} cached", flush=True)
                    selected_results.append(cached)
                    continue
            if (
                reference is not None
                and mask == [reference["e1"][m] for m in facilities]
                and reference["solver_gap"] <= args.mip_gap + 1e-12
            ):
                result = dict(plan["reference"]) if args.cart16_pool else _reference_result(reference)
            else:
                print(f"instance={instance_id} mask={mask} building {sum(mask)}/{len(facilities)} facilities", flush=True)
                result = solve_mask(
                    data, mask, time_horizon=args.time_horizon,
                    max_open_facilities=args.max_open_facilities, time_limit=args.time_limit,
                    threads=args.threads, mip_gap=args.mip_gap, tee=args.tee,
                    turnaround_limit=args.turnaround_limit,
                )
            record = dict(result, signature=signature, instance_id=instance_id,
                          turnaround_limit=float(args.turnaround_limit), formulation="detailed",
                          open_facilities=[m for m, bit in zip(facilities, mask) if bit])
            _write_records(path, record)
            selected_results.append(record)
            print(f"instance={instance_id} mask={mask} status={result['status']} objective={result['objective']}", flush=True)
    if args.mask_index is None and args.num_mask_shards == 1:
        summary, rows = collect_instance(data, instance_id, args.out_dir, input_digest, args, metadata)
    else:
        summary = {
            "instance_id": instance_id, "candidate_count": len(chosen),
            "recorded_masks": len(selected_results),
            "complete": all(is_complete(result, args.mip_gap) for result in selected_results),
            "selected_mask_index": args.mask_index,
            "unresolved_masks": [
                "".join(str(bit) for bit in result["signature"]["mask"])
                for result in selected_results if not is_complete(result, args.mip_gap)
            ],
            "objective_model": "cart_detailed",
            "turnaround_limit": float(args.turnaround_limit), "formulation": "detailed",
        }
        rows = []
    print(json.dumps(summary), flush=True)
    return summary, rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dat-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--references", type=Path, help="Detailed reference summary JSON or cluster metrics directory.")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--instance-id", type=int, nargs="+")
    selection.add_argument("--instance-index", type=int, help="Zero-based index into numerically sorted instance files.")
    parser.add_argument("--mask-index", type=int, help="Evaluate just this candidate index, for a pilot or array job.")
    parser.add_argument("--time-horizon", type=int, default=130)
    parser.add_argument("--turnaround-limit", type=float, default=17.0)
    parser.add_argument("--max-open-facilities", type=int, default=None)
    parser.add_argument("--time-limit", type=float, default=3600)
    parser.add_argument("--threads", "--solver-threads", dest="threads", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1, help="Concurrent instance processes; masks within each instance run sequentially.")
    parser.add_argument("--num-shards", type=int, default=1, help="Round-robin partition of numerically sorted instances.")
    parser.add_argument("--shard-index", type=int, default=0, help="Zero-based shard index.")
    parser.add_argument("--sampled-candidates", type=int, help="Sample proposals around a completed reference. Heuristics may exceed this target.")
    parser.add_argument("--sampling-seed", type=int, default=515151)
    parser.add_argument("--num-mask-shards", type=int, default=1, help="Partition each instance's candidates across jobs.")
    parser.add_argument("--mask-shard-index", type=int, default=0)
    parser.add_argument("--cart16-pool", action="store_true", help="Use the default CART16 proposal and selection workflow explicitly.")
    parser.add_argument("--input-source", type=Path, help="Existing split directory containing static_config.json and instances_meta.jsonl.")
    parser.add_argument("--split", choices=("train", "val", "test"))
    parser.add_argument("--candidates-per-instance", type=int, default=160)
    parser.add_argument("--oversample-factor", type=int, default=2)
    parser.add_argument("--max-positives", type=int, default=3)
    parser.add_argument("--infeasible-fraction", type=float, default=0.30)
    parser.add_argument("--full-time-limit", type=float, default=86400)
    parser.add_argument("--prepare-plans-only", action="store_true")
    parser.add_argument("--summary-json", type=Path, help="Optional progress report; use a different path for each shard.")
    parser.add_argument("--mip-gap", type=float, default=1e-4)
    parser.add_argument("--tie-tolerance", type=float, default=1e-6, help="Relative cost tolerance for tied positives.")
    parser.add_argument("--collect-only", action="store_true", help="Collect completed mask files without solving.")
    parser.add_argument("--dataset-source", type=Path, help="On full collection, verify and copy this split's static config and arrival metadata.")
    parser.add_argument("--tee", action="store_true")
    args = parser.parse_args(argv)
    args.cart16_pool = args.cart16_pool or args.sampled_candidates is None
    if args.candidates_per_instance < 1 or args.oversample_factor < 1 or args.max_positives < 1:
        parser.error("Pool size, oversampling and maximum positives must be positive.")
    if not math.isfinite(args.infeasible_fraction) or not 0 <= args.infeasible_fraction <= 1:
        parser.error("--infeasible-fraction must be between zero and one.")
    if not math.isfinite(args.full_time_limit) or args.full_time_limit <= 0:
        parser.error("--full-time-limit must be finite and positive.")
    if args.prepare_plans_only and (not args.cart16_pool or args.collect_only or args.num_mask_shards != 1 or args.mask_index is not None):
        parser.error("Plan preparation requires --cart16-pool without collection or mask selection/shards.")
    if args.cart16_pool and any(value is not None for value in (args.sampled_candidates, args.references, args.max_open_facilities)):
        parser.error("--cart16-pool uses its own reference solves and cannot use sampled-candidates, references or facility bounds.")
    if args.time_horizon <= 0 or args.time_limit <= 0 or args.threads <= 0 or args.workers <= 0 or args.num_shards <= 0:
        parser.error("Horizon, time limit, threads, workers and shard count must be positive.")
    if not math.isfinite(args.turnaround_limit) or args.turnaround_limit <= 0:
        parser.error("--turnaround-limit must be finite and positive.")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be between zero and --num-shards minus one.")
    if args.num_mask_shards <= 0 or not 0 <= args.mask_shard_index < args.num_mask_shards:
        parser.error("Mask shard count must be positive and its index must be in range.")
    if args.sampled_candidates is not None and args.sampled_candidates < 3:
        parser.error("--sampled-candidates must be at least three.")
    if args.sampling_seed < 0:
        parser.error("--sampling-seed must be nonnegative.")
    if args.num_mask_shards > 1 and (args.collect_only or args.mask_index is not None):
        parser.error("Mask shards cannot be combined with collection or single-mask selection.")
    if args.num_shards > 1 and any(value is not None for value in
                                 (args.instance_id, args.instance_index)):
        parser.error("Instance selection cannot be combined with multiple shards.")
    if args.mip_gap < 0 or args.tie_tolerance < 0 or args.max_open_facilities is not None and args.max_open_facilities < 0:
        parser.error("Gaps, tie tolerance and facility bound must be nonnegative.")
    if args.collect_only and args.mask_index is not None:
        parser.error("Collection requires complete instance pools, not a mask selection.")
    if args.dataset_source is not None:
        if not args.collect_only or args.num_shards != 1 or any(value is not None for value in
                                       (args.instance_id, args.instance_index, args.mask_index)):
            parser.error("--dataset-source requires a full-directory --collect-only run.")
        source, output = args.dataset_source.resolve(), args.out_dir.resolve()
        if source == output or source in output.parents or output in source.parents:
            parser.error("Dataset source and output must be separate, non-nested directories.")
    files = sorted(args.dat_dir.glob("instance_*.dat"), key=_instance_id)
    if not files:
        parser.error("No instance_*.dat files found.")
    if args.instance_id:
        requested = set(args.instance_id)
        if requested.difference(_instance_id(p) for p in files):
            parser.error("Requested instance ID was not found.")
        files = [p for p in files if _instance_id(p) in requested]
    if args.instance_index is not None:
        if not 0 <= args.instance_index < len(files):
            parser.error("--instance-index is outside the input file list.")
        files = [files[args.instance_index]]
    references = load_references(args.references)
    if args.references is not None:
        print(f"[references] loaded {len(references)} completed detailed optima", flush=True)
    metadata = _load_metadata(args.dat_dir, args.input_source or args.dataset_source)
    expected_facilities = None
    files = files[args.shard_index::args.num_shards]
    tasks = [
        (path, args, references.get(_instance_id(path)), metadata.get(_instance_id(path)), expected_facilities)
        for path in files
    ]
    active_workers = min(args.workers, len(tasks))
    print(f"[shard {args.shard_index}/{args.num_shards}] instances={len(tasks)} workers={active_workers} solver_threads={args.threads}", flush=True)
    all_rows = []
    summaries = []

    def record_output(output):
        summary, rows = output
        summaries.append(summary)
        all_rows.extend(rows)

    if active_workers <= 1:
        for task in tasks:
            record_output(_process_instance(task))
    else:
        # Spawn avoids inheriting native solver state in child processes.
        with ProcessPoolExecutor(max_workers=active_workers, mp_context=get_context("spawn")) as executor:
            futures = [executor.submit(_process_instance, task) for task in tasks]
            for future in as_completed(futures):
                record_output(future.result())
    summaries.sort(key=lambda item: item["instance_id"])
    all_rows.sort(key=lambda row: row["instance_id"])
    incomplete = any(not summary["complete"] for summary in summaries)
    if args.summary_json is not None:
        _write_records(args.summary_json, {
            "num_shards": args.num_shards, "shard_index": args.shard_index,
            "num_mask_shards": args.num_mask_shards, "mask_shard_index": args.mask_shard_index,
            "sampled_candidates": args.sampled_candidates, "sampling_seed": args.sampling_seed,
            "workers": args.workers, "active_workers": active_workers, "solver_threads": args.threads,
            "time_horizon": args.time_horizon, "time_limit_per_mask": args.time_limit,
            "mip_gap": args.mip_gap, "objective_model": "cart_detailed",
            "turnaround_limit": float(args.turnaround_limit), "formulation": "detailed",
            "selected_instance_ids": [_instance_id(path) for path in files],
            "complete_instance_ids": [item["instance_id"] for item in summaries if item["complete"]],
            "incomplete_instance_ids": [item["instance_id"] for item in summaries if not item["complete"]],
            "complete": not incomplete, "instances": summaries,
        })
    if args.collect_only and args.num_shards == 1 and args.instance_id is None and args.instance_index is None and not incomplete:
        if args.dataset_source is not None:
            export_dataset_metadata(args.dataset_source, files, args)
        _write_records(args.out_dir / "samples.jsonl", all_rows, jsonl=True)
    return 2 if incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
