"""Export existing CART demand as patient-level Pyomo inputs."""

import argparse
import json
import os
from pathlib import Path

from .dataset_maker import load_static_from_dat
from .model_cart_detailed import build_detailed_data_from_aggregate


def load_static_from_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    clinics, facilities = list(config["C"]), list(config["M"])
    capacities = config.get("CAPMAX", config.get("FCAP"))
    if capacities is None:
        raise ValueError("Static config must define CAPMAX (or legacy FCAP).")
    sets = {"c": clinics, "m": facilities}
    params = {name: {m: float(config[name][m]) for m in facilities} for name in ("CIM", "CVM")}
    params["CAPMAX"] = {m: float(capacities[m]) for m in facilities}
    params["U1"] = {(c, m): float(config["U1"][c][m]) for c in clinics for m in facilities}
    params["TMFE"] = int(config.get("TMFE", 7))
    return sets, params


def write_patient_dat(path, data, title=""):
    """Serialize the same parameter schema used by the detailed model builder."""
    lines = ["# Auto-generated detailed CART patient input", f"# {title}"]
    for name in ("c", "m", "h", "j", "p"):
        lines.append(f"set {name} := {' '.join(data[name][None])};")
    for name, entries in data.items():
        if name in ("c", "m", "h", "j", "p"):
            continue
        if None in entries:
            lines.append(f"param {name} := {entries[None]};")
        else:
            lines.append(f"param {name} :=")
            for key, value in entries.items():
                indices = key if isinstance(key, tuple) else (key,)
                lines.append(" ".join(map(str, (*indices, value))))
            lines.append(";")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--instances-json", type=Path)
    inputs.add_argument("--instances-meta-jsonl", type=Path)
    parser.add_argument("--static-config", type=Path)
    parser.add_argument("--base-dat", type=Path, default=Path(__file__).with_name("Data_CART_og.dat"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--time-horizon", type=int, default=130)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.time_horizon <= 0 or args.limit < 0:
        parser.error("Horizon must be positive and limit nonnegative.")
    static_path = args.static_config
    if static_path is None and args.instances_meta_jsonl is not None:
        candidate = args.instances_meta_jsonl.parent / "static_config.json"
        if candidate.is_file():
            static_path = candidate
    sets, params = load_static_from_config(static_path) if static_path else load_static_from_dat(args.base_dat)
    if args.instances_json:
        entries = json.loads(args.instances_json.read_text(encoding="utf-8-sig"))["instances"]
    else:
        entries = [json.loads(line) for line in args.instances_meta_jsonl.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if args.limit:
        entries = entries[:args.limit]
    ids = [int(entry.get("instance_id", entry.get("id"))) for entry in entries]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate instance IDs in the source metadata.")
    for instance_id in ids:
        target = args.out_dir / f"instance_{instance_id}.dat"
        if target.exists() and not args.overwrite:
            raise FileExistsError(f"{target} exists; use a new directory or --overwrite.")
    for instance_id, entry in zip(ids, entries):
        inc = {(str(c), int(day)): int(value) for c, days in entry["inc"].items() for day, value in days.items()}
        if any(value < 0 or c not in sets["c"] or not 1 <= day <= args.time_horizon for (c, day), value in inc.items()):
            raise ValueError(f"Instance {instance_id} has demand outside the detailed schema.")
        data = build_detailed_data_from_aggregate(sets, params, inc, time_horizon=args.time_horizon)
        target = args.out_dir / f"instance_{instance_id}.dat"
        write_patient_dat(target, data, title=f"Instance {instance_id}; horizon={args.time_horizon}")
        print(f"Wrote {target} ({len(data['p'][None])} patients)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
