"""Split QKP pools while retaining reference optimality provenance."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


def _read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _ensure_clean_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"{path.as_posix()} already exists. Pass --overwrite to replace it."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _copy_npz_files(
    source_dir: Path,
    target_dir: Path,
    instance_rows: Sequence[Dict],
) -> None:
    for row in instance_rows:
        inst_id = int(row["instance_id"])
        npz_name = str(row.get("npz_path") or f"instance_{inst_id:04d}.npz")
        src = source_dir / npz_name
        if not src.exists():
            raise FileNotFoundError(f"Missing instance file: {src.as_posix()}")
        shutil.copy2(src, target_dir / npz_name)


def _count_samples(samples_rows: Sequence[Dict], instance_ids: set[int]) -> int:
    return sum(1 for row in samples_rows if int(row["instance_id"]) in instance_ids)


def _filter_samples(samples_rows: Sequence[Dict], instance_ids: set[int]) -> List[Dict]:
    return [row for row in samples_rows if int(row["instance_id"]) in instance_ids]


def _validate_fracs(train_frac: float, val_frac: float, test_frac: float) -> None:
    total = float(train_frac) + float(val_frac) + float(test_frac)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            f"train/val/test fractions must sum to 1.0, got {total:.6f}"
        )


def _split_instance_ids(
    instance_ids: Sequence[int],
    val_frac: float,
    test_frac: float,
    seed: int,
    certified_ids: Sequence[int],
) -> tuple[List[int], List[int], List[int]]:
    rng = random.Random(seed)
    ids = sorted(instance_ids)
    rng.shuffle(ids)

    n_total = len(ids)
    n_val = int(n_total * float(val_frac))
    n_test = int(n_total * float(test_frac))

    eligible = set(certified_ids)
    test_ids = [inst_id for inst_id in ids if inst_id in eligible][:n_test]
    if len(test_ids) < n_test:
        raise ValueError(
            f"Need {n_test} certified optimal QKP references for testing, "
            f"but found only {len(test_ids)}. Generate more certified instances "
            "or reduce the test fraction."
        )
    test_set = set(test_ids)
    remaining = [inst_id for inst_id in ids if inst_id not in test_set]
    val_ids = remaining[:n_val]
    train_ids = remaining[n_val:]
    return train_ids, val_ids, test_ids


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Split all generated QKP instances into seeded train/val/test folders "
            "with certified optimal test references, preserving all solver metadata."
        )
    )
    parser.add_argument("--input-dir", "--input_dir", type=str, required=True)
    parser.add_argument("--output-dir", "--output_dir", type=str, required=True)
    parser.add_argument("--train-frac", "--train_frac", type=float, default=0.8)
    parser.add_argument("--val-frac", "--val_frac", type=float, default=0.1)
    parser.add_argument("--test-frac", "--test_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    _validate_fracs(args.train_frac, args.val_frac, args.test_frac)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    source, target = input_dir.resolve(), output_dir.resolve()
    if source == target or source in target.parents or target in source.parents:
        raise ValueError("Source and output must be separate, non-nested directories.")
    instances_path = input_dir / "instances.jsonl"
    samples_path = input_dir / "samples.jsonl"

    if not instances_path.exists():
        raise FileNotFoundError(f"Missing {instances_path.as_posix()}")
    if not samples_path.exists():
        raise FileNotFoundError(f"Missing {samples_path.as_posix()}")

    instance_rows = sorted(_read_jsonl(instances_path), key=lambda row: int(row["instance_id"]))
    if len({int(row["instance_id"]) for row in instance_rows}) != len(instance_rows):
        raise ValueError("Duplicate QKP instance IDs.")
    sample_rows = _read_jsonl(samples_path)

    main_rows = instance_rows
    if not main_rows:
        raise RuntimeError("No QKP instances found; cannot create train/val/test.")

    main_ids = [int(row["instance_id"]) for row in main_rows]
    train_ids, val_ids, test_ids = _split_instance_ids(
        main_ids,
        val_frac=float(args.val_frac),
        test_frac=float(args.test_frac),
        seed=int(args.seed),
        certified_ids=[int(row["instance_id"]) for row in main_rows if bool(row.get("optimal", False))],
    )

    split_to_rows = {
        "train": [row for row in main_rows if int(row["instance_id"]) in set(train_ids)],
        "val": [row for row in main_rows if int(row["instance_id"]) in set(val_ids)],
        "test": [row for row in main_rows if int(row["instance_id"]) in set(test_ids)],
    }

    summary: Dict[str, Dict[str, int]] = {}
    for split_name, rows in split_to_rows.items():
        split_dir = output_dir / split_name
        _ensure_clean_dir(split_dir, overwrite=bool(args.overwrite))

        split_ids = {int(row["instance_id"]) for row in rows}
        split_samples = _filter_samples(sample_rows, split_ids)

        _write_jsonl(split_dir / "instances.jsonl", rows)
        _write_jsonl(split_dir / "samples.jsonl", split_samples)
        _copy_npz_files(input_dir, split_dir, rows)

        summary[split_name] = {
            "instances": len(rows),
            "samples": len(split_samples),
            "instance_ids": sorted(split_ids),
        }

    if (input_dir / "generation_config.json").exists():
        shutil.copy2(input_dir / "generation_config.json", output_dir / "source_generation_config.json")
    summary_payload = {
        "input_dir": input_dir.as_posix(),
        "output_dir": output_dir.as_posix(),
        "seed": int(args.seed),
        "train_frac": float(args.train_frac),
        "val_frac": float(args.val_frac),
        "test_frac": float(args.test_frac),
        "certified_optimal_instances": sum(bool(row.get("optimal", False)) for row in instance_rows),
        "main_split_instances": len(main_rows),
        "test_reference_policy": "certified_optimal",
        "splits": summary,
    }
    with (output_dir / "split_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary_payload, fh, indent=2)
        fh.write("\n")

    print(
        "[split_qkp_dataset] "
        f"instances={len(instance_rows)} "
        f"train={summary['train']['instances']} "
        f"val={summary['val']['instances']} "
        f"test={summary['test']['instances']}"
    )


if __name__ == "__main__":
    main()
