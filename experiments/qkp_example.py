"""Optimize a pretrained QKP EnerGNN energy with Gurobi under knapsack constraints."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch

from gnn.data.dataset import InstancePoolDataset
from gnn.data.reader import load_instances_meta_qkp
from gnn.inference.checkpoint import load_trained_model
from qkp_dataset.dataset_maker import qkp_objective


def default_example_dir():
    candidates = (
        Path(__file__).resolve().parents[1] / "examples/qkp",
        Path(sys.prefix) / "examples/qkp",
    )
    return next((path for path in candidates if (path / "manifest.json").is_file()), candidates[0])


def solve_examples(example_dir, checkpoint=None, time_limit=60.0, solver_threads=1, limit=0):
    if not math.isfinite(time_limit) or time_limit <= 0:
        raise ValueError("Time limit must be positive and finite.")
    if solver_threads <= 0 or limit < 0:
        raise ValueError("Solver threads must be positive and instance limit must be nonnegative.")
    example_dir = Path(example_dir)
    manifest = json.loads((example_dir / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["sha256"].items():
        path = example_dir / name
        if not path.resolve().is_relative_to(example_dir.resolve()):
            raise ValueError(f"Invalid example asset path: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Example asset checksum mismatch: {name}")

    try:
        import gurobipy as gp
        from gnn.inference.run_milp import solve_instance_simple
    except ModuleNotFoundError as error:
        if error.name != "gurobipy":
            raise
        raise RuntimeError('Install the solver extra with python -m pip install -e ".[solver]" and configure a Gurobi license.') from error

    metas = load_instances_meta_qkp(str(example_dir / "data"))
    # Empty pools build the instance graphs without supplying any decisions.
    dataset = InstancePoolDataset(metas, {key: [] for key in metas}, TMFE=1)
    if len(dataset) == 0:
        raise ValueError("The example contains no QKP instances.")
    model = load_trained_model(
        "energnn", manifest["model_config"],
        checkpoint or example_dir / "best.ckpt", torch.device("cpu"),
    )
    results = []
    for index in range(min(limit, len(dataset)) if limit else len(dataset)):
        graph, _, _, _, metadata = dataset[index]
        instance_id = metadata["instance_id"]
        started = time.perf_counter()
        try:
            mask, milp_energy, energy, status, _ = solve_instance_simple(
                model, graph, metas[instance_id], problem="qkp",
                time_limit=time_limit, solver_threads=solver_threads,
                enforce_problem_constraints=True,
            )
        except gp.GurobiError as error:
            raise RuntimeError(f"Gurobi could not solve instance {instance_id}: {error}. A working license supporting this model size is required.") from error
        params = metas[instance_id].globals
        row = {
            "instance_id": instance_id,
            "status": status if mask is not None else "NO_SOLUTION",
            "runtime_sec": time.perf_counter() - started,
            "time_limit_sec": float(time_limit),
            "energy": energy, "milp_energy": milp_energy,
            "energy_proven_optimal": bool(mask is not None and status == "OPTIMAL"),
            "objective": None, "feasible": None, "weight": None,
            "capacity": float(params["W"]), "selected_items": None,
        }
        if mask is not None:
            objective, feasible, weight = qkp_objective(mask, params["p"], params["Q"], params["w"], params["W"])
            if not feasible or energy is None or not math.isfinite(energy):
                raise RuntimeError(f"Invalid Gurobi solution for instance {instance_id}.")
            row.update(objective=objective, feasible=feasible, weight=weight,
                       selected_items=[i for i, value in enumerate(mask) if value])
        results.append(row)
        if mask is None:
            print(f"Instance {instance_id}: no solution found within the solver run.", flush=True)
        else:
            print(f"Instance {instance_id}: status={status} objective={objective:.2f} "
                  f"weight={weight:.2f}/{params['W']:.2f} energy={energy:.6f}", flush=True)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example-dir", type=Path, default=default_example_dir())
    parser.add_argument("--ckpt", type=Path, help="Optional compatible checkpoint override.")
    parser.add_argument("--time-limit", type=float, default=60.0, help="Gurobi seconds per instance.")
    parser.add_argument("--solver-threads", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="Maximum instances; 0 solves all three.")
    parser.add_argument("--output-json", type=Path, default=Path("outputs/qkp_example.json"))
    args = parser.parse_args(argv)
    print("Pretrained QKP EnerGNN optimization with Gurobi", flush=True)
    try:
        results = solve_examples(args.example_dir, args.ckpt, args.time_limit, args.solver_threads, args.limit)
    except (RuntimeError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Results saved to {args.output_json}")
    return 0 if all(row["feasible"] for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
