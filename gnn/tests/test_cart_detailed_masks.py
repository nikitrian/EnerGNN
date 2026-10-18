import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pyomo.environ import SolverFactory

from cart_dataset.model_cart_detailed import (
    build_detailed_model,
    build_detailed_parameters,
    build_fixed_mask_instance,
    load_detailed_data,
    load_static_network,
    restrict_detailed_facilities,
)
from cart_dataset import dataset_maker_masks as runner


@pytest.fixture
def patient_data():
    base = Path(__file__).resolve().parents[2] / "cart_dataset/Data_CART_og.dat"
    sets, params = load_static_network(base)
    sets["m"] = ["m1", "m2"]
    detailed = build_detailed_parameters(sets, params)
    data = {name: {None: values} for name, values in {
        "c": sets["c"], "m": sets["m"], "h": detailed["h"],
        "j": detailed["j"], "p": ["p1"],
    }.items()}
    for name in ("CIM", "CVM"):
        data[name] = {m: params[name][m] for m in sets["m"]}
    for name in ("CAPMAX", "TT1", "TT3", "U1", "U3"):
        data[name] = detailed[name]
    for name in ("FMAX", "FMIN", "TAD", "TLS", "TMFE"):
        data[name] = {None: detailed[name]}
    data["INC"] = {("p1", "c1", 1): 1}
    return data


def write_patient_dat(path, data):
    lines = []
    for name, entries in data.items():
        if name in ("c", "m", "h", "j", "p"):
            lines.append(f"set {name} := {' '.join(entries[None])};")
        elif None in entries:
            lines.append(f"param {name} := {entries[None]};")
        else:
            lines.append(f"param {name} :=")
            for key, value in entries.items():
                indices = key if isinstance(key, tuple) else (key,)
                lines.append(" ".join(map(str, (*indices, value))))
            lines.append(";")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_patient_parser_preserves_indexed_transport_parameters(patient_data, tmp_path):
    path = tmp_path / "instance_9.dat"
    write_patient_dat(path, patient_data)
    assert load_detailed_data(path, 20) == patient_data


def test_restricted_build_removes_facilities_before_variables_are_created(patient_data):
    before = copy.deepcopy(patient_data)
    restricted = restrict_detailed_facilities(patient_data, ["m1"])
    reduced = build_fixed_mask_instance(patient_data, [1, 0], time_horizon=20)
    full = build_detailed_model(20).create_instance({None: patient_data})

    assert list(reduced.m) == ["m1"]
    assert reduced.E1["m1"].fixed and reduced.E1["m1"].value == 1
    assert len(reduced.Y1) * 2 == len(full.Y1)
    assert len(reduced.Y3) * 2 == len(full.Y3)
    assert set(restricted["CAPMAX"]) == {"m1"}
    assert all(key[1] == "m1" for key in restricted["U1"])
    assert all(key[0] == "m1" for key in restricted["U3"])
    assert restricted["INC"] == patient_data["INC"]
    assert patient_data == before


@pytest.mark.parametrize("mask", [[1], [0, 2], [0, 0]])
def test_bad_or_empty_masks_are_rejected_by_model_builder(patient_data, mask):
    with pytest.raises(ValueError):
        build_fixed_mask_instance(patient_data, mask, time_horizon=20)


@pytest.mark.parametrize("facilities", [["m7"], ["m1", "m1"]])
def test_bad_facility_subsets_are_rejected(patient_data, facilities):
    with pytest.raises(ValueError):
        restrict_detailed_facilities(patient_data, facilities)


def fake_solve(data, mask, **kwargs):
    feasible = any(mask)
    return {
        "status": "optimal" if feasible else "infeasible", "feasible": feasible,
        "proven_optimal": feasible, "objective": 100.0 if mask == [1, 0] else 200.0 if feasible else None,
        "best_bound": 100.0 if mask == [1, 0] else 200.0 if feasible else None,
        "mip_gap_rel": 0.0 if feasible else None,
    }


def arguments(tmp_path):
    inputs = tmp_path / "inputs"
    references = tmp_path / "test_references.json"
    entries = []
    for dat in inputs.glob("instance_*.dat"):
        data = load_detailed_data(dat, 20)
        facilities = list(data["m"][None])
        entries.append({
            "instance_id": runner._instance_id(dat), "horizon": 20,
            "patients": len(data["p"][None]), "turnaround_limit": 17,
            "objective": 100, "best_bound": 100, "solver_gap": 0,
            "e1": {m: int(i == 0) for i, m in enumerate(facilities)},
            "input_sha256": runner._digest(dat),
        })
    references.write_text(json.dumps({"changes": entries}))
    return ["--dat-dir", str(inputs), "--out-dir", str(tmp_path / "results"),
            "--time-horizon", "20", "--sampled-candidates", "3",
            "--references", str(references)]


def test_runner_exports_full_masks_and_retries_only_unfinished_results(patient_data, tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    write_patient_dat(inputs / "instance_9.dat", patient_data)
    calls = []

    def solve(data, mask, **kwargs):
        calls.append(tuple(mask))
        if mask == [0, 1] and calls.count((0, 1)) == 1:
            return {"status": "maxTimeLimit", "feasible": None, "proven_optimal": False,
                    "objective": None, "best_bound": None, "mip_gap_rel": None}
        return fake_solve(data, mask, **kwargs)

    monkeypatch.setattr(runner, "solve_mask", solve)
    assert runner.main(arguments(tmp_path)) == 2
    assert not (tmp_path / "results/instance_9/samples.jsonl").exists()
    assert runner.main(arguments(tmp_path)) == 0
    assert calls.count((0, 1)) == 2
    assert calls.count((1, 0)) == 0
    assert all(calls.count(mask) == 1 for mask in [(0, 0), (1, 1)])
    assert runner.main(arguments(tmp_path) + ["--collect-only"]) == 0
    rows = [json.loads(line) for line in (tmp_path / "results/samples.jsonl").read_text().splitlines()]
    assert len(rows) == 4
    assert {tuple(row["e1"].values()) for row in rows} == {(0, 0), (1, 0), (0, 1), (1, 1)}
    assert next(row for row in rows if row["label_type"] == "positive")["e1"] == {"m1": 1, "m2": 0}
    assert next(row for row in rows if row["e1"] == {"m1": 0, "m2": 1})["regret_rel"] == 1.0
    assert all(row["objective_model"] == "cart_detailed" for row in rows)


def test_sampled_mask_shards_reuse_reference_resume_and_collect(patient_data, tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    dat = inputs / "instance_9.dat"
    write_patient_dat(dat, patient_data)
    reference = tmp_path / "references.json"
    reference.write_text(json.dumps({"changes": [{
        "instance_id": 9, "objective": 100, "best_bound": 100, "solver_gap": 0,
        "horizon": 20, "patients": 1, "turnaround_limit": 17,
        "e1": {"m1": 1, "m2": 0}, "input_sha256": runner._digest(dat),
    }]}), encoding="utf-8")
    calls = []

    def solve(data, mask, **kwargs):
        calls.append(tuple(mask))
        return fake_solve(data, mask, **kwargs)

    monkeypatch.setattr(runner, "solve_mask", solve)
    base = arguments(tmp_path) + ["--sampled-candidates", "3", "--sampling-seed", "42"]
    shard = base + ["--references", str(reference), "--num-mask-shards", "2"]
    assert runner.main(shard) == 0
    plan_path = tmp_path / "results/instance_9/candidate_plan.json"
    plan = runner._read_json(plan_path)
    assert plan["masks"][0] == [1, 0]
    assert len(plan["masks"]) == 4
    assert not (tmp_path / "results/instance_9/samples.jsonl").exists()
    assert runner.main(base + ["--collect-only"]) == 2
    assert not (tmp_path / "results/samples.jsonl").exists()
    assert runner.main(shard + ["--mask-shard-index", "1"]) == 0
    assert len(calls) == 3 and (1, 0) not in calls
    assert runner.main(shard) == 0
    assert len(calls) == 3
    assert runner._read_json(plan_path) == plan
    assert runner.main(base + ["--collect-only"]) == 0
    rows = [json.loads(line) for line in (tmp_path / "results/samples.jsonl").read_text().splitlines()]
    assert len(rows) == 4
    assert next(row for row in rows if row["optimal"])["e1"] == {"m1": 1, "m2": 0}
    with pytest.raises(ValueError, match="different input/model settings"):
        runner.main(base + ["--sampling-seed", "43", "--collect-only"])


def test_sampled_plan_requires_completed_reference(patient_data, tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    write_patient_dat(inputs / "instance_9.dat", patient_data)
    monkeypatch.setattr(runner, "solve_mask", lambda *a, **kw: pytest.fail("Must validate before solving"))
    with pytest.raises(ValueError, match="matching completed reference"):
        runner.main(["--dat-dir", str(inputs), "--out-dir", str(tmp_path / "results"),
                     "--time-horizon", "20", "--sampled-candidates", "3"])


def test_cart16_plan_samples_without_enumeration(tmp_path, monkeypatch):
    facilities = [f"m{i}" for i in range(1, 17)]
    data = {"m": {None: facilities}, "c": {None: ["c1"]},
            "TMFE": {None: 7}, "INC": {("p1", "c1", 1): 1},
            "CAPMAX": {m: 10 for m in facilities},
            "CIM": {m: 100 for m in facilities}, "CVM": {m: 10 for m in facilities},
            "U1": {("c1", m, "j1"): i + 1 for i, m in enumerate(facilities)}}
    args = SimpleNamespace(sampled_candidates=320, sampling_seed=42, out_dir=tmp_path,
                           time_horizon=130, turnaround_limit=17, max_open_facilities=None,
                           collect_only=False, mip_gap=1e-4)
    reference = {"e1": {m: int(i == 0) for i, m in enumerate(facilities)}, "solver_gap": 0}
    masks = runner.candidate_masks(data, 9, "digest", args, reference)
    assert 320 <= len(masks) < 65536
    assert len({tuple(mask) for mask in masks}) == len(masks)
    assert masks[0] == [1] + [0] * 15
    assert [0] * 16 in masks and [1] * 16 in masks
    assert runner.candidate_masks(data, 9, "digest", args) == masks
    assert all(len(mask) == 16 and set(mask) <= {0, 1} for mask in masks)


def test_generator_pool_matches_serial_detailed_generator(patient_data, tmp_path, monkeypatch):
    import numpy as np
    from cart_dataset import dataset_maker_sampled as generator

    data = copy.deepcopy(patient_data)
    facilities = [f"m{i}" for i in range(1, 17)]
    data["m"] = {None: facilities}
    for name in ("CIM", "CVM", "CAPMAX", "TT1", "TT3", "U1", "U3"):
        expanded = {}
        for key, value in patient_data[name].items():
            indices = key if isinstance(key, tuple) else (key,)
            if "m1" not in indices:
                continue
            for facility in facilities:
                new = tuple(facility if part == "m1" else part for part in indices)
                expanded[new if isinstance(key, tuple) else new[0]] = value
        data[name] = expanded
    clinics = list(data["c"][None])
    static = {"M": facilities, "C": clinics, "FCAP": data["CAPMAX"],
              "CIM": data["CIM"], "CVM": data["CVM"], "TMFE": 7, "util_cap": 0.8,
              "demand_horizon": 110,
              "U1": {c: {m: data["U1"][(c, m, "j1")] for m in facilities} for c in clinics}}
    inc = {c: ({"1": 1} if c == "c1" else {}) for c in clinics}
    metadata = {"instance_id": 9, "seed_demand": 123, "total_therapies": 1,
                "demand_kwargs": {}, "inc": inc, "formulation": "detailed", "turnaround_limit": 17}
    inputs, source = tmp_path / "inputs", tmp_path / "source"
    inputs.mkdir()
    source.mkdir()
    write_patient_dat(inputs / "instance_9.dat", data)
    (source / "static_config.json").write_text(json.dumps(static), encoding="utf-8")
    (source / "instances_meta.jsonl").write_text(json.dumps(metadata) + "\n", encoding="utf-8")
    best = np.asarray([1] + [0] * 15, dtype=np.int8)
    incumbent = np.asarray([0, 1] + [0] * 14, dtype=np.int8)
    seed = 515151 + 16 * 100000 + 20000 + 9
    full_calls = []

    def solve(data, mask, **kwargs):
        feasible = any(mask)
        objective = 100.0 if mask == best.tolist() else 200.0 if feasible else None
        return {"status": "optimal" if feasible else "infeasible", "feasible": feasible,
                "proven_optimal": feasible, "objective": objective, "best_bound": objective,
                "mip_gap_rel": 0.0 if feasible else None, "source": "fake_detailed",
                "solver_time_sec": 0.0}

    class Scorer:
        def __init__(self, config, arrivals, **kwargs):
            assert kwargs["seed"] == seed and config["formulation"] == "detailed"
            self.data = data
            self.T = 130
            self.peak_active_demand = 1.0
            self.reference_result = None

        def solve_full(self):
            assert self.data == data
            full_calls.append(1)
            self.reference_result = solve(data, best.tolist())
            return best.copy(), 100.0, "optimal", True, 0.0, [(incumbent.copy(), 200.0)]

        def score_fixed(self, mask, source):
            result = solve(data, list(mask))
            capacity = sum(static["FCAP"][m] * static["util_cap"] for m, bit in zip(facilities, mask) if bit)
            return dict(result, mask=list(mask), solver_source=result["source"], source=source,
                        capacity=capacity, capacity_heuristic_violation=max(0.0, 1.0 - capacity))

    monkeypatch.setattr(generator, "DetailedCartScorer", Scorer)
    monkeypatch.setattr(runner, "solve_mask", solve)
    base = ["--dat-dir", str(inputs), "--out-dir", str(tmp_path / "results"),
            "--time-horizon", "130",
                                  "--input-source", str(source), "--split", "val",
                                  "--candidates-per-instance", "160", "--oversample-factor", "2"]
    assert runner.main(base + ["--prepare-plans-only"]) == 0
    plan = runner._read_json(tmp_path / "results/instance_9/candidate_plan.json")
    assert plan["masks"][3] == incumbent.tolist()
    assert plan["sources"][0] == "optimal"
    assert runner.main(base + ["--prepare-plans-only"]) == 0
    assert len(full_calls) == 1
    for shard in range(3):
        assert runner.main(base + ["--num-mask-shards", "3", "--mask-shard-index", str(shard)]) == 0
    assert len(full_calls) == 1
    assert runner.main(base + ["--collect-only", "--dataset-source", str(source)]) == 0
    actual = [json.loads(line) for line in (tmp_path / "results/samples.jsonl").read_text().splitlines()]
    serial_static = dict(static, time_horizon=130, formulation="detailed", turnaround_limit=17)
    serial = generator._generate_one_instance(serial_static, metadata, 160, 2, 3, 1e-6, 0.30,
                                              86400, 3600, 1, seed, 1e-4, instance_meta=metadata)
    for row in serial["samples"]:
        if row["source"] == "reference":
            row["source"] = "optimal"
    assert len(actual) == 160
    fields = ("e1", "label_type", "objective", "regret_abs", "regret_rel", "hamming_r", "source", "capacity")
    assert [{key: row[key] for key in fields} for row in actual] == [
        {key: row[key] for key in fields} for row in serial["samples"]]
    assert actual[0]["extras"]["full_milp_optimal"]
    assert runner._read_json(tmp_path / "results/static_config.json")["candidate_generation"]["candidates_per_instance"] == 160
    with pytest.raises(ValueError, match="different input/model settings"):
        runner.main(base + ["--max-positives", "2", "--collect-only"])


def test_default_pool_rejects_inputs_without_16_facilities(patient_data, tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    write_patient_dat(inputs / "instance_9.dat", patient_data)
    with pytest.raises(ValueError, match="exactly 16 facilities"):
        runner.main(["--dat-dir", str(inputs), "--out-dir", str(tmp_path / "results"),
                     "--time-horizon", "20", "--prepare-plans-only"])


def test_instance_shards_partition_files_and_resume_with_different_shard_count(
    patient_data, tmp_path, monkeypatch
):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    for instance_id in (101, 12, 9, 25):
        write_patient_dat(inputs / f"instance_{instance_id}.dat", patient_data)
    calls = []

    def solve(data, mask, **kwargs):
        calls.append((tuple(mask), kwargs["threads"]))
        return fake_solve(data, mask, **kwargs)

    monkeypatch.setattr(runner, "solve_mask", solve)
    for shard_index, expected_ids in enumerate(([9, 25], [12, 101])):
        summary = tmp_path / f"shard_{shard_index}.json"
        assert runner.main(arguments(tmp_path) + [
            "--num-shards", "2", "--shard-index", str(shard_index),
            "--workers", "1", "--solver-threads", "4", "--summary-json", str(summary),
        ]) == 0
        report = runner._read_json(summary)
        assert report["selected_instance_ids"] == expected_ids
        assert report["complete_instance_ids"] == expected_ids
        assert report["incomplete_instance_ids"] == []
        assert report["complete"] and report["solver_threads"] == 4
        assert not (tmp_path / "results/samples.jsonl").exists()
    assert len(calls) == 12 and all(threads == 4 for _, threads in calls)
    assert runner.main(arguments(tmp_path) + ["--num-shards", "3", "--shard-index", "0"]) == 0
    assert len(calls) == 12
    assert runner.main(arguments(tmp_path) + ["--num-shards", "2", "--collect-only"]) == 0
    assert not (tmp_path / "results/samples.jsonl").exists()
    assert runner.main(arguments(tmp_path) + ["--collect-only"]) == 0
    rows = [json.loads(line) for line in (tmp_path / "results/samples.jsonl").read_text().splitlines()]
    assert [row["instance_id"] for row in rows] == [9] * 4 + [12] * 4 + [25] * 4 + [101] * 4


def test_empty_shard_is_a_successful_noop(patient_data, tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    write_patient_dat(inputs / "instance_9.dat", patient_data)
    monkeypatch.setattr(runner, "solve_mask", lambda *args, **kwargs: pytest.fail("Empty shard must not solve"))
    summary = tmp_path / "empty.json"
    assert runner.main(arguments(tmp_path) + [
        "--num-shards", "3", "--shard-index", "2", "--workers", "2",
        "--collect-only", "--summary-json", str(summary),
    ]) == 0
    report = runner._read_json(summary)
    assert report["selected_instance_ids"] == [] and report["active_workers"] == 0
    assert report["complete"]
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("flags", [
    ["--workers", "0"], ["--num-shards", "0"], ["--shard-index", "-1"],
    ["--num-shards", "2", "--shard-index", "2"], ["--solver-threads", "0"],
    ["--num-shards", "2", "--instance-id", "9"],
    ["--num-shards", "2", "--instance-index", "0"],
    ["--num-shards", "2", "--task-index", "0"],
    ["--num-shards", "2", "--collect-only", "--dataset-source", "source"],
    ["--num-mask-shards", "0"], ["--num-mask-shards", "2", "--mask-shard-index", "2"],
    ["--num-mask-shards", "2", "--collect-only"],
    ["--sampled-candidates", "2"], ["--sampled-candidates", "3", "--task-index", "0"],
    ["--cart16-pool", "--references", "references.json"],
    ["--cart16-pool", "--sampled-candidates", "320"],
    ["--prepare-plans-only"], ["--cart16-pool", "--prepare-plans-only", "--num-mask-shards", "2"],
])
def test_invalid_worker_and_shard_arguments_are_rejected(tmp_path, flags):
    with pytest.raises(SystemExit) as error:
        runner.main(arguments(tmp_path) + flags)
    assert error.value.code == 2


def test_unfinished_single_mask_sets_nonzero_exit_and_shard_summary(patient_data, tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    write_patient_dat(inputs / "instance_9.dat", patient_data)
    monkeypatch.setattr(runner, "solve_mask", lambda *args, **kwargs: {
        "status": "maxTimeLimit", "feasible": True, "proven_optimal": False,
        "objective": 100.0, "best_bound": 80.0, "mip_gap_rel": 0.2,
    })
    summary = tmp_path / "unfinished.json"
    assert runner.main(arguments(tmp_path) + ["--mask-index", "3", "--summary-json", str(summary)]) == 2
    report = runner._read_json(summary)
    assert report["incomplete_instance_ids"] == [9] and not report["complete"]
    assert report["instances"][0]["unresolved_masks"] == ["01"]
    assert not (tmp_path / "results/instance_9/samples.jsonl").exists()


def test_process_workers_collect_in_numeric_instance_order(patient_data, tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    for instance_id in (12, 9):
        write_patient_dat(inputs / f"instance_{instance_id}.dat", patient_data)
    monkeypatch.setattr(runner, "solve_mask", fake_solve)
    assert runner.main(arguments(tmp_path)) == 0
    summary = tmp_path / "parallel.json"
    assert runner.main(arguments(tmp_path) + [
        "--collect-only", "--workers", "2", "--summary-json", str(summary),
    ]) == 0
    report = runner._read_json(summary)
    assert report["active_workers"] == 2 and report["complete_instance_ids"] == [9, 12]
    rows = [json.loads(line) for line in (tmp_path / "results/samples.jsonl").read_text().splitlines()]
    assert [row["instance_id"] for row in rows] == [9] * 4 + [12] * 4


def test_verified_reference_mask_is_reused_without_solving(patient_data, tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    dat = inputs / "instance_9.dat"
    write_patient_dat(dat, patient_data)
    reference = tmp_path / "references.json"
    reference.write_text(json.dumps({"changes": [{
        "instance_id": 9, "horizon": 20, "patients": 1,
        "turnaround_limit": 17,
        "objective": 100, "best_bound": 100, "solver_gap": 0,
        "e1": {"m1": 1, "m2": 0}, "input_sha256": runner._digest(dat),
    }]}))

    def unexpected_solve(*args, **kwargs):
        pytest.fail("The existing optimal mask must not be solved again")

    monkeypatch.setattr(runner, "solve_mask", unexpected_solve)
    args = arguments(tmp_path) + ["--references", str(reference), "--mask-index", "0"]
    assert runner.main(args) == 0
    result = runner._read_json(tmp_path / "results/instance_9/mask_10.json")
    assert result["source"] == "detailed_reference"
    assert result["open_facilities"] == ["m1"]


def test_reference_input_mismatch_is_rejected(patient_data):
    reference = {"horizon": 20, "patients": 1, "max_open_facilities": None, "turnaround_limit": 17,
                 "e1": {"m1": 1, "m2": 0}, "input_sha256": "different"}
    with pytest.raises(ValueError, match="different patient input"):
        runner.validate_reference(reference, patient_data, "actual", 20, None)


def test_cluster_reference_directory_can_contain_unfinished_jobs(tmp_path):
    path = tmp_path / "instance_9.json"
    path.write_text(json.dumps({"solve": {"termination_condition": "maxTimeLimit"}, "instance": {}}))
    assert runner.load_references(tmp_path) == {}
    with pytest.raises(ValueError, match="not optimal"):
        runner.load_references(path)


def test_time_limit_incumbent_is_unresolved_and_tighter_gap_requires_retry():
    record = {"status": "maxTimeLimit", "feasible": True, "proven_optimal": False,
              "mip_gap_rel": 0.2}
    assert not runner.is_complete(record, 1e-4)
    record.update(status="optimal", proven_optimal=True, mip_gap_rel=1e-4)
    assert runner.is_complete(record, 1e-4)
    assert not runner.is_complete(record, 1e-5)


def write_source_metadata(path, data):
    path.mkdir()
    clinics, facilities = data["c"][None], data["m"][None]
    config = {
        "C": clinics, "M": facilities, "TMFE": data["TMFE"][None],
        "CAPMAX": data["CAPMAX"], "CIM": data["CIM"], "CVM": data["CVM"],
        "U1": {c: {m: data["U1"][(c, m, "j1")] for m in facilities} for c in clinics},
    }
    metadata = {"instance_id": 9, "seed_demand": 321, "total_therapies": 1,
                "inc": {c: {"1": int(c == "c1")} for c in clinics}}
    (path / "static_config.json").write_text(json.dumps(config))
    (path / "instances_meta.jsonl").write_text(json.dumps(metadata) + "\n")
    return metadata


def test_collection_exports_a_verified_directly_trainable_split(patient_data, tmp_path, monkeypatch):
    from gnn.data.reader import load_instances_meta, load_samples, load_static_config

    inputs = tmp_path / "inputs"
    inputs.mkdir()
    write_patient_dat(inputs / "instance_9.dat", patient_data)
    source = tmp_path / "source"
    write_source_metadata(source, patient_data)
    monkeypatch.setattr(runner, "solve_mask", fake_solve)
    assert runner.main(arguments(tmp_path)) == 0
    assert runner.main(arguments(tmp_path) + ["--collect-only", "--dataset-source", str(source)]) == 0
    output = tmp_path / "results"
    config = load_static_config(str(output / "static_config.json"))
    metadata = load_instances_meta(str(output / "instances_meta.jsonl"), config)
    samples = load_samples(str(output / "samples.jsonl"), metadata)
    assert config["objective_model"] == "cart_detailed" and config["time_horizon"] == 20
    assert len(samples[9]) == 4
    assert all(row.seed_demand == 321 for row in samples[9])
    assert json.loads((source / "static_config.json").read_text()).get("objective_model") is None


def test_collection_rejects_source_metadata_for_different_patient_arrivals(patient_data, tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    write_patient_dat(inputs / "instance_9.dat", patient_data)
    source = tmp_path / "source"
    metadata = write_source_metadata(source, patient_data)
    metadata["inc"]["c1"]["1"] = 2
    (source / "instances_meta.jsonl").write_text(json.dumps(metadata) + "\n")
    monkeypatch.setattr(runner, "solve_mask", fake_solve)
    assert runner.main(arguments(tmp_path)) == 0
    with pytest.raises(ValueError, match="arrival metadata differs"):
        runner.main(arguments(tmp_path) + ["--collect-only", "--dataset-source", str(source)])
    assert not (tmp_path / "results/samples.jsonl").exists()


@pytest.fixture
def licensed_gurobi():
    gp = pytest.importorskip("gurobipy")
    try:
        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()
    except gp.GurobiError as error:
        pytest.skip(f"Gurobi license unavailable: {error}")


@pytest.mark.parametrize("solve_fails", [False, True])
@pytest.mark.parametrize("turnaround_limit", [None, 18])
def test_direct_mask_solve_disposes_model_then_environment(patient_data, monkeypatch, solve_fails, turnaround_limit):
    from cart_dataset import model_cart_detailed_gurobi as backend

    disposed = []

    def optimize():
        if solve_fails:
            raise RuntimeError("Solver failed")

    model = SimpleNamespace(
        optimize=optimize,
        dispose=lambda: disposed.append("model"),
        Status=backend.GRB.OPTIMAL, SolCount=1,
        ObjVal=100.0, ObjBound=100.0, MIPGap=0.0, Runtime=0.1,
        NumVars=12, NumConstrs=10, NumBinVars=6,
    )
    env = SimpleNamespace(dispose=lambda: disposed.append("environment"))
    artifacts = backend.DetailedCartArtifacts(E1={}, env=env)

    def build(data, **kwargs):
        assert data.m == ["m1"]
        assert kwargs["fix_e1"] is True
        assert kwargs["solver_threads"] == 4
        assert kwargs["turnaround_limit"] == (17 if turnaround_limit is None else turnaround_limit)
        return model, artifacts

    monkeypatch.setattr(backend, "build_detailed_gurobi_model", build)
    options = {} if turnaround_limit is None else {"turnaround_limit": turnaround_limit}
    if solve_fails:
        with pytest.raises(RuntimeError, match="Solver failed"):
            runner.solve_mask(patient_data, [1, 0], time_horizon=20, threads=4, **options)
    else:
        result = runner.solve_mask(patient_data, [1, 0], time_horizon=20, threads=4, **options)
        assert result["proven_optimal"] and result["objective"] == 100.0
    assert disposed == ["model", "environment"]


def test_mask_runner_defaults_to_17_and_rejects_cache_with_different_turnaround(patient_data, tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    write_patient_dat(inputs / "instance_9.dat", patient_data)

    def solve(data, mask, **kwargs):
        assert kwargs["turnaround_limit"] == 17
        return fake_solve(data, mask, **kwargs)

    monkeypatch.setattr(runner, "solve_mask", solve)
    args = arguments(tmp_path)
    assert runner.main(args) == 0
    checkpoint = tmp_path / "results/instance_9/mask_10.json"
    saved = checkpoint.read_bytes()
    record = runner._read_json(checkpoint)
    assert record["signature"]["turnaround_limit"] == record["turnaround_limit"] == 17
    references = tmp_path / "test_references.json"
    records = json.loads(references.read_text())
    records["changes"][0]["turnaround_limit"] = 18
    references.write_text(json.dumps(records))
    with pytest.raises(ValueError, match="Cached mask belongs to different input/model settings"):
        runner.main(args + ["--turnaround-limit", "18"])
    assert checkpoint.read_bytes() == saved


def test_process_workers_solve_separate_instance_masks(patient_data, tmp_path, licensed_gurobi):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    for instance_id in (12, 9):
        write_patient_dat(inputs / f"instance_{instance_id}.dat", patient_data)
    summary = tmp_path / "solved_parallel.json"
    assert runner.main(arguments(tmp_path) + [
        "--workers", "2", "--mask-index", "1", "--solver-threads", "1",
        "--time-limit", "60", "--summary-json", str(summary),
    ]) == 0
    assert runner._read_json(summary)["active_workers"] == 2
    for instance_id in (9, 12):
        record = runner._read_json(tmp_path / f"results/instance_{instance_id}/mask_01.json")
        assert record["proven_optimal"] and record["model_num_variables"] > 0
    assert not (tmp_path / "results/samples.jsonl").exists()


@pytest.mark.parametrize("mask", [[1, 0], [0, 1], [1, 1]])
def test_pruned_mask_matches_full_model_with_closed_facilities_fixed_zero(
    patient_data, mask, licensed_gurobi
):
    full = build_detailed_model(20).create_instance({None: patient_data})
    for facility, bit in zip(full.m, mask):
        full.E1[facility].fix(bit)
    solver = SolverFactory("gurobi_direct")
    try:
        solver.solve(full, load_solutions=False, options={"OutputFlag": 0, "Threads": 1, "MIPGap": 0})
        full_objective = float(solver._solver_model.ObjVal)
        full_variables = int(solver._solver_model.NumVars)
    finally:
        solver.close()
    reduced = runner.solve_mask(patient_data, mask, time_horizon=20, threads=1, mip_gap=0)
    assert reduced["proven_optimal"]
    assert reduced["objective"] == pytest.approx(full_objective, abs=1e-6)
    assert reduced["model_num_variables"] <= full_variables
