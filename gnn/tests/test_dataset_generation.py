import copy
import itertools
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from ca_dataset import dataset_maker as ca
from cart_dataset.candidate_scoring import IncompleteCandidatePool, label_candidate_results
from cart_dataset.model_cart_detailed import load_detailed_data, load_static_network
from cart_dataset.patient_dats import main as export_patients
from cart_dataset.split_dataset import main as split_cart, split_instance_ids
from gnn.data.reader import load_instances_meta_ca, load_instances_meta_qkp, load_samples
from qkp_dataset import dataset_maker as qkp


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("module", [
    "cart_dataset.dataset_maker",
    "cart_dataset.dataset_maker_masks", "cart_dataset.patient_dats",
    "cart_dataset.dataset_maker_sampled",
    "cart_dataset.split_dataset",
    "ca_dataset.dataset_maker", "qkp_dataset.dataset_maker",
    "qkp_dataset.split_dataset",
])
def test_benchmark_cli_help(module):
    result = subprocess.run([sys.executable, "-m", module, "--help"], cwd=ROOT,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_ca_instance_generation_is_reproducible_without_a_solver(tmp_path):
    kwargs = {"n_items": 5, "n_bids": 8}
    first = ca.generate_cauctions(np.random.RandomState(123), str(tmp_path / "first.lp"), **kwargs)
    second = ca.generate_cauctions(np.random.RandomState(123), str(tmp_path / "second.lp"), **kwargs)
    assert first.n_bids == 8
    assert (first.A_csr != second.A_csr).nnz == 0
    np.testing.assert_array_equal(first.costs, second.costs)
    _, feasible, violation = ca.evaluate_cauction_mask(first.A_csr, first.costs, np.zeros(8))
    assert feasible and violation == 0


def test_qkp_parameter_generation_and_scoring_are_reproducible():
    first = qkp._generate_qkp_params(np.random.default_rng(123), 8, 0.3)
    second = qkp._generate_qkp_params(np.random.default_rng(123), 8, 0.3)
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)
    p, matrix, weights, capacity = first
    np.testing.assert_array_equal(matrix, matrix.T)
    assert np.all(np.diag(matrix) == 0)
    objective, feasible, weight = qkp.qkp_objective(np.zeros(8), p, matrix, weights, capacity)
    assert objective == weight == 0 and feasible


def solved(mask):
    feasible = any(mask)
    objective = float(sum((i + 1) * bit for i, bit in enumerate(mask))) if feasible else None
    return {"status": "optimal" if feasible else "infeasible", "feasible": feasible,
            "proven_optimal": feasible, "objective": objective, "best_bound": objective,
            "mip_gap_rel": 0.0 if feasible else None}


def test_cart_timeout_is_not_exported_as_an_infeasible_negative():
    unfinished = {"status": "maxTimeLimit", "feasible": None, "proven_optimal": False,
                  "objective": None, "mip_gap_rel": None}
    with pytest.raises(IncompleteCandidatePool):
        label_candidate_results([solved([1, 0]), unfinished])


def test_cart_splits_are_disjoint_deterministic_and_preserve_demand_order():
    metadata = [{"instance_id": i, "total_therapies": i} for i in range(1000)]
    random = split_instance_ids(metadata)
    assert random == split_instance_ids(metadata)
    assert [len(random[name]) for name in ("train", "val", "test")] == [800, 100, 100]
    hard = split_instance_ids(metadata, mode="demand_generalization", test_frac=0.2)
    assert [len(hard[name]) for name in ("train", "val", "test")] == [720, 80, 200]
    assert hard["test"] == set(range(800, 1000))
    for split in (random, hard):
        assert not split["train"] & split["val"]
        assert not split["train"] & split["test"]
        assert not split["val"] & split["test"]
        assert set.union(*split.values()) == set(range(1000))


def test_cart_split_does_not_replace_the_source_directory(tmp_path):
    with pytest.raises(SystemExit):
        split_cart(["--input-dir", str(tmp_path), "--output-dir", str(tmp_path)])
    assert tmp_path.exists()


def test_patient_export_preserves_capmax_and_transport_parameters(tmp_path):
    sets, params = load_static_network(ROOT / "cart_dataset/Data_CART_og.dat")
    config = {"C": sets["c"], "M": sets["m"], "TMFE": 7, "CAPMAX": params["CAPMAX"],
              "CIM": params["CIM"], "CVM": params["CVM"],
              "U1": {c: {m: params["U1"][(c, m)] for m in sets["m"]} for c in sets["c"]}}
    (tmp_path / "static_config.json").write_text(json.dumps(config))
    metadata = tmp_path / "instances_meta.jsonl"
    metadata.write_text(json.dumps({"instance_id": 9, "inc": {"c1": {"1": 2}, "c2": {"3": 1}}}) + "\n")
    assert export_patients(["--instances-meta-jsonl", str(metadata), "--out-dir", str(tmp_path / "patients"),
                           "--time-horizon", "20"]) == 0
    data = load_detailed_data(tmp_path / "patients/instance_9.dat", 20)
    assert len(data["p"][None]) == 3
    assert data["CAPMAX"] == params["CAPMAX"]
    assert data["U1"][("c1", "m1", "j1")] == params["U1"][("c1", "m1")]


def test_qkp_split_preserves_all_instances_and_reference_metadata(tmp_path):
    from qkp_dataset.split_dataset import main as split_qkp

    source, output = tmp_path / "source", tmp_path / "split"
    source.mkdir()
    rows = [{"instance_id": i, "optimal": i == 0, "opt_obj": float(i),
             "solve_time": 3600, "npz_path": f"instance_{i:04d}.npz"}
            for i in range(10)]
    samples = [{"instance_id": i, "objective": float(i)} for i in range(10)]
    (source / "instances.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (source / "samples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in samples))
    for row in rows:
        np.savez(source / row["npz_path"], n_items=np.array([1]))

    split_qkp(["--input-dir", str(source), "--output-dir", str(output)])

    collected = []
    for name, count in (("train", 8), ("val", 1), ("test", 1)):
        split = output / name
        split_rows = [json.loads(line) for line in (split / "instances.jsonl").read_text().splitlines()]
        assert len(split_rows) == count
        if name == "test":
            assert all(row["optimal"] for row in split_rows)
        split_samples = [json.loads(line) for line in (split / "samples.jsonl").read_text().splitlines()]
        assert {row["instance_id"] for row in split_samples} == {row["instance_id"] for row in split_rows}
        for row in split_rows:
            assert (split / row["npz_path"]).read_bytes() == (source / row["npz_path"]).read_bytes()
        collected.extend(split_rows)
    assert sorted(collected, key=lambda row: row["instance_id"]) == rows
    assert set(json.loads((output / "split_summary.json").read_text())["splits"]) == {"train", "val", "test"}
    assert not (output / "test_hard").exists()


def test_qkp_split_helpers_ignore_multiprocessing_completion_order():
    from qkp_dataset.split_dataset import _split_instance_ids

    ids = list(range(10))
    assert _split_instance_ids(ids, 0.1, 0.1, 123, certified_ids=[0, 2]) == _split_instance_ids(
        list(reversed(ids)), 0.1, 0.1, 123, certified_ids=[2, 0])


def test_qkp_split_rejects_insufficient_test_certificates_before_writing(tmp_path):
    from qkp_dataset.split_dataset import main as split_qkp

    source, output = tmp_path / "source", tmp_path / "split"
    source.mkdir()
    rows = [{"instance_id": i, "optimal": False} for i in range(10)]
    (source / "instances.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (source / "samples.jsonl").write_text("")
    with pytest.raises(ValueError, match="certified optimal QKP references"):
        split_qkp(["--input-dir", str(source), "--output-dir", str(output)])
    assert not output.exists()


@pytest.fixture
def licensed_solver():
    gp = pytest.importorskip("gurobipy")
    try:
        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()
    except gp.GurobiError as error:
        pytest.skip(f"Gurobi license unavailable: {error}")


@pytest.mark.parametrize("scale", ["small", "175", "250"])
def test_ca_cli_generates_reader_compatible_pools(tmp_path, licensed_solver, scale):
    ca.main(["--output-dir", str(tmp_path), "--scale", scale, "--n-items", "5", "--n-bids", "8",
             "--splits", "train", "--train-count", "2", "--workers", "2", "--solve-time-limit", "10",
             "--n-feasible-negs", "2", "--n-infeasible-negs", "2"])
    split = tmp_path / scale / "train"
    metadata = load_instances_meta_ca(str(split))
    pools = load_samples(str(split / "samples.jsonl"), metadata)
    assert len(metadata) == len(pools) == 2
    assert any(row.label_type == "positive" for row in pools[0])
    assert json.loads((tmp_path / "summary.json").read_text())["seed"] == 123


def test_qkp_cli_generates_reader_compatible_pools_and_guards_resume(tmp_path, licensed_solver):
    args = ["--output-dir", str(tmp_path), "--n-instances", "2", "--n-items", "8", "--n-jobs", "2",
            "--n-feasible-negs", "2", "--n-infeasible-negs", "2", "--time-limit", "10"]
    qkp.main(args)
    metadata = load_instances_meta_qkp(str(tmp_path))
    pools = load_samples(str(tmp_path / "samples.jsonl"), metadata)
    assert len(metadata) == len(pools) == 2
    assert any(row.label_type == "positive" for row in pools[0])
    before = (tmp_path / "samples.jsonl").read_bytes()
    qkp.main(args)
    assert (tmp_path / "samples.jsonl").read_bytes() == before
    with pytest.raises(ValueError, match="Resume settings differ"):
        qkp.main(args + ["--seed", "456"])


def test_recovered_qkp_milp_matches_exhaustive_small_instance(licensed_solver):
    params = qkp._generate_qkp_params(np.random.default_rng(123), 5, 0.3)
    expected = max(qkp.qkp_objective(mask, *params)[0] for mask in itertools.product((0, 1), repeat=5)
                   if qkp.qkp_objective(mask, *params)[1])
    _, objective, _, optimal, _ = qkp.solve_qkp_gurobi(*params, seed=123, time_limit=10)
    assert optimal and objective == pytest.approx(expected)
