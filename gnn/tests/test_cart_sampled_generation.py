import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cart_dataset import dataset_maker_sampled as sampled
from cart_dataset.candidate_scoring import IncompleteCandidatePool, is_complete
from cart_dataset.facility_sampling import _expand_facilities, _static_config
from cart_dataset.model_cart_detailed import load_detailed_data, load_static_network
from gnn.data.reader import load_instances_meta, load_samples, load_static_config


ROOT = Path(__file__).resolve().parents[2]


def completed_result(mask):
    feasible = any(mask)
    objective = 100.0 + sum((i + 1) * bit for i, bit in enumerate(mask)) if feasible else None
    return {
        "status": "optimal" if feasible else "infeasible", "feasible": feasible,
        "proven_optimal": feasible, "objective": objective, "best_bound": objective,
        "mip_gap_rel": 0.0 if feasible else None, "source": "reduced_detailed_milp",
    }


@pytest.fixture
def expanded_config():
    sets, params = load_static_network(ROOT / "cart_dataset/Data_CART_og.dat")
    return _static_config(*_expand_facilities(sets, params, 16, 515151))


@pytest.fixture
def detailed_scorer(expanded_config):
    return sampled.DetailedCartScorer(
        dict(expanded_config, time_horizon=20, demand_horizon=3),
        {"c1": {"1": 1}}, seed=123, threads=2, full_time_limit=60,
        fixed_time_limit=30, mip_gap=1e-4,
    )


def test_facility_expansion_preserves_legacy_seed_and_accepts_capmax():
    sets, params = load_static_network(ROOT / "cart_dataset/Data_CART_og.dat")
    original = copy.deepcopy((sets, params))
    legacy_params = dict(params)
    legacy_params["FCAP"] = legacy_params.pop("CAPMAX")
    expanded = _expand_facilities(sets, params, 16, 515151)
    assert expanded == _expand_facilities(sets, legacy_params, 16, 515151)
    assert (sets, params) == original
    expanded_sets, expanded_params, metadata = expanded
    assert expanded_sets["m"] == [f"m{i}" for i in range(1, 17)]
    assert metadata["synthetic_facilities"] == 10
    for facility in sets["m"]:
        assert expanded_params["FCAP"][facility] == params["CAPMAX"][facility]
        assert expanded_params["CIM"][facility] == params["CIM"][facility]
    assert expanded != _expand_facilities(sets, params, 16, 515152)


def test_sampled_cart16_pool_is_seeded_diverse_and_bounded(expanded_config):
    best_mask = np.zeros(16, dtype=np.int8)
    best_mask[:3] = 1
    incumbent = np.zeros(16, dtype=np.int8)
    incumbent[-3:] = 1
    kwargs = dict(
        best_mask=best_mask, incumbents=[(incumbent, 1.0)],
        peak_active_demand=40.0, target_temp_size=320, seed=123,
    )
    inc = {"c1": {"1": 40}}
    pool = sampled._propose_candidates(expanded_config, inc, **kwargs)
    assert pool == sampled._propose_candidates(expanded_config, inc, **kwargs)
    assert 320 <= len(pool) < 1024
    assert tuple(best_mask) in pool and tuple(incumbent) in pool
    assert (0,) * 16 in pool and (1,) * 16 in pool
    assert all(len(mask) == 16 and set(mask) <= {0, 1} for mask in pool)
    assert any(source.startswith("hamming_") for source in pool.values())
    assert any(source.startswith("heuristic_") for source in pool.values())
    assert any(source.startswith("random_") for source in pool.values())


@pytest.mark.parametrize("has_incumbent", [False, True])
def test_fixed_mask_timeout_is_not_exported_as_a_cost_or_infeasible_label(
    monkeypatch, detailed_scorer, has_incumbent,
):
    monkeypatch.setattr(sampled, "solve_mask", lambda *a, **kw: {
        "status": "time_limit", "feasible": True if has_incumbent else None,
        "proven_optimal": False, "objective": 100.0 if has_incumbent else None,
        "best_bound": 50.0, "mip_gap_rel": 0.5 if has_incumbent else None,
    })
    with pytest.raises(IncompleteCandidatePool, match="fixed-mask solve"):
        detailed_scorer.score_fixed([1] * 16, "random")


def test_optimal_fixed_mask_accepts_solver_gap_dust_at_zero_mip_gap():
    record = completed_result([1, 0, 0, 1])
    record.update(
        objective=46969532.16779212,
        best_bound=46969532.167657,
        mip_gap_rel=2.876679261835726e-12,
    )
    assert is_complete(record, 0.0)

    record["mip_gap_rel"] = 1e-6
    assert not is_complete(record, 0.0)


def test_detailed_fixed_solver_preserves_patient_input_and_does_not_screen_capacity(
    monkeypatch, detailed_scorer,
):
    original = copy.deepcopy(detailed_scorer.data)
    calls = []

    def solve(data, mask, **kwargs):
        calls.append((data, mask, kwargs))
        return completed_result(mask)

    monkeypatch.setattr(sampled, "solve_mask", solve)
    detailed_scorer.peak_active_demand = 1000.0
    mask = [1] + [0] * 15
    result = detailed_scorer.score_fixed(mask, "random")
    assert result["feasible"] and result["capacity_heuristic_violation"] > 0
    assert result["solver_source"] == "reduced_detailed_milp"
    data, bits, kwargs = calls[0]
    assert data is detailed_scorer.data and data == original
    assert data["p"][None] == ["p1"] and data["INC"] == {("p1", "c1", 1): 1}
    assert len(data["U1"]) == len(data["U3"]) == 4 * 16 * 2
    assert bits == mask and kwargs == {
        "time_horizon": 20, "max_open_facilities": None,
        "time_limit": 30.0, "threads": 2, "mip_gap": 1e-4,
        "formulation": "detailed", "turnaround_limit": 17.0,
    }


def test_completed_detailed_reference_mask_is_reused(monkeypatch, detailed_scorer):
    mask = [1] + [0] * 15
    detailed_scorer.reference_mask = tuple(mask)
    detailed_scorer.reference_result = dict(completed_result(mask), source="detailed_reference")
    monkeypatch.setattr(sampled, "solve_mask", lambda *a, **kw: pytest.fail("Reference should be reused"))
    result = detailed_scorer.score_fixed(mask, "reference")
    assert result["solver_source"] == "detailed_reference"
    assert result["objective"] == result["best_bound"] == 101.0


@pytest.mark.parametrize("solve_fails", [False, True])
def test_full_reference_uses_detailed_model_and_disposes_resources(
    monkeypatch, detailed_scorer, solve_fails,
):
    pytest.importorskip("gurobipy")
    from cart_dataset import model_cart_detailed_gurobi as backend

    disposed = []
    captured = []
    mask = [1] + [0] * 15
    variables = {f"m{i + 1}": SimpleNamespace(X=bit) for i, bit in enumerate(mask)}
    model = SimpleNamespace(
        Status=sampled.GRB.OPTIMAL, SolCount=1, ObjVal=101.0, ObjBound=101.0, MIPGap=0.0,
        Runtime=0.1, NumVars=100, NumConstrs=90, NumBinVars=50,
        cbGetSolution=lambda _: mask, cbGet=lambda _: 101.0,
        dispose=lambda: disposed.append("model"),
    )

    def optimize(callback):
        if solve_fails:
            raise RuntimeError("Solver failed")
        callback(model, sampled.GRB.Callback.MIPSOL)

    model.optimize = optimize
    artifacts = SimpleNamespace(E1=variables, env=SimpleNamespace(dispose=lambda: disposed.append("env")))

    def build(data, **kwargs):
        captured.append((data, kwargs))
        return model, artifacts

    monkeypatch.setattr(backend, "build_detailed_gurobi_model", build)
    if solve_fails:
        with pytest.raises(RuntimeError, match="Solver failed"):
            detailed_scorer.solve_full()
    else:
        result_mask, objective, status, optimal, _, incumbents = detailed_scorer.solve_full()
        np.testing.assert_array_equal(result_mask, mask)
        assert objective == 101.0 and status == "optimal" and optimal
        assert len(incumbents) == 1
        assert detailed_scorer.reference_result["best_bound"] == 101.0
    data, kwargs = captured[0]
    assert data.m == [f"m{i}" for i in range(1, 17)] and data.p == ["p1"]
    assert data.time_horizon == 20
    assert kwargs == {
        "max_open_facilities": None, "solver_threads": 2, "time_limit": 60.0,
        "mip_gap": 1e-4, "solver_options": {"Seed": 123, "MIPGapAbs": 0.0},
        "turnaround_limit": 17.0,
    }
    assert disposed == ["model", "env"]


@pytest.mark.parametrize("has_incumbent", [False, True])
def test_reference_timeout_does_not_claim_optimality(monkeypatch, expanded_config, has_incumbent):
    mask = np.ones(16, dtype=np.int8)
    alternative = mask.copy()
    alternative[-1] = 0

    class TimeLimitedScorer:
        peak_active_demand = 1.0
        T = 130

        def __init__(self, *args, **kwargs):
            self.reference_result = {
                "status": "time_limit", "feasible": True if has_incumbent else None,
                "proven_optimal": False, "objective": 200.0 if has_incumbent else None,
                "best_bound": 50.0, "mip_gap_rel": 0.75 if has_incumbent else None,
            }

        def solve_full(self):
            return mask, self.reference_result["objective"], "time_limit", False, 1.0, []

        def score_fixed(self, proposed_mask, source):
            objective = 200.0 if sum(proposed_mask) == 16 else 100.0
            return dict(
                completed_result(proposed_mask), mask=list(proposed_mask),
                objective=objective, best_bound=objective, source=source,
                solver_source="reduced_detailed_milp", capacity=100.0,
                capacity_heuristic_violation=0.0,
            )

    monkeypatch.setattr(sampled, "DetailedCartScorer", TimeLimitedScorer)
    monkeypatch.setattr(sampled, "_propose_candidates", lambda *a, **kw: {
        tuple(mask): "reference", tuple(alternative): "random",
    })
    spec = dict(
        instance_id=0, total_therapies=1, seed_demand=123,
        demand_kwargs={"distribution": "poisson", "lambda_demand": 2.0},
        size_bucket="small", distribution="poisson", global_demand_index=0,
    )
    kwargs = dict(
        static_config=expanded_config, spec=spec, target_candidates=160, oversample_factor=2,
        max_positives=3, pos_threshold=1e-6, infeasible_fraction=0.3,
        full_time_limit=1.0, fixed_time_limit=1.0, threads=1, seed=123, mip_gap=1e-4,
    )
    if not has_incumbent:
        with pytest.raises(IncompleteCandidatePool, match="reference solve has no incumbent"):
            sampled._generate_one_instance(**kwargs)
    else:
        result = sampled._generate_one_instance(**kwargs)
        row = result["samples"][0]
        assert row["label_type"] == "positive" and not row["optimal"]
        assert not row["extras"]["full_milp_optimal"]
        assert row["objective_model"] == result["meta"]["objective_model"] == "cart_detailed"
        assert row["objective"] == result["summary"]["best_obj"] == 100.0
        reference_row = next(r for r in result["samples"] if r["source"] == "reference")
        assert reference_row["label_type"] == "ham_neg_feas"
        assert reference_row["regret_rel"] == 1.0


def test_existing_scale_is_preserved_without_overwrite(tmp_path):
    scale = tmp_path / "m16"
    scale.mkdir()
    marker = scale / "existing.txt"
    marker.write_text("keep")
    with pytest.raises(SystemExit):
        sampled.main(["--facility-counts", "16", "--out-dir", str(tmp_path)])
    assert marker.read_text() == "keep"
    assert list(tmp_path.iterdir()) == [scale]


def persistence_args(out_dir):
    return [
        "--facility-counts", "16", "--train-count", "2", "--val-count", "1", "--test-count", "1",
        "--total-low", "10", "--total-high", "10", "--candidates-per-instance", "3",
        "--time-horizon", "20", "--demand-horizon", "3", "--fixed-time-limit", "30",
        "--formulation", "detailed",
        "--out-dir", str(out_dir),
    ]


def test_prepare_only_saves_all_inputs_and_resume_reuses_them(tmp_path, monkeypatch):
    monkeypatch.setattr(sampled.DetailedCartScorer, "solve_full", lambda *a: pytest.fail("No solves during preparation"))
    args = persistence_args(tmp_path) + ["--facility-counts", "16", "20", "--prepare-only"]
    assert sampled.main(args) == 0
    saved = {}
    for scale in ("m16", "m20"):
        for split, count in (("train", 2), ("val", 1), ("test", 1)):
            directory = tmp_path / scale / split
            path = directory / "instances_meta.jsonl"
            saved[path] = path.read_bytes()
            config = load_static_config(directory / "static_config.json")
            metas = load_instances_meta(path, config)
            assert len(metas) == count
            assert all(sum(sum(days.values()) for days in meta.INC.values()) == 10 for meta in metas.values())
            assert path.read_bytes() == (directory / "instances_input.jsonl").read_bytes()
            assert (directory / "samples.jsonl").read_bytes() == b""
            assert not (directory / "checkpoints").exists()
            patient_dir = tmp_path / scale / f"{split}_patient_dats"
            assert len(list(patient_dir.glob("instance_*.dat"))) == count
            for instance_id, meta in metas.items():
                patient_path = patient_dir / f"instance_{instance_id}.dat"
                data = load_detailed_data(patient_path, meta.T)
                assert len(data["p"][None]) == 10
                assert data["m"][None] == config["M"]
                assert data["CAPMAX"] == config["CAPMAX"]
                actual = {}
                for (_, clinic, day), value in data["INC"].items():
                    actual[(clinic, day)] = actual.get((clinic, day), 0) + value
                expected = {(clinic, day): value for clinic, days in meta.INC.items() for day, value in days.items() if value}
                assert actual == expected
                saved[patient_path] = patient_path.read_bytes()

    missing_patient_path = tmp_path / "m16" / "test_patient_dats" / "instance_0.dat"
    missing_patient_path.unlink()

    monkeypatch.setattr(sampled, "draw_inc_for_instance", lambda *a, **kw: pytest.fail("Use saved demand"))
    assert sampled.main(args + ["--resume"]) == 0
    assert all(path.read_bytes() == contents for path, contents in saved.items())


@pytest.mark.parametrize("interruption", [IncompleteCandidatePool, KeyboardInterrupt])
def test_interrupted_run_resumes_instances_and_masks_without_duplicate_samples(
    tmp_path, monkeypatch, interruption,
):
    reference_calls, fixed_calls = [], []
    first_seed = 515151 + 16 * 100_000
    mask = [1] + [0] * 15
    alternative = [0, 1] + [0] * 14

    def reference(scorer):
        reference_calls.append(scorer.seed)
        if scorer.seed == first_seed:
            for split, count in (("train", 2), ("val", 1), ("test", 1)):
                rows = (tmp_path / "m16" / split / "instances_meta.jsonl").read_text().splitlines()
                assert len(rows) == count
        scorer.reference_mask = tuple(mask)
        scorer.reference_result = dict(completed_result(mask), source="detailed_reference")
        return np.array(mask), 101.0, "optimal", True, 0.1, []

    monkeypatch.setattr(sampled.DetailedCartScorer, "solve_full", reference)
    monkeypatch.setattr(sampled, "_propose_candidates", lambda *a, **kw: {
        tuple(mask): "reference", (0,) * 16: "all_zero",
        tuple(np.asarray(alternative, dtype=np.int8)): "random",
    })
    monkeypatch.setattr(sampled, "solve_mask", lambda data, bits, **kw: completed_result(bits))
    original_score = sampled.DetailedCartScorer.score_fixed

    def interrupted_score(scorer, bits, source):
        fixed_calls.append((scorer.seed, tuple(bits)))
        if scorer.seed == first_seed + 1 and tuple(bits) == tuple(alternative):
            raise interruption("Interrupted during the second instance")
        return original_score(scorer, bits, source)

    monkeypatch.setattr(sampled.DetailedCartScorer, "score_fixed", interrupted_score)
    args = persistence_args(tmp_path)
    with pytest.raises(interruption, match="Interrupted"):
        sampled.main(args)
    directory = tmp_path / "m16" / "train"
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["instances"] == 1 and summary["input_instances"] == 2 and not summary["complete"]
    assert len((directory / "samples.jsonl").read_text().splitlines()) == 3
    checkpoint = directory / "checkpoints" / "instance_0000" / "result.json"
    saved_checkpoint = checkpoint.read_bytes()
    partial_path = directory / "checkpoints" / "instance_0001" / "scored_masks.jsonl"
    assert len(partial_path.read_text().splitlines()) == 2

    with partial_path.open("ab") as fh:
        fh.write(b'{"mask": [0, 1')
    with (directory / "samples.jsonl").open("a") as fh:
        fh.write('{"torn":')
    fixed_calls.clear()

    def resumed_score(scorer, bits, source):
        fixed_calls.append((scorer.seed, tuple(bits)))
        return original_score(scorer, bits, source)

    monkeypatch.setattr(sampled.DetailedCartScorer, "score_fixed", resumed_score)
    monkeypatch.setattr(sampled, "draw_inc_for_instance", lambda *a, **kw: pytest.fail("Use saved demand"))
    assert sampled.main(args + ["--resume", "--fixed-time-limit", "60"]) == 0
    assert checkpoint.read_bytes() == saved_checkpoint
    assert reference_calls == [first_seed, first_seed + 1, first_seed + 20_000, first_seed + 40_000]
    assert [bits for seed, bits in fixed_calls if seed == first_seed + 1] == [tuple(alternative)]
    assert all(seed != first_seed for seed, _ in fixed_calls)
    assert len(partial_path.read_text().splitlines()) == 3
    for split, count in (("train", 2), ("val", 1), ("test", 1)):
        split_dir = tmp_path / "m16" / split
        metas = load_instances_meta(split_dir / "instances_meta.jsonl", load_static_config(split_dir / "static_config.json"))
        pools = load_samples(split_dir / "samples.jsonl", metas)
        rows = [json.loads(line) for line in (split_dir / "samples.jsonl").read_text().splitlines()]
        assert len(metas) == len(pools) == count
        assert len(rows) == count * 3
        assert all(len(pool) == 3 for pool in pools.values())
        assert json.loads((split_dir / "summary.json").read_text())["complete"]
    assert json.loads((tmp_path / "summary.json").read_text())["scales"]["m16"]["complete"]

    saved_samples = (directory / "samples.jsonl").read_bytes()
    monkeypatch.setattr(sampled.DetailedCartScorer, "solve_full", lambda *a: pytest.fail("Completed run should be reused"))
    monkeypatch.setattr(sampled.DetailedCartScorer, "score_fixed", lambda *a, **kw: pytest.fail("Completed run should be reused"))
    assert sampled.main(args + ["--resume"]) == 0
    assert (directory / "samples.jsonl").read_bytes() == saved_samples


@pytest.mark.parametrize("changed_args", [
    ["--seed", "123"], ["--turnaround-limit", "18"],
    ["--train-count", "3"], ["--mip-gap", "0"],
])
def test_resume_rejects_incompatible_settings_without_changing_saved_inputs(tmp_path, changed_args):
    args = persistence_args(tmp_path) + ["--prepare-only"]
    assert sampled.main(args) == 0
    saved = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    with pytest.raises(SystemExit, match="settings or base data differ"):
        sampled.main(args + ["--resume", *changed_args])
    assert all(path.read_bytes() == contents for path, contents in saved.items())


def test_resume_adopts_legacy_static_only_run_and_preserves_unknown_files(tmp_path):
    source, legacy = tmp_path / "source", tmp_path / "legacy"
    assert sampled.main(persistence_args(source) + ["--prepare-only"]) == 0
    for path in (source / "m16").rglob("static_config.json"):
        target = legacy / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    marker = legacy / "m16" / "existing.txt"
    marker.write_text("keep")
    with pytest.raises(SystemExit, match="no compatible generation checkpoint"):
        sampled.main(persistence_args(legacy) + ["--resume", "--prepare-only"])
    assert marker.read_text() == "keep"
    marker.unlink()
    assert sampled.main(persistence_args(legacy) + ["--resume", "--prepare-only"]) == 0
    assert (legacy / "m16" / "generation_state.json").is_file()
    assert len((legacy / "m16" / "train" / "instances_meta.jsonl").read_text().splitlines()) == 2


@pytest.fixture
def licensed_solver():
    gp = pytest.importorskip("gurobipy")
    try:
        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()
    except gp.GurobiError as error:
        pytest.skip(f"Gurobi license unavailable: {error}")


def test_sampled_cart16_cli_generates_training_reader_compatible_splits(tmp_path, monkeypatch):
    calls = []

    def reference(scorer):
        mask = [1] + [0] * 15
        scorer.reference_mask = tuple(mask)
        scorer.reference_result = dict(completed_result(mask), source="detailed_reference")
        return np.array(mask), 101.0, "optimal", True, 0.1, []

    def fixed(data, mask, **kwargs):
        calls.append((data, kwargs))
        assert len(data["p"][None]) == 10
        assert kwargs["time_horizon"] == 130 and kwargs["time_limit"] == 3600
        return completed_result(mask)

    monkeypatch.setattr(sampled.DetailedCartScorer, "solve_full", reference)
    monkeypatch.setattr(sampled, "solve_mask", fixed)
    assert sampled.main([
        "--facility-counts", "16", "--train-count", "1", "--val-count", "1", "--test-count", "1",
        "--total-low", "10", "--total-high", "10", "--oversample-factor", "2",
        "--out-dir", str(tmp_path),
    ]) == 0
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["generation_config"]["oversample_factor"] == 2
    assert summary["objective_model"] == "cart_detailed"
    assert summary["formulation"] == "detailed"
    assert len({id(data) for data, _ in calls}) == 3
    for split in ("train", "val", "test"):
        split_dir = tmp_path / "m16" / split
        config = load_static_config(split_dir / "static_config.json")
        metas = load_instances_meta(split_dir / "instances_meta.jsonl", config)
        pools = load_samples(split_dir / "samples.jsonl", metas)
        rows = [json.loads(line) for line in (split_dir / "samples.jsonl").read_text().splitlines()]
        assert len(metas) == len(pools) == 1
        assert len(config["M"]) == 16
        assert config["formulation"] == "detailed" and config["turnaround_limit"] == 17
        assert len(rows) == len(pools[0]) == 160
        assert len({tuple(row["e1"].values()) for row in rows}) == 160
        assert any(row["label_type"] == "positive" for row in rows)
        assert all(row["objective_model"] == "cart_detailed" for row in rows)
        assert config["time_horizon"] == 130 and config["demand_horizon"] == 110
        assert metas[0].T == 130
        assert all(row["best_bound"] == row["objective"] for row in rows if row["feasible"])
        assert sum(sum(days.values()) for days in metas[0].INC.values()) == 10


@pytest.mark.parametrize("formulation", ["auto", "detailed"])
def test_sampled_cart16_detailed_solver_smoke(tmp_path, monkeypatch, licensed_solver, formulation):
    def proposals(config, inc, *, best_mask, **kwargs):
        alternative = np.zeros(16, dtype=np.int8)
        alternative[next(i for i in range(16) if not best_mask[i])] = 1
        return {
            tuple(best_mask): "reference",
            (0,) * 16: "all_zero",
            tuple(alternative): "random",
        }

    monkeypatch.setattr(sampled, "_propose_candidates", proposals)
    horizon = 22
    assert sampled.main([
        "--facility-counts", "16", "--train-count", "1", "--val-count", "1", "--test-count", "1",
        "--total-low", "1", "--total-high", "1", "--oversample-factor", "2",
        "--time-horizon", str(horizon), "--demand-horizon", "3", "--candidates-per-instance", "3",
        "--formulation", formulation, "--mip-gap", "0",
        "--full-time-limit", "30", "--fixed-time-limit", "30", "--out-dir", str(tmp_path),
    ]) == 0
    for split in ("train", "val", "test"):
        split_dir = tmp_path / "m16" / split
        config = load_static_config(split_dir / "static_config.json")
        assert config["turnaround_limit"] == 17 and config["formulation"] == "detailed"
        metas = load_instances_meta(split_dir / "instances_meta.jsonl", config)
        assert metas[0].T == horizon
        assert len(load_samples(split_dir / "samples.jsonl", metas)[0]) == 3
        rows = [json.loads(line) for line in (split_dir / "samples.jsonl").read_text().splitlines()]
        assert all(row["objective_model"] == "cart_detailed" for row in rows)
        assert all(row["mip_gap_rel"] <= 1e-4 for row in rows if row["feasible"])
        assert all(row["turnaround_limit"] == 17 and row["formulation"] == "detailed" for row in rows)
        solver_source = "reduced_detailed_milp"
        assert any(row["extras"]["solver_source"] == solver_source for row in rows)
        assert any(row["label_type"] == "positive" and row["optimal"] for row in rows)
