from types import SimpleNamespace

import pytest

from gnn.inference.run_milp import (
    _apply_cart_detailed_prediction_record,
    _cart_detailed_reference_stats_from_record,
    _collect_sample_reference,
    _conpas_rank_sets,
    _eval_prediction_mask,
    _prediction_primal_gap,
    _requires_exact_reference,
    _solve_cart_detailed_milp,
)
from gnn.data.reader import load_samples


def test_infeasible_prediction_has_full_primal_gap():
    assert _prediction_primal_gap(None, 25.0, feasible=False) == 1.0
    assert _prediction_primal_gap(1000.0, 25.0, feasible=False) == 1.0


def test_infeasible_prediction_without_reference_has_no_primal_gap():
    assert _prediction_primal_gap(None, None, feasible=False) is None


@pytest.mark.parametrize(
    ("pred_obj", "best_obj", "sense", "expected"),
    [
        (110.0, 100.0, "min", 0.1),
        (90.0, 100.0, "max", 0.1),
        (100.0, 100.0, "min", 0.0),
    ],
)
def test_feasible_prediction_keeps_relative_objective_gap(
    pred_obj, best_obj, sense, expected
):
    assert _prediction_primal_gap(
        pred_obj, best_obj, feasible=True, sense=sense
    ) == pytest.approx(expected)


def test_conpas_rank_sets_skip_user_fixed_variables():
    probs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    x0_indices, x1_indices = _conpas_rank_sets(
        probs,
        k0=2,
        k1=1,
        excluded_indices={0, 5},
    )

    assert x0_indices == [1, 2]
    assert x1_indices == [4]


def test_detailed_reference_excludes_aggregate_objectives(tmp_path):
    samples_path = tmp_path / "samples.jsonl"
    samples_path.write_text(
        '{"instance_id":1,"feasible":true,"objective":90,"e1":{"m1":0,"m2":1},'
        '"objective_model":"cart_aggregate"}\n'
        '{"instance_id":1,"feasible":true,"objective":110,"e1":{"m1":1,"m2":0},'
        '"label_type":"positive","objective_model":"cart_detailed"}\n'
        '{"instance_id":1,"feasible":true,"objective":95,"e1":{"m1":1,"m2":0},'
        '"objective_model":"cart_aggregate"}\n',
        encoding="utf-8",
    )
    meta = SimpleNamespace(M=["m1", "m2"], globals={})
    samples = load_samples(str(samples_path))[1]

    reference = _collect_sample_reference(meta, samples)

    assert reference["best_obj"] == 110
    assert reference["best_mask"] == [1, 0]
    assert reference["optimal_masks"] == {(1, 0)}
    assert reference["feasible_obj_by_mask"] == {(1, 0): 110}
    assert reference["objective_model"] == "cart_detailed"
    filtered = _collect_sample_reference(
        meta, samples, mask_filter=lambda mask: mask == (0, 1)
    )
    assert filtered["best_obj"] is None


def test_unmarked_sample_references_keep_existing_behavior():
    meta = SimpleNamespace(M=["m1", "m2"], globals={})
    samples = [
        SimpleNamespace(feasible=True, objective=90, e1={"m1":0,"m2":1}, extras={}),
        SimpleNamespace(feasible=True, objective=110, e1={"m1":1,"m2":0}, extras={}),
    ]

    reference = _collect_sample_reference(meta, samples)

    assert reference["best_obj"] == 90
    assert reference["feasible_obj_by_mask"] == {(0, 1): 90, (1, 0): 110}


def test_cart_exact_flag_keeps_detailed_test_reference():
    args = SimpleNamespace(
        problem="cart",
        model_type="energnn",
        cart_exact_reference=True,
        cart_force_off=None,
        cart_force_on=None,
    )

    assert _requires_exact_reference(args)
    assert _requires_exact_reference(args, {"objective_model": None})
    assert not _requires_exact_reference(args, {"objective_model": "cart_detailed"})


def test_exact_solver_accepts_detailed_dataset():
    args = SimpleNamespace(problem="cart", model_type="exact_solver")

    assert _requires_exact_reference(args, {"objective_model": "cart_detailed"})


@pytest.mark.parametrize(
    ("mask", "expected_obj", "expected_source", "matches_reference"),
    [
        ([1, 0], 110.0, "samples_cart_detailed", True),
        ([0, 1], None, None, False),
    ],
)
def test_detailed_prediction_scoring_never_uses_aggregate_candidate_cost(
    mask, expected_obj, expected_source, matches_reference
):
    meta = SimpleNamespace(
        M=["m1", "m2"], CAPMAX={"m1": 4, "m2": 4},
        globals={"peak_active_demand": 2},
    )
    samples = [
        SimpleNamespace(
            feasible=True, objective=90, e1={"m1": 0, "m2": 1},
            extras={"objective_model": "cart_aggregate"},
        ),
        SimpleNamespace(
            feasible=True, objective=110, e1={"m1": 1, "m2": 0},
            extras={"objective_model": "cart_detailed"},
        ),
    ]
    reference = _collect_sample_reference(meta, samples)

    prediction = _eval_prediction_mask("cart", meta, mask, 0, 0, ref_stats=reference)

    assert prediction["objective"] == expected_obj
    assert prediction["objective_source"] == expected_source
    assert (prediction["mask_key"] in reference["optimal_masks"]) == matches_reference
    gap = _prediction_primal_gap(
        prediction["objective"], reference["best_obj"], prediction["feasible"]
    )
    assert gap == (0.0 if matches_reference else None)


def test_detailed_cart_reference_record_supplies_missing_detailed_reference():
    meta = SimpleNamespace(M=["m1", "m2"], globals={})
    record = {
        "objective": 123.0,
        "opened_facilities": ["m2"],
        "status": "optimal",
    }

    reference = _cart_detailed_reference_stats_from_record(meta, record)

    assert reference == {
        "best_obj": 123.0,
        "best_mask": [0, 1],
        "optimal_masks": {(0, 1)},
        "feasible_obj_by_mask": {(0, 1): 123.0},
        "objective_model": "cart_detailed",
    }


def test_constrained_reference_requires_proven_optimum():
    meta = SimpleNamespace(M=["m1", "m2"])
    record = {"objective": 123.0, "opened_facilities": ["m2"],
              "status": "time_limit", "proven_optimal": False}
    assert _cart_detailed_reference_stats_from_record(
        meta, record, require_optimal=True,
    ) is None
    record.update(status="optimal", proven_optimal=True)
    reference = _cart_detailed_reference_stats_from_record(meta, record, require_optimal=True)
    assert reference["best_obj"] == 123.0


@pytest.mark.parametrize("formulation", ["detailed"])
@pytest.mark.parametrize("tee", [False, True])
def test_detailed_inference_passes_scenario_constraints(monkeypatch, formulation, tee):
    from gnn.inference import run_milp
    from cart_dataset import candidate_scoring

    meta = SimpleNamespace(M=["m1", "m2", "m3"], T=35)
    data = object()
    monkeypatch.setattr(run_milp, "_cart_detailed_data_from_meta", lambda meta: data)
    captured = {}

    def solve_stub(received_data, **kwargs):
        assert received_data is data
        captured.update(kwargs)
        return {"status": "optimal"}

    monkeypatch.setattr(candidate_scoring, "solve_mask", solve_stub)
    _solve_cart_detailed_milp(
        meta, max_open_facilities=2, min_open_facilities=1,
        fixed_assignments={0: 0, 2: 1},
        formulation=formulation,
        tee=tee,
    )
    assert captured["forced_e1_values"] == {"m1": 0, "m3": 1}
    assert captured["min_open_facilities"] == 1
    assert captured["max_open_facilities"] == 2
    assert captured["tee"] is tee
    if formulation == "detailed":
        assert captured["formulation"] == "detailed"


@pytest.mark.parametrize("min_k,max_k", [(2, 0), (0, 3)])
def test_aggregate_count_limits_require_new_reference(min_k, max_k):
    args = SimpleNamespace(
        problem="cart", model_type="energnn", cart_exact_reference=False,
        cart_force_on=None, cart_force_off=None, min_k=min_k, max_k=max_k,
    )
    assert _requires_exact_reference(args)


@pytest.mark.parametrize("formulation", ["detailed"])
@pytest.mark.parametrize("scenario", ["injected", "stored", "unknown", "recompute"])
@pytest.mark.parametrize("proven_optimal", [True, False])
def test_cart_reference_reuse_and_recomputation(
    monkeypatch, tmp_path, capsys, formulation, scenario, proven_optimal,
):
    import numpy as np
    from gnn.inference import run_milp

    meta = SimpleNamespace(
        M=["m1", "m2", "m3"], CAPMAX={"m1": 4, "m2": 4, "m3": 4},
        globals={"peak_active_demand": 1},
    )
    sample = SimpleNamespace(
        feasible=True, objective=10.0, e1={"m1": 1, "m2": 0, "m3": 0},
        extras={"objective_model": "cart_detailed"},
    )
    graph = {key: np.zeros((3, 1)) for key in ("Xc", "Xm", "edge_attr")}

    class DatasetStub:
        def __init__(self, *args, **kwargs):
            pass

        def __getitem__(self, index):
            return graph, None, None, None, None

        def _build_graph(self, instance_id):
            return graph, None

    monkeypatch.setattr(run_milp, "InstancePoolDataset", DatasetStub)
    monkeypatch.setattr(run_milp, "_load_eval_data", lambda *args: ({0: meta}, {0: [sample]}, None, 7))
    monkeypatch.setattr(run_milp, "load_trained_model", lambda *args: object())
    predicted_mask = [1, 0, 0] if scenario == "stored" else [0, 1, 0]
    monkeypatch.setattr(run_milp, "solve_instance_simple", lambda *args, **kwargs:
                        (predicted_mask, 0.0, 0.0, "OPTIMAL", {}))
    calls = []

    def scenario_solver(meta, mask=None, **kwargs):
        calls.append((mask, kwargs))
        return {
            "objective": 200.0 if mask is None else 250.0,
            "opened_facilities": ["m2", "m3"] if mask is None else ["m2"],
            "proven_optimal": proven_optimal if mask is None else True,
            "status": "optimal" if mask is not None or proven_optimal else "time_limit",
            "feasible": True, "formulation": formulation,
        }

    monkeypatch.setattr(run_milp, "_solve_cart_detailed_milp", scenario_solver)
    logged = []
    monkeypatch.setattr(run_milp, "_append_jsonl", lambda path, record: logged.append(record))
    argv = [
        "run_milp", "--problem", "cart", "--ckpt", "unused.ckpt",
        "--data_dir", str(tmp_path), "--instance_id", "0",
        "--cart_reference_formulation", formulation,
    ]
    if scenario == "injected":
        argv.extend(["--cart_force_off", "0", "--cart_force_on", "1",
                     "--min_k", "1", "--max_k", "2"])
    elif scenario == "recompute":
        argv.append("--cart_exact_reference")
    monkeypatch.setattr(run_milp.sys, "argv", argv)
    run_milp.main()
    output = capsys.readouterr().out
    assert ("[cart_scoring] instance=0" in output) == (scenario != "stored")
    assert ("for a new reference" in output) == (scenario in {"injected", "recompute"})
    assert all(kwargs["tee"] is True for mask, kwargs in calls)
    if scenario in {"stored", "unknown"}:
        assert [mask for mask, kwargs in calls] == ([] if scenario == "stored" else [predicted_mask])
        assert logged[0]["reference_source"] == "samples_cart_detailed"
        assert logged[0]["reference_objective"] == 10.0
        assert logged[0]["objective_gap_rel"] == (0.0 if scenario == "stored" else 24.0)
        return
    assert [mask for mask, kwargs in calls] == [None, [0, 1, 0]]
    for mask, kwargs in calls:
        assert kwargs["fixed_assignments"] == ({0: 0, 1: 1} if scenario == "injected" else {})
        assert kwargs["min_open_facilities"] == (1 if scenario == "injected" else 0)
        assert kwargs["max_open_facilities"] == (2 if scenario == "injected" else None)
        assert kwargs["formulation"] == formulation
    assert logged[0]["reference_source"] == f"{formulation}_cart_milp_constrained"
    assert logged[0]["reference_objective"] == (200.0 if proven_optimal else None)
    assert logged[0]["objective_gap_rel"] == (0.25 if proven_optimal else None)


@pytest.mark.parametrize("formulation", ["detailed"])
def test_cart_outage_reference_uses_selected_formulation(monkeypatch, formulation):
    from zero_shot_constraint_injection import evaluate_cart_facility_outages as outages

    meta = SimpleNamespace(M=["m1", "m2", "m3"])
    calls = []

    def solve_stub(meta, **kwargs):
        calls.append(kwargs)
        return {"objective": 200.0, "opened_facilities": ["m2"],
                "status": "optimal", "proven_optimal": True,
                "solver_time_sec": 1.0, "best_bound": 200.0, "mip_gap_rel": 0.0}

    monkeypatch.setattr(outages, "_solve_cart_detailed_milp", solve_stub)
    result = outages._solve_cart_details(
        meta, min_k=1, max_k=2, time_limit=30, threads=8, mip_gap=0,
        fixed_assignments={0: 0, 1: 1}, formulation=formulation,
    )
    assert result["configuration"] == [0, 1, 0]
    assert result["objective"] == 200.0 and result["proven_optimal"]
    assert calls[0]["fixed_assignments"] == {0: 0, 1: 1}
    assert calls[0]["min_open_facilities"] == 1
    assert calls[0]["max_open_facilities"] == 2
    assert calls[0]["formulation"] == formulation


def test_cart_outages_only_remove_originally_active_facilities():
    from zero_shot_constraint_injection import evaluate_cart_facility_outages as outages

    scenarios = outages._generate_scenarios(
        [1, 0, 1, 0, 1], outage_sizes=[1, 2, 3],
        max_pairs_per_instance=None, max_triples_per_instance=None,
        active_off_inactive_on=False, seed=0, instance_id=0,
    )
    assert len(scenarios) == 7
    assert all(set(scenario["off_indices"]).issubset({0, 2, 4}) for scenario in scenarios)
    assert all(not scenario["on_indices"] for scenario in scenarios)


def test_detailed_cart_prediction_record_scores_unknown_detailed_mask():
    reference = {
        "best_obj": 100.0,
        "best_mask": [1, 0],
        "optimal_masks": {(1, 0)},
        "feasible_obj_by_mask": {(1, 0): 100.0},
        "objective_model": "cart_detailed",
    }
    pred_stats = {
        "feasible": True,
        "objective": None,
        "objective_source": None,
        "mask_key": (0, 1),
    }

    updated = _apply_cart_detailed_prediction_record(
        pred_stats,
        reference,
        {"objective": 130.0, "feasible": True, "status": "optimal"},
    )

    assert updated["objective"] == 130.0
    assert updated["objective_source"] == "detailed_cart_milp"
    assert reference["feasible_obj_by_mask"][(0, 1)] == 130.0
