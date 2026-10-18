import json
from pathlib import Path

from cart_dataset.extract import pack_inc, pack_instance_metadata
from cart_dataset.model_cart_detailed import (
    build_detailed_instance_from_aggregate,
    expand_patient_arrivals,
    load_static_network,
)
from gnn.data.reader import load_instances_meta, load_static_config


ROOT = Path(__file__).resolve().parents[2]
BASE_DAT = ROOT / "cart_dataset" / "Data_CART_og.dat"


def test_expand_patient_arrivals_is_deterministic():
    patients, inc = expand_patient_arrivals(
        ["c1", "c2", "c3", "c4"],
        3,
        {("c1", 1): 2, ("c2", 3): 1},
    )

    assert patients == ["p1", "p2", "p3"]
    assert inc == {
        ("p1", "c1", 1): 1,
        ("p2", "c1", 1): 1,
        ("p3", "c2", 3): 1,
    }


def test_legacy_fcap_static_config_loads_as_capmax(tmp_path):
    path = tmp_path / "static_config.json"
    path.write_text(json.dumps({"FCAP": {"m1": 4}}), encoding="utf-8")

    config = load_static_config(path.as_posix())

    assert config["CAPMAX"] == {"m1": 4}


def test_cart_reader_uses_declared_horizon_with_legacy_fallback(tmp_path):
    config = {
        "C": ["c1"], "M": ["m1"], "TMFE": 7, "CAPMAX": {"m1": 4},
        "CIM": {"m1": 10}, "CVM": {"m1": 20}, "U1": {"c1": {"m1": 100}},
    }
    path = tmp_path / "instances_meta.jsonl"
    row = {"instance_id": 9, "inc": {"c1": {"1": 1, "3": 0}}}
    path.write_text(json.dumps(row) + "\n")
    assert load_instances_meta(path, config)[9].T == 10
    assert load_instances_meta(path, dict(config, time_horizon=130))[9].T == 130
    row["time_horizon"] = 20
    path.write_text(json.dumps(row) + "\n")
    assert load_instances_meta(path, dict(config, time_horizon=130))[9].T == 20


def test_aggregate_demand_builds_detailed_cart_instance():
    static_sets, static_params = load_static_network(BASE_DAT)
    static_params["TMFE"] = 7
    instance = build_detailed_instance_from_aggregate(
        static_sets,
        static_params,
        {("c1", 1): 2, ("c2", 3): 1},
        time_horizon=20,
        max_open_facilities=3,
    )

    packed_inc = pack_inc(instance)
    metadata = pack_instance_metadata(instance)

    assert len(instance.p) == 3
    assert sum(sum(days.values()) for days in packed_inc.values()) == 3
    assert packed_inc["c1"][1] == 2
    assert packed_inc["c2"][3] == 1
    assert metadata["CAPMAX"]["m2"] == 31.0
    assert metadata["U1"]["c1"]["m1"] == 1040.0
    assert hasattr(instance, "CON1")
    assert hasattr(instance, "obj")
