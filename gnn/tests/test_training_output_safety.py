import pytest

from gnn.training import train


@pytest.mark.parametrize("mode", ["fresh", "resume", "eval_only"])
def test_existing_run_is_preserved_before_data_loading(tmp_path, monkeypatch, mode):
    original = {
        "best.ckpt": b"original checkpoint",
        "run_config.json": b'{"original": true}',
        "metrics.csv": b"original metrics\n",
    }
    for name, contents in original.items():
        (tmp_path / name).write_bytes(contents)

    def unexpected_load(*args, **kwargs):
        pytest.fail("Existing output must be rejected before loading data")

    monkeypatch.setattr(train, "_load_problem_data", unexpected_load)
    with pytest.raises(FileExistsError, match="Choose a new --out_dir"):
        train.run_train(
            data_dir=str(tmp_path / "missing-data"), out_dir=str(tmp_path),
            epochs=1, ckpt=str(tmp_path / "best.ckpt") if mode != "fresh" else None,
            eval_only=mode == "eval_only",
        )
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == original


@pytest.mark.parametrize("exists", [True, False])
def test_fresh_output_reaches_data_loading(tmp_path, monkeypatch, exists):
    output = tmp_path / "new-run"
    if exists:
        output.mkdir()

    class DataLoadingReached(Exception):
        pass

    def stop_at_data(*args, **kwargs):
        raise DataLoadingReached

    monkeypatch.setattr(train, "_load_problem_data", stop_at_data)
    with pytest.raises(DataLoadingReached):
        train.run_train(data_dir=str(tmp_path / "data"), out_dir=str(output), epochs=1)
