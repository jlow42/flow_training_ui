from run_registry import RunRegistry


def test_run_registry_add_and_update(tmp_path):
    registry_path = tmp_path / "registry.json"
    registry = RunRegistry(registry_path)

    record = {
        "id": "run-1",
        "timestamp": 10.0,
        "model_name": "Random Forest",
        "target": "label",
        "tags": ["baseline"],
        "notes": "initial",
        "seed": 42,
    }

    registry.add_run(record)
    assert registry.find("run-1") is not None

    updated = registry.update_metadata("run-1", tags=["updated"], notes="refined", seed=77)
    assert updated is True

    reloaded = RunRegistry(registry_path)
    stored = reloaded.find("run-1")
    assert stored is not None
    assert stored["tags"] == ["updated"]
    assert stored["notes"] == "refined"
    assert stored["seed"] == 77


def test_run_registry_update_missing_run(tmp_path):
    registry = RunRegistry(tmp_path / "registry.json")

    assert registry.update_metadata("missing", tags=["x"]) is False
