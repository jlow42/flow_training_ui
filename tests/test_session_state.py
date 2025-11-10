from session_state import SessionSnapshot, SessionStateStore


def test_session_state_round_trip(tmp_path):
    store_path = tmp_path / "session.json"
    store = SessionStateStore(store_path)
    snapshot = SessionSnapshot(
        files=["/tmp/a.csv", "/tmp/b.csv"],
        datasets=[{"path": "/tmp/a.csv", "columns": ["col"], "dtype_hints": {"col": "f"}}],
        training={
            "features": ["col"],
            "target": "label",
            "model": "Random Forest",
            "class_balance": "None",
            "random_seed": 99,
            "model_params": {"n_estimators": 200},
        },
        saved_at=1234.5,
    )

    store.save(snapshot)
    loaded = store.load()

    assert loaded is not None
    assert loaded.to_dict() == snapshot.to_dict()


def test_session_snapshot_from_dict_defaults():
    payload = {"files": ["/tmp/file.csv"], "training": {"random_seed": 7}}
    snapshot = SessionSnapshot.from_dict(payload)

    assert snapshot.files == ["/tmp/file.csv"]
    assert snapshot.datasets == []
    assert snapshot.training["random_seed"] == 7


def test_session_store_handles_corrupt_file(tmp_path):
    store_path = tmp_path / "session.json"
    store_path.write_text("not json")
    store = SessionStateStore(store_path)

    assert store.load() is None
