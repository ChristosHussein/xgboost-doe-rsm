import numpy as np


def synthetic_dataset():
    X = np.arange(240, dtype=float).reshape(30, 8)
    y = np.linspace(0.0, 1.0, 30)
    return X, y


def test_development_manager_never_exposes_holdout_arrays():
    from pipeline import CaliforniaHousingDevelopmentDataManager

    X, y = synthetic_dataset()
    manager = CaliforniaHousingDevelopmentDataManager(X=X, y=y, external_test_seed=42)
    split = manager.get_split(7)

    assert tuple(split._fields) == ("X_train", "X_val", "y_train", "y_val")
    assert not hasattr(manager, "X_test")
    assert not hasattr(manager, "y_test")
    assert not hasattr(manager, "_X_test")
    assert not hasattr(manager, "_y_test")


def test_development_evaluation_has_no_test_metric(monkeypatch):
    import pipeline

    class FakeRegressor:
        def __init__(self, **kwargs):
            pass

        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.zeros(len(X))

    X, y = synthetic_dataset()
    manager = pipeline.CaliforniaHousingDevelopmentDataManager(X=X, y=y)
    monkeypatch.setattr(pipeline.xgb, "XGBRegressor", FakeRegressor)

    result = pipeline.evaluate_development_model(
        eta=0.1,
        depth=3,
        subsample=0.8,
        reg_lambda=1.0,
        split_seed=5,
        model_seed=5,
        data_mgr=manager,
        measure_latency_details=False,
    )

    assert "val_rmse" in result
    assert not any("test" in key.lower() for key in result)


def test_new_design_output_never_logs_test_rmse(monkeypatch, tmp_path):
    import pipeline

    run = {
        "run_id": 1,
        "phase": "synthetic",
        "point_id": 1,
        "block": 1,
        "split_seed": 42,
        "model_seed": 42,
        "seed": 42,
        "replicate": 1,
        "run_order": 1,
        "x1": 0.0,
        "x2": 0.0,
        "x3": 0.0,
        "x4": 0.0,
    }

    monkeypatch.setattr(
        pipeline,
        "evaluate_development_model",
        lambda **kwargs: {
            "val_rmse": 0.5,
            "latency_us_median": 100.0,
            "latency_us_iqr": 1.0,
            "inplace_latency_us_median": 80.0,
            "inplace_latency_us_iqr": 1.0,
            "fit_time_s": 0.1,
        },
    )

    X, y = synthetic_dataset()
    manager = pipeline.CaliforniaHousingDevelopmentDataManager(X=X, y=y)
    output = tmp_path / "development_runs.csv"
    frame = pipeline.execute_design_pipeline(
        output_csv=str(output), runs_plan=[run], data_mgr=manager
    )

    assert "test_rmse" not in frame.columns
    assert "test_rmse" not in output.read_text(encoding="utf-8").splitlines()[0]
