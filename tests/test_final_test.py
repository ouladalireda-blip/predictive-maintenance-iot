"""Tests de l'examen final (données synthétiques, tout dans tmp_path)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from predictive_maintenance_iot.config import Config, load_config
from predictive_maintenance_iot.models import evaluate as evaluate_module
from predictive_maintenance_iot.models import final_test as final_module
from predictive_maintenance_iot.models.evaluate import B0, B2, HYBRID, XGB
from predictive_maintenance_iot.models.final_test import TestAlreadyRunError, run_final_test
from predictive_maintenance_iot.models.fingerprint import (
    FingerprintMismatchError,
    write_fingerprint,
)
from predictive_maintenance_iot.models.train import build_xgb_pipeline

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yaml"

# Seuils volontairement « reconnaissables » : on vérifie qu'ils sont repris tels quels.
FROZEN: dict[str, Any] = {
    "frozen_for_test": True,
    "primary_approach": "hybrid",
    "secondary_approach": "xgboost",
    "cost_missed_failure_eur": 3000,
    "cost_false_alarm_eur": 150,
    "config_ratio_fn_fp": 20.0,
    "min_precision": 0.5,
    "preventive_tool_wear_min": 200,
    "thresholds": {
        "xgboost": {
            "threshold": 0.37,
            "thresholds_by_ratio": {"5": 0.61, "10": 0.47, "20": 0.37, "50": 0.23},
        },
        "hybrid": {
            "threshold": 0.43,
            "thresholds_by_ratio": {"5": 0.71, "10": 0.53, "20": 0.43, "50": 0.29},
        },
    },
}


def _synthetic(cfg: Config, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = cfg.columns
    data: dict[str, Any] = {
        cols.row_id: np.arange(seed * 10_000, seed * 10_000 + n),
        "Product ID": [f"X{i:05d}" for i in range(n)],
        cols.product_type.column: rng.choice(cols.product_type.allowed, size=n),
    }
    for column in cols.numeric:
        bound = cfg.sensor_bounds[column]
        data[column] = rng.uniform(bound.min, bound.max, size=n).round(1)
    torque = data["Torque [Nm]"]
    failures = (torque > np.quantile(torque, 0.9)).astype(int)
    data[cols.target] = failures
    for i, column in enumerate(cols.leakage):
        data[column] = failures * (np.arange(n) % len(cols.leakage) == i)
    return pd.DataFrame(data)


@pytest.fixture
def cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Config:
    """Modèle entraîné sur un train synthétique, test.csv, thresholds.json figé, empreinte."""
    monkeypatch.chdir(tmp_path)
    base = load_config(CONFIG_PATH)
    config = base.model_copy(
        update={
            "paths": base.paths.model_copy(
                update={
                    "processed_dir": tmp_path / "processed",
                    "artifacts_dir": tmp_path / "artifacts",
                    "reports_dir": tmp_path / "reports",
                }
            ),
            "mlflow": base.mlflow.model_copy(
                update={"tracking_uri": f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"}
            ),
            "model": base.model.model_copy(
                update={"xgboost": base.model.xgboost.model_copy(update={"n_estimators": 20})}
            ),
            "analysis": base.analysis.model_copy(update={"bootstrap_iterations": 50}),
        }
    )
    target = config.columns.target
    train = _synthetic(config, 600, seed=1)
    config.paths.processed_dir.mkdir()
    _synthetic(config, 300, seed=3).to_csv(config.paths.processed_dir / "test.csv", index=False)

    config.paths.artifacts_dir.mkdir()
    X, y = train.drop(columns=[target]), train[target]
    joblib.dump(
        build_xgb_pipeline(config, y).fit(X, y), config.paths.artifacts_dir / "xgboost.joblib"
    )
    (config.paths.artifacts_dir / "thresholds.json").write_text(
        json.dumps(FROZEN), encoding="utf-8"
    )
    write_fingerprint(config)
    return config


@pytest.fixture
def no_optimisation(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Fait échouer tout appel à find_best_threshold et enregistre les tentatives."""
    calls: list[Any] = []

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        calls.append(args)
        raise AssertionError("find_best_threshold ne doit jamais être appelé sur le test")

    monkeypatch.setattr(evaluate_module, "find_best_threshold", forbidden)
    return calls


def test_final_test_uses_frozen_thresholds_without_optimisation(
    cfg: Config, no_optimisation: list[Any]
) -> None:
    payload = run_final_test(cfg)

    assert no_optimisation == []
    assert not hasattr(final_module, "find_best_threshold")  # même pas importé

    at_config = pd.DataFrame(payload["results_at_config_ratio"]).set_index("approach")
    assert list(at_config.index) == [
        B0,
        "B1 preventive",
        B2,
        XGB,
        HYBRID,
    ]
    assert at_config.loc[XGB, "threshold"] == 0.37
    assert at_config.loc[HYBRID, "threshold"] == 0.43

    by_ratio = pd.DataFrame(payload["results_by_ratio"])
    for ratio, expected in FROZEN["thresholds"]["xgboost"]["thresholds_by_ratio"].items():
        row = by_ratio[(by_ratio["ratio_fn_fp"] == ratio) & (by_ratio["approach"] == XGB)]
        assert row["threshold"].item() == expected
    for ratio, expected in FROZEN["thresholds"]["hybrid"]["thresholds_by_ratio"].items():
        row = by_ratio[(by_ratio["ratio_fn_fp"] == ratio) & (by_ratio["approach"] == HYBRID)]
        assert row["threshold"].item() == expected


def test_detections_match_frozen_threshold(cfg: Config) -> None:
    payload = run_final_test(cfg)

    test_df = pd.read_csv(cfg.paths.processed_dir / "test.csv")
    target = cfg.columns.target
    model = joblib.load(cfg.paths.artifacts_dir / "xgboost.joblib")
    proba = model.predict_proba(test_df.drop(columns=[target]))[:, 1]
    y = test_df[target].to_numpy().astype(bool)
    alert = proba >= 0.37

    xgb = next(r for r in payload["results_at_config_ratio"] if r["approach"] == XGB)
    assert xgb["detected"] == int((alert & y).sum())
    assert xgb["false_alarms"] == int((alert & ~y).sum())
    assert xgb["total_cost_eur"] == int((~alert & y).sum()) * 3000 + xgb["false_alarms"] * 150
    b0 = next(r for r in payload["results_at_config_ratio"] if r["approach"] == B0)
    assert b0["gain_vs_b0_eur"] == 0
    b2 = next(r for r in payload["results_at_config_ratio"] if r["approach"] == B2)
    assert b2["share_of_b2_gain"] == pytest.approx(1.0)


def test_outputs_written(cfg: Config) -> None:
    payload = run_final_test(cfg)

    results = json.loads((cfg.paths.artifacts_dir / "test_results.json").read_text("utf-8"))
    assert results == payload
    assert results["forced_rerun"] is False
    assert {"bootstrap_config_ratio", "error_analysis", "model_sha256"} <= set(results)
    report = (cfg.paths.reports_dir / "final_test.md").read_text(encoding="utf-8")
    assert "# Examen final sur le jeu de test" in report
    assert "| approach |" in report

    client = MlflowClient(tracking_uri=cfg.mlflow.tracking_uri)
    experiment = client.get_experiment_by_name(cfg.mlflow.experiment_name)
    assert experiment is not None
    run = client.search_runs([experiment.experiment_id])[0]
    assert run.info.run_name == "final-test"
    assert {a.path for a in client.list_artifacts(run.info.run_id)} == {
        "test_results.json",
        "final_test.md",
    }


def _spy_read_csv(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    read_paths: list[Path] = []
    real_read_csv = pd.read_csv

    def spy(path: Any, *args: Any, **kwargs: Any) -> pd.DataFrame:
        read_paths.append(Path(path))
        return real_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", spy)
    return read_paths


def test_refuses_second_run(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    run_final_test(cfg)
    results_path = cfg.paths.artifacts_dir / "test_results.json"
    before = results_path.read_bytes()
    read_paths = _spy_read_csv(monkeypatch)
    monkeypatch.setattr(final_module, "load_config", lambda: cfg)

    with pytest.raises(TestAlreadyRunError, match="déjà été passé"):
        final_module.main([])

    assert read_paths == []  # test.csv n'a pas été relu
    assert results_path.read_bytes() == before


def test_force_reruns_with_warning(cfg: Config, caplog: pytest.LogCaptureFixture) -> None:
    run_final_test(cfg)

    with caplog.at_level(logging.WARNING, logger=final_module.__name__):
        payload = run_final_test(cfg, force=True)

    assert payload["forced_rerun"] is True
    assert "--force" in caplog.text


def test_fingerprint_is_verified(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    model_path = cfg.paths.artifacts_dir / "xgboost.joblib"
    model_path.write_bytes(model_path.read_bytes() + b"\0")
    read_paths = _spy_read_csv(monkeypatch)

    with pytest.raises(FingerprintMismatchError):
        run_final_test(cfg)
    assert read_paths == []
    assert not (cfg.paths.artifacts_dir / "test_results.json").exists()


def test_requires_frozen_thresholds(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    path = cfg.paths.artifacts_dir / "thresholds.json"
    path.write_text(json.dumps({**FROZEN, "frozen_for_test": False}), encoding="utf-8")
    read_paths = _spy_read_csv(monkeypatch)

    with pytest.raises(ValueError, match="frozen_for_test"):
        run_final_test(cfg)
    assert read_paths == []
