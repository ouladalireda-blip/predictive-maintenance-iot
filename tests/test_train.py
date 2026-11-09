"""Tests de l'entraînement et du suivi MLflow (données synthétiques, MLflow dans tmp_path)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from predictive_maintenance_iot.config import Config, load_config
from predictive_maintenance_iot.models import train as train_module
from predictive_maintenance_iot.models.baseline import build_logistic_baseline
from predictive_maintenance_iot.models.train import (
    build_xgb_pipeline,
    compute_scale_pos_weight,
    train_model,
)
from predictive_maintenance_iot.utils import git_commit

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
N_ESTIMATORS_TEST = 20  # petit modèle : les tests vérifient la mécanique, pas la performance


def _synthetic(cfg: Config, n: int, seed: int) -> pd.DataFrame:
    """Lignes au format du CSV brut ; les pannes dépendent du couple (signal apprenable)."""
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
    failures = (torque > np.quantile(torque, 0.93)).astype(int)
    data[cols.target] = failures
    for column in cols.leakage:
        data[column] = failures
    return pd.DataFrame(data)


@pytest.fixture
def cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Config:
    """Config réelle redirigée vers tmp_path (données, artefacts, base MLflow)."""
    monkeypatch.chdir(tmp_path)  # les artefacts MLflow par défaut restent dans tmp_path
    base = load_config(CONFIG_PATH)
    return base.model_copy(
        update={
            "paths": base.paths.model_copy(
                update={
                    "raw_data": tmp_path / "raw" / "ai4i2020.csv",
                    "processed_dir": tmp_path / "processed",
                    "artifacts_dir": tmp_path / "artifacts",
                }
            ),
            "mlflow": base.mlflow.model_copy(
                update={"tracking_uri": f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"}
            ),
            "model": base.model.model_copy(
                update={
                    "xgboost": base.model.xgboost.model_copy(
                        update={"n_estimators": N_ESTIMATORS_TEST}
                    )
                }
            ),
        }
    )


@pytest.fixture
def train_df(cfg: Config) -> pd.DataFrame:
    return _synthetic(cfg, n=600, seed=1)


@pytest.fixture
def val_df(cfg: Config) -> pd.DataFrame:
    return _synthetic(cfg, n=200, seed=2)


def _with_scale_pos_weight(cfg: Config, value: str | float) -> Config:
    xgb = cfg.model.xgboost.model_copy(update={"scale_pos_weight": value})
    return cfg.model_copy(update={"model": cfg.model.model_copy(update={"xgboost": xgb})})


# --- scale_pos_weight -------------------------------------------------------------------


def test_compute_scale_pos_weight() -> None:
    y = pd.Series([0] * 28 + [1] * 2)
    assert compute_scale_pos_weight(y) == pytest.approx(14.0)


def test_compute_scale_pos_weight_without_failures_raises() -> None:
    with pytest.raises(ValueError, match="aucune panne"):
        compute_scale_pos_weight(pd.Series([0, 0, 0]))


def test_auto_scale_pos_weight_uses_train_ratio(cfg: Config, train_df: pd.DataFrame) -> None:
    y = train_df[cfg.columns.target]
    pipeline = build_xgb_pipeline(_with_scale_pos_weight(cfg, "auto"), y)

    expected = (y == 0).sum() / (y == 1).sum()
    assert pipeline.named_steps["model"].get_params()["scale_pos_weight"] == pytest.approx(expected)


def test_xgb_pipeline_has_no_renaming_step(cfg: Config, train_df: pd.DataFrame) -> None:
    pipeline = build_xgb_pipeline(cfg, train_df[cfg.columns.target])
    assert [name for name, _ in pipeline.steps] == ["preprocess", "model"]


def test_numeric_scale_pos_weight_is_respected(cfg: Config, train_df: pd.DataFrame) -> None:
    pipeline = build_xgb_pipeline(_with_scale_pos_weight(cfg, 5.0), train_df[cfg.columns.target])
    assert pipeline.named_steps["model"].get_params()["scale_pos_weight"] == 5.0


# --- train_model ------------------------------------------------------------------------


def test_train_model_logs_run_and_saves_joblib(
    cfg: Config, train_df: pd.DataFrame, val_df: pd.DataFrame
) -> None:
    pipeline = build_xgb_pipeline(cfg, train_df[cfg.columns.target])

    train_model("xgboost", pipeline, cfg, train_df, val_df)

    client = MlflowClient(tracking_uri=cfg.mlflow.tracking_uri)
    experiment = client.get_experiment_by_name(cfg.mlflow.experiment_name)
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    run = runs[0]

    assert run.info.run_name == "xgboost"
    params = run.data.params
    assert params["random_seed"] == str(cfg.project.random_seed)
    assert params["n_train"] == str(len(train_df))
    assert params["n_val"] == str(len(val_df))
    assert float(params["train_failure_rate"]) == pytest.approx(train_df[cfg.columns.target].mean())
    assert float(params["scale_pos_weight"]) == pytest.approx(
        compute_scale_pos_weight(train_df[cfg.columns.target])
    )
    assert params["n_estimators"] == str(N_ESTIMATORS_TEST)
    assert params["max_depth"] == str(cfg.model.xgboost.max_depth)

    metrics = run.data.metrics
    assert set(metrics) >= {"pr_auc", "roc_auc", "brier_score"}
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0

    assert run.data.tags["model_type"] == "XGBClassifier"
    assert run.data.tags["git_commit"] == git_commit()  # format vérifié dans test_utils.py
    artifacts = {a.path for a in client.list_artifacts(run.info.run_id)}
    assert "feature_importance_gain.json" in artifacts
    importance = json.loads(
        Path(client.download_artifacts(run.info.run_id, "feature_importance_gain.json")).read_text(
            encoding="utf-8"
        )
    )
    feature_names = set(pipeline.named_steps["preprocess"].get_feature_names_out())
    assert importance  # au moins une variable utilisée par les arbres
    assert set(importance) <= feature_names  # noms snake_case du préprocesseur

    assert (cfg.paths.artifacts_dir / "xgboost.joblib").is_file()


def test_logistic_run_has_no_feature_importance(
    cfg: Config, train_df: pd.DataFrame, val_df: pd.DataFrame
) -> None:
    train_model("logistic", build_logistic_baseline(cfg), cfg, train_df, val_df)

    client = MlflowClient(tracking_uri=cfg.mlflow.tracking_uri)
    experiment = client.get_experiment_by_name(cfg.mlflow.experiment_name)
    assert experiment is not None
    run = client.search_runs([experiment.experiment_id])[0]
    assert run.data.tags["model_type"] == "LogisticRegression"
    assert "feature_importance_gain.json" not in {
        a.path for a in client.list_artifacts(run.info.run_id)
    }
    assert (cfg.paths.artifacts_dir / "logistic.joblib").is_file()


def test_same_seed_gives_identical_probabilities(
    cfg: Config, train_df: pd.DataFrame, val_df: pd.DataFrame
) -> None:
    target = cfg.columns.target
    X_train, y_train = train_df.drop(columns=[target]), train_df[target]
    X_val = val_df.drop(columns=[target])

    probas = [
        build_xgb_pipeline(cfg, y_train).fit(X_train, y_train).predict_proba(X_val)
        for _ in range(2)
    ]
    np.testing.assert_array_equal(probas[0], probas[1])


# --- main : le jeu de test n'est jamais lu ----------------------------------------------


def test_main_never_reads_test_csv(
    cfg: Config,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processed = cfg.paths.processed_dir
    processed.mkdir(parents=True)
    train_df.to_csv(processed / "train.csv", index=False)
    val_df.to_csv(processed / "val.csv", index=False)
    assert not (processed / "test.csv").exists()

    read_paths: list[Path] = []
    real_read_csv = pd.read_csv

    def spy_read_csv(path: Any, *args: Any, **kwargs: Any) -> pd.DataFrame:
        read_paths.append(Path(path))
        return real_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(train_module, "load_config", lambda: cfg)
    monkeypatch.setattr(pd, "read_csv", spy_read_csv)

    train_module.main()

    assert {p.name for p in read_paths} == {"train.csv", "val.csv"}
    for name in ("xgboost", "logistic"):
        assert (cfg.paths.artifacts_dir / f"{name}.joblib").is_file()
