"""Tests du coût métier, du choix de seuil et de la comparaison sur la validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from predictive_maintenance_iot.config import Config, load_config
from predictive_maintenance_iot.models import evaluate as evaluate_module
from predictive_maintenance_iot.models.baseline import build_logistic_baseline
from predictive_maintenance_iot.models.evaluate import (
    B0,
    SCORED_MODELS,
    compare_on_validation,
    cost_at_threshold,
    find_best_threshold,
    sensitivity_analysis,
)
from predictive_maintenance_iot.models.train import build_xgb_pipeline

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


# --- cost_at_threshold ------------------------------------------------------------------


def test_cost_at_threshold_hand_computed() -> None:
    y = [1, 1, 0, 0, 0]
    p = [0.9, 0.4, 0.6, 0.2, 0.1]
    # Seuil 0,5 : alertes sur 0,9 (TP) et 0,6 (FP) ; 0,4 est une panne ratée (FN).
    result = cost_at_threshold(y, p, 0.5, cost_fn=3000, cost_fp=150)

    assert (result["tp"], result["fp"], result["fn"], result["tn"]) == (1, 1, 1, 2)
    assert result["recall"] == pytest.approx(0.5)
    assert result["precision"] == pytest.approx(0.5)
    assert result["total_cost_eur"] == 3000 + 150


def test_cost_at_threshold_without_alert_has_zero_precision() -> None:
    result = cost_at_threshold([1, 0], [0.3, 0.2], 1.0, cost_fn=10, cost_fp=1)
    assert result["precision"] == 0.0
    assert result["total_cost_eur"] == 10


# --- find_best_threshold ----------------------------------------------------------------
# y = [0, 0, 1, 1], p = [0.1, 0.4, 0.35, 0.8], cost_fn = 10, cost_fp = 1
# seuil 0 / 0,1 : 2 FP            → coût 2,  précision 0,50
# seuil 0,35    : 2 TP, 1 FP      → coût 1,  précision 0,67   ← optimum sans contrainte
# seuil 0,4     : 1 TP, 1 FP, 1 FN → coût 11, précision 0,50
# seuil 0,8     : 1 TP, 1 FN      → coût 10, précision 1,00
# seuil 1       : 2 FN            → coût 20, précision 0 (aucune alerte)
Y_SMALL = [0, 0, 1, 1]
P_SMALL = [0.1, 0.4, 0.35, 0.8]


def test_find_best_threshold_known_minimum() -> None:
    best = find_best_threshold(Y_SMALL, P_SMALL, cost_fn=10, cost_fp=1, min_precision=0.0)

    assert best["threshold"] == pytest.approx(0.35)
    assert best["total_cost_eur"] == 1
    assert best["constraint_satisfied"] is True


def test_min_precision_excludes_cheaper_threshold() -> None:
    best = find_best_threshold(Y_SMALL, P_SMALL, cost_fn=10, cost_fp=1, min_precision=0.7)

    assert best["threshold"] == pytest.approx(0.8)  # 0,35 (précision 0,67) est exclu
    assert best["total_cost_eur"] == 10
    assert best["precision"] >= 0.7
    assert best["constraint_satisfied"] is True


def test_tie_breaks_towards_highest_threshold() -> None:
    # Seuils 0 et 0,2 : mêmes alertes, même coût → on garde 0,2 (le plus élevé).
    best = find_best_threshold([0, 1], [0.9, 0.2], cost_fn=10, cost_fp=1, min_precision=0.0)
    assert best["threshold"] == pytest.approx(0.2)


def test_no_admissible_threshold_is_reported() -> None:
    # La panne a un score plus bas que la ligne normale : aucune précision de 100 % possible.
    best = find_best_threshold([0, 1], [0.9, 0.2], cost_fn=10, cost_fp=1, min_precision=1.0)

    assert best["constraint_satisfied"] is False
    assert best["threshold"] == pytest.approx(0.2)  # coût minimal sans contrainte
    assert best["total_cost_eur"] == 1


def test_threshold_moves_with_cost_of_missed_failure() -> None:
    rng = np.random.default_rng(0)
    y = (rng.random(2_000) < 0.05).astype(int)
    proba = np.clip(0.3 * y + rng.normal(0.3, 0.15, size=y.size), 0, 1)  # scores bruités

    expensive_miss = find_best_threshold(y, proba, cost_fn=1_000, cost_fp=1, min_precision=0.0)
    cheap_miss = find_best_threshold(y, proba, cost_fn=1, cost_fp=1, min_precision=0.0)

    assert expensive_miss["threshold"] < cheap_miss["threshold"]
    assert expensive_miss["tp"] + expensive_miss["fp"] > cheap_miss["tp"] + cheap_miss["fp"]


# --- Comparaison sur données synthétiques -----------------------------------------------


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
    """Config réelle redirigée vers tmp_path, avec train/val et les 2 pipelines entraînés."""
    monkeypatch.chdir(tmp_path)  # artefacts MLflow par défaut dans tmp_path
    base = load_config(CONFIG_PATH)
    xgb = base.model.xgboost.model_copy(update={"n_estimators": 20})
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
            "model": base.model.model_copy(update={"xgboost": xgb}),
        }
    )

    target = config.columns.target
    train, val = _synthetic(config, 600, seed=1), _synthetic(config, 300, seed=2)
    config.paths.processed_dir.mkdir()
    train.to_csv(config.paths.processed_dir / "train.csv", index=False)
    val.to_csv(config.paths.processed_dir / "val.csv", index=False)

    X, y = train.drop(columns=[target]), train[target]
    config.paths.artifacts_dir.mkdir()
    pipelines = {
        "xgboost": build_xgb_pipeline(config, y),
        "logistic": build_logistic_baseline(config),
    }
    for name, pipeline in pipelines.items():
        joblib.dump(pipeline.fit(X, y), config.paths.artifacts_dir / f"{name}.joblib")
    return config


def test_compare_on_validation(cfg: Config) -> None:
    df = compare_on_validation(cfg)
    val = pd.read_csv(cfg.paths.processed_dir / "val.csv")
    n_failures = int(val[cfg.columns.target].sum())

    assert len(df) == 5
    assert df.loc[B0, "total_cost_eur"] == n_failures * cfg.decision.cost_missed_failure_eur
    assert df.loc[B0, "gain_vs_b0_eur"] == 0
    assert (df["detected"] + df["missed"] == n_failures).all()
    for approach in SCORED_MODELS:
        assert 0.0 <= df.loc[approach, "threshold"] <= 1.0
    assert df.loc[B0, ["threshold"]].isna().all()


def test_sensitivity_analysis(cfg: Config) -> None:
    ratios = (5, 50)
    df = sensitivity_analysis(cfg, ratios=ratios)

    assert len(df) == len(ratios) * 5
    for ratio, group in df.groupby("ratio_fn_fp"):
        assert (group["cost_fn_eur"] == ratio * cfg.decision.cost_false_alarm_eur).all()
        assert group["rank"].min() == 1
        best = group.loc[group["rank"] == 1, "total_cost_eur"].iloc[0]
        assert best == group["total_cost_eur"].min()


def test_main_never_reads_test_csv(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    assert not (cfg.paths.processed_dir / "test.csv").exists()
    read_paths: list[Path] = []
    real_read_csv = pd.read_csv

    def spy_read_csv(path: Any, *args: Any, **kwargs: Any) -> pd.DataFrame:
        read_paths.append(Path(path))
        return real_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(evaluate_module, "load_config", lambda: cfg)
    monkeypatch.setattr(pd, "read_csv", spy_read_csv)

    evaluate_module.main()

    assert {p.name for p in read_paths} == {"train.csv", "val.csv"}
    assert (cfg.paths.reports_dir / "validation_comparison.csv").is_file()
    assert (cfg.paths.reports_dir / "sensitivity.csv").is_file()

    thresholds = json.loads((cfg.paths.artifacts_dir / "thresholds.json").read_text("utf-8"))
    assert set(thresholds["thresholds"]) == {"xgboost", "logistic"}
    assert thresholds["cost_missed_failure_eur"] == cfg.decision.cost_missed_failure_eur
    assert {"generated_at", "git_commit"} <= set(thresholds)

    client = MlflowClient(tracking_uri=cfg.mlflow.tracking_uri)
    experiment = client.get_experiment_by_name(cfg.mlflow.experiment_name)
    assert experiment is not None
    run = client.search_runs([experiment.experiment_id])[0]
    assert run.info.run_name == "evaluation-validation"
    assert {"xgb_val_cost_eur", "xgb_threshold", "b2_val_cost_eur"} <= set(run.data.metrics)
    artifacts = {a.path for a in client.list_artifacts(run.info.run_id)}
    assert artifacts == {"validation_comparison.csv", "sensitivity.csv", "thresholds.json"}
