"""Tests des analyses complémentaires (stabilité, bootstrap, erreurs, ablation, SHAP, main)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml
from mlflow.tracking import MlflowClient
from pydantic import ValidationError

from predictive_maintenance_iot.config import Config, load_config
from predictive_maintenance_iot.models import analysis as analysis_module
from predictive_maintenance_iot.models.analysis import (
    NO_MODE,
    bootstrap_costs,
    cross_validate_models,
    error_analysis,
    summarize_cv,
)
from predictive_maintenance_iot.models.evaluate import HYBRID
from predictive_maintenance_iot.models.fingerprint import (
    FingerprintMismatchError,
    write_fingerprint,
)
from predictive_maintenance_iot.models.train import build_xgb_pipeline

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


# --- Section analysis de la config ------------------------------------------------------


def test_real_config_analysis_section() -> None:
    analysis = load_config(CONFIG_PATH).analysis
    assert (analysis.cv_folds, analysis.bootstrap_iterations, analysis.shap_examples) == (
        5,
        1000,
        2,
    )


@pytest.mark.parametrize(
    ("key", "value"),
    [("cv_folds", 1), ("bootstrap_iterations", 0), ("shap_examples", 0), ("shap_examples", -1)],
)
def test_analysis_section_validation(tmp_path: Path, key: str, value: int) -> None:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["analysis"][key] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(path)


# --- Bootstrap --------------------------------------------------------------------------


@pytest.fixture
def boot_inputs() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.08).astype(int)
    alerts = {
        "B2 regles expertes": (y == 1) & (rng.random(400) < 0.9) | (rng.random(400) < 0.05),
        "XGBoost": (y == 1) & (rng.random(400) < 0.7) | (rng.random(400) < 0.02),
    }
    return y, alerts


def test_bootstrap_is_reproducible(boot_inputs: tuple[np.ndarray, dict[str, np.ndarray]]) -> None:
    y, alerts = boot_inputs
    first = bootstrap_costs(y, alerts, 3000, 150, n_iterations=200, seed=42)
    second = bootstrap_costs(y, alerts, 3000, 150, n_iterations=200, seed=42)
    other = bootstrap_costs(y, alerts, 3000, 150, n_iterations=200, seed=7)

    pd.testing.assert_frame_equal(first, second)
    assert not first.equals(other)


def test_bootstrap_interval_contains_median(
    boot_inputs: tuple[np.ndarray, dict[str, np.ndarray]],
) -> None:
    y, alerts = boot_inputs
    df = bootstrap_costs(y, alerts, 3000, 150, n_iterations=300, seed=42)

    assert set(df["metric"]) == {"total_cost_eur", "gain_vs_b0_eur", "share_of_reference_gain"}
    assert ((df["ci95_low"] <= df["median"]) & (df["median"] <= df["ci95_high"])).all()
    # La référence (B2) représente par définition 100 % de son propre gain.
    b2_share = df[(df["approach"] == "B2 regles expertes")]
    assert b2_share.loc[b2_share["metric"] == "share_of_reference_gain", "median"].item() == 1.0


def test_bootstrap_without_resampling_noise_matches_direct_cost() -> None:
    # Toutes les lignes identiques : chaque rééchantillon donne exactement le même coût.
    y = np.ones(10, dtype=int)
    alerts = {"B2 regles expertes": np.ones(10, dtype=bool), "XGBoost": np.zeros(10, bool)}
    df = bootstrap_costs(y, alerts, 3000, 150, n_iterations=50, seed=0).set_index(
        ["approach", "metric"]
    )
    assert df.loc[("XGBoost", "total_cost_eur"), "median"] == 30_000
    assert df.loc[("XGBoost", "total_cost_eur"), "ci95_low"] == 30_000
    assert df.loc[("B2 regles expertes", "gain_vs_b0_eur"), "median"] == 30_000


# --- Analyse d'erreurs ------------------------------------------------------------------


def test_error_analysis_hand_built_case() -> None:
    cfg = load_config(CONFIG_PATH)
    # Lignes : 0 = TWF seule (détectée), 1 = HDF + OSF (ratée), 2 = panne sans mode (détectée),
    #          3 = RNF actif mais cible 0 (anomalie AI4I), 4 = normale.
    df = pd.DataFrame(
        {
            cfg.columns.target: [1, 1, 1, 0, 0],
            "TWF": [1, 0, 0, 0, 0],
            "HDF": [0, 1, 0, 0, 0],
            "PWF": [0, 0, 0, 0, 0],
            "OSF": [0, 1, 0, 0, 0],
            "RNF": [0, 0, 0, 1, 0],
        }
    )
    alerts = {"model": np.array([True, False, True, True, False])}

    out = error_analysis(df, alerts, cfg)

    assert list(out.index) == ["TWF", "HDF", "PWF", "OSF", "RNF", NO_MODE]
    assert out.loc["TWF", ["failures", "detected_model", "missed_model"]].tolist() == [1, 1, 0]
    for mode in ("HDF", "OSF"):  # panne à deux modes : comptée dans chacun, ratée
        assert out.loc[mode, ["failures", "multi_mode", "missed_model"]].tolist() == [1, 1, 1]
    assert out.loc["PWF", "failures"] == 0
    assert out.loc["RNF", "failures"] == 0
    assert out.loc["RNF", "mode_active_target_0"] == 1
    assert out.loc[NO_MODE, ["failures", "detected_model"]].tolist() == [1, 1]
    assert np.isnan(out.loc[NO_MODE, "mode_active_target_0"])


# --- Données synthétiques et main -------------------------------------------------------


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
    failures = (torque > np.quantile(torque, 0.9)).astype(int)
    data[cols.target] = failures
    for i, column in enumerate(cols.leakage):  # un mode par panne, en alternance
        data[column] = failures * (np.arange(n) % len(cols.leakage) == i)
    return pd.DataFrame(data)


@pytest.fixture
def cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Config:
    """Config redirigée vers tmp_path, train/val, xgboost.joblib et thresholds.json figés."""
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
            "analysis": base.analysis.model_copy(
                update={"cv_folds": 3, "bootstrap_iterations": 50}
            ),
        }
    )
    target = config.columns.target
    train, val = _synthetic(config, 600, seed=1), _synthetic(config, 300, seed=2)
    config.paths.processed_dir.mkdir()
    train.to_csv(config.paths.processed_dir / "train.csv", index=False)
    val.to_csv(config.paths.processed_dir / "val.csv", index=False)

    config.paths.artifacts_dir.mkdir()
    X, y = train.drop(columns=[target]), train[target]
    joblib.dump(
        build_xgb_pipeline(config, y).fit(X, y), config.paths.artifacts_dir / "xgboost.joblib"
    )
    thresholds = {
        "frozen_for_test": True,
        "primary_approach": "hybrid",
        "secondary_approach": "xgboost",
        "preventive_tool_wear_min": 200,
        "thresholds": {"xgboost": {"threshold": 0.5}, "hybrid": {"threshold": 0.6}},
    }
    (config.paths.artifacts_dir / "thresholds.json").write_text(
        json.dumps(thresholds), encoding="utf-8"
    )
    write_fingerprint(config)  # modèle figé : empreinte de référence
    return config


def test_cross_validation_uses_train_only(cfg: Config) -> None:
    train = pd.read_csv(cfg.paths.processed_dir / "train.csv")

    folds = cross_validate_models(cfg, train)

    for _, group in folds.groupby("model"):
        assert len(group) == cfg.analysis.cv_folds
        assert group["n_eval"].sum() == len(train)  # chaque ligne de TRAIN évaluée une fois
    summary = summarize_cv(folds)
    assert set(summary.index) == {"xgboost", "logistic", "xgboost_sans_derivees"}
    assert {"pr_auc_mean", "pr_auc_std", "roc_auc_mean", "roc_auc_std"} <= set(summary.columns)


def test_main_never_reads_test_csv_and_keeps_thresholds(
    cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    thresholds_path = cfg.paths.artifacts_dir / "thresholds.json"
    before = hashlib.sha256(thresholds_path.read_bytes()).hexdigest()
    assert not (cfg.paths.processed_dir / "test.csv").exists()

    read_paths: list[Path] = []
    real_read_csv = pd.read_csv

    def spy_read_csv(path: Any, *args: Any, **kwargs: Any) -> pd.DataFrame:
        read_paths.append(Path(path))
        return real_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(analysis_module, "load_config", lambda: cfg)
    monkeypatch.setattr(pd, "read_csv", spy_read_csv)

    analysis_module.main()

    assert {p.name for p in read_paths} == {"train.csv", "val.csv"}
    assert hashlib.sha256(thresholds_path.read_bytes()).hexdigest() == before

    out_dir = cfg.paths.reports_dir / "analysis"
    expected = {
        "cv_folds.csv",
        "cv_summary.csv",
        "bootstrap_costs.csv",
        "error_analysis.csv",
        "ablation.csv",
        "shap_importance.csv",
        "shap_examples.csv",
        "shap_importance.png",
    }
    assert {p.name for p in out_dir.iterdir()} == expected
    assert (out_dir / "shap_importance.png").stat().st_size > 0

    ablation = real_read_csv(out_dir / "ablation.csv", index_col=0)
    assert set(ablation.index) == {"xgboost", "xgboost_sans_derivees"}
    shap_examples = real_read_csv(out_dir / "shap_examples.csv")
    assert len(shap_examples) <= 3 * cfg.analysis.shap_examples

    client = MlflowClient(tracking_uri=cfg.mlflow.tracking_uri)
    experiment = client.get_experiment_by_name(cfg.mlflow.experiment_name)
    assert experiment is not None
    run = client.search_runs([experiment.experiment_id])[0]
    assert run.info.run_name == "analysis-validation"
    logged = {a.path for a in client.list_artifacts(run.info.run_id, "analysis")}
    assert logged == {f"analysis/{name}" for name in expected}


# --- Valeurs figées : usure préventive et empreinte du modèle ----------------------------


def test_hybrid_uses_frozen_preventive_wear(cfg: Config) -> None:
    train = pd.read_csv(cfg.paths.processed_dir / "train.csv")
    val = pd.read_csv(cfg.paths.processed_dir / "val.csv")
    pipeline = joblib.load(cfg.paths.artifacts_dir / "xgboost.joblib")
    wear = val["Tool wear [min]"]
    assert cfg.baselines.preventive_tool_wear_min == 200
    assert ((wear >= 100) & (wear < 200)).any()  # lignes où 100 et 200 donnent un écart

    # Seuil XGBoost > 1 : l'hybride se réduit à son alerte préventive.
    frozen = {
        "preventive_tool_wear_min": 100,
        "thresholds": {"xgboost": {"threshold": 0.5}, "hybrid": {"threshold": 1.1}},
    }
    _, alerts = analysis_module._validation_alerts(cfg, train, val, pipeline, frozen)

    np.testing.assert_array_equal(alerts[HYBRID], (wear >= 100).to_numpy())  # JSON, pas config


def test_missing_frozen_preventive_wear_raises(cfg: Config) -> None:
    train = pd.read_csv(cfg.paths.processed_dir / "train.csv")
    val = pd.read_csv(cfg.paths.processed_dir / "val.csv")
    pipeline = joblib.load(cfg.paths.artifacts_dir / "xgboost.joblib")
    frozen = {"thresholds": {"xgboost": {"threshold": 0.5}, "hybrid": {"threshold": 0.6}}}

    with pytest.raises(KeyError, match="preventive_tool_wear_min"):
        analysis_module._validation_alerts(cfg, train, val, pipeline, frozen)


def test_main_refuses_modified_model(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    model_path = cfg.paths.artifacts_dir / "xgboost.joblib"
    model_path.write_bytes(model_path.read_bytes() + b"\0")  # modèle altéré après le gel

    read_paths: list[Any] = []
    monkeypatch.setattr(analysis_module, "load_config", lambda: cfg)
    monkeypatch.setattr(pd, "read_csv", lambda path, *a, **k: read_paths.append(path))

    with pytest.raises(FingerprintMismatchError):
        analysis_module.main()
    assert read_paths == []  # arrêt avant toute lecture de données
    assert not (cfg.paths.reports_dir / "analysis").exists()
