"""Tests du Predictor (artefacts complets construits dans tmp_path, sans les vrais fichiers)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest

from predictive_maintenance_iot.config import Config, load_config
from predictive_maintenance_iot.data.validation import validate_payload
from predictive_maintenance_iot.models.fingerprint import (
    FingerprintMismatchError,
    write_fingerprint,
)
from predictive_maintenance_iot.models.predict import (
    FEATURE_UNITS,
    Decision,
    Direction,
    Predictor,
    TriggeredBy,
    decide,
)
from predictive_maintenance_iot.models.train import build_xgb_pipeline

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
PREVENTIVE_WEAR_MIN = 200

# Mesures de référence : les pannes synthétiques dépendent du couple.
HIGH_RISK: dict[str, Any] = {
    "product_type": "L",
    "air_temperature_k": 300.0,
    "process_temperature_k": 310.0,
    "rotational_speed_rpm": 1500.0,
    "torque_nm": 115.0,
    "tool_wear_min": 50.0,
}
LOW_RISK: dict[str, Any] = {**HIGH_RISK, "product_type": "H", "torque_nm": 20.0}


def _synthetic(cfg: Config, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = cfg.columns
    data: dict[str, Any] = {
        cols.row_id: np.arange(n),
        "Product ID": [f"X{i:05d}" for i in range(n)],
        cols.product_type.column: rng.choice(cols.product_type.allowed, size=n),
    }
    for column in cols.numeric:
        bound = cfg.sensor_bounds[column]
        data[column] = rng.uniform(bound.min, bound.max, size=n).round(1)
    failures = (data["Torque [Nm]"] > 100).astype(int)
    data[cols.target] = failures
    for column in cols.leakage:
        data[column] = failures
    return pd.DataFrame(data)


def _write_thresholds(cfg: Config, threshold: float, **overrides: Any) -> None:
    payload = {
        "frozen_for_test": True,
        "primary_approach": "hybrid",
        "secondary_approach": "xgboost",
        "git_commit": "gel1234",
        "cost_missed_failure_eur": 3000,
        "cost_false_alarm_eur": 150,
        "min_precision": 0.5,
        "preventive_tool_wear_min": PREVENTIVE_WEAR_MIN,
        "thresholds": {
            "hybrid": {"threshold": threshold, "thresholds_by_ratio": {"20": threshold}},
            "xgboost": {"threshold": threshold, "thresholds_by_ratio": {"20": threshold}},
        },
        **overrides,
    }
    (cfg.paths.artifacts_dir / "thresholds.json").write_text(json.dumps(payload), "utf-8")


def build_artifacts(tmp_path: Path, max_batch_size: int = 3) -> Config:
    """Config + modèle + thresholds.json + empreinte dans tmp_path.

    Le seuil figé est placé à mi-chemin entre les probabilités de HIGH_RISK et LOW_RISK,
    pour que les tests de décision ne dépendent pas des valeurs exactes du petit modèle.
    """
    base = load_config(CONFIG_PATH)
    cfg = base.model_copy(
        update={
            "paths": base.paths.model_copy(
                update={
                    "artifacts_dir": tmp_path / "artifacts",
                    "reports_dir": tmp_path / "reports",
                }
            ),
            "model": base.model.model_copy(
                update={"xgboost": base.model.xgboost.model_copy(update={"n_estimators": 60})}
            ),
            "api": base.api.model_copy(
                update={
                    "inference_log_path": tmp_path / "logs" / "inference_log.db",
                    "max_batch_size": max_batch_size,
                }
            ),
        }
    )
    train = _synthetic(cfg, 800, seed=1)
    target = cfg.columns.target
    X, y = train.drop(columns=[target]), train[target]
    cfg.paths.artifacts_dir.mkdir(parents=True)
    joblib.dump(build_xgb_pipeline(cfg, y).fit(X, y), cfg.paths.artifacts_dir / "xgboost.joblib")
    write_fingerprint(cfg)

    _write_thresholds(cfg, 0.5)
    probe = Predictor.load(cfg)
    high, low = (
        p.failure_probability
        for p in probe.predict_batch(
            [validate_payload(HIGH_RISK, cfg), validate_payload(LOW_RISK, cfg)]
        )
    )
    assert high > low, "le petit modèle doit séparer les deux mesures de référence"
    _write_thresholds(cfg, (high + low) / 2)
    return cfg


@pytest.fixture(scope="module")
def cfg(tmp_path_factory: pytest.TempPathFactory) -> Config:
    return build_artifacts(tmp_path_factory.mktemp("predict"))


@pytest.fixture(scope="module")
def predictor(cfg: Config) -> Predictor:
    return Predictor.load(cfg)


# --- Règle de décision ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model_alert", "preventive", "decision", "triggered"),
    [
        (True, True, Decision.CRITIQUE, TriggeredBy.BOTH),
        (True, False, Decision.CRITIQUE, TriggeredBy.MODEL),
        (False, True, Decision.MAINTENANCE_PREVENTIVE, TriggeredBy.PREVENTIVE),
        (False, False, Decision.OPTIMAL, TriggeredBy.NONE),
    ],
)
def test_decide(
    model_alert: bool, preventive: bool, decision: Decision, triggered: TriggeredBy
) -> None:
    assert decide(model_alert, preventive) == (decision, triggered)


# --- Chargement -------------------------------------------------------------------------


def test_load_uses_frozen_values(cfg: Config, predictor: Predictor) -> None:
    frozen = json.loads((cfg.paths.artifacts_dir / "thresholds.json").read_text("utf-8"))
    assert predictor.threshold == frozen["thresholds"]["hybrid"]["threshold"]
    assert predictor.preventive_wear_min == PREVENTIVE_WEAR_MIN
    assert predictor.version.git_commit == "gel1234"
    assert len(predictor.version.model_sha256) == 64
    assert len(predictor.version.thresholds_sha256) == 64


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"frozen_for_test": False}, "frozen_for_test"),
        ({"primary_approach": "xgboost"}, "non prise en charge"),
    ],
)
def test_load_refuses_unfrozen_or_other_approach(
    tmp_path: Path, overrides: dict[str, Any], message: str
) -> None:
    cfg = build_artifacts(tmp_path)
    _write_thresholds(cfg, 0.5, **overrides)
    with pytest.raises(ValueError, match=message):
        Predictor.load(cfg)


def test_load_refuses_modified_model(tmp_path: Path) -> None:
    cfg = build_artifacts(tmp_path)
    model = cfg.paths.artifacts_dir / "xgboost.joblib"
    model.write_bytes(model.read_bytes() + b"\0")
    with pytest.raises(FingerprintMismatchError):
        Predictor.load(cfg)


# --- Prédictions ------------------------------------------------------------------------


def test_placeholder_columns_have_no_effect(cfg: Config, predictor: Predictor) -> None:
    X = predictor.to_model_frame([validate_payload(HIGH_RISK, cfg)])
    altered = X.copy()
    for column in [*cfg.columns.leakage, cfg.columns.row_id]:
        altered[column] = 999
    np.testing.assert_array_equal(
        predictor.pipeline.predict_proba(X), predictor.pipeline.predict_proba(altered)
    )


def test_high_probability_is_critical(cfg: Config, predictor: Predictor) -> None:
    prediction = predictor.predict(validate_payload(HIGH_RISK, cfg))

    assert prediction.failure_probability >= predictor.threshold
    assert prediction.decision is Decision.CRITIQUE
    assert prediction.triggered_by is TriggeredBy.MODEL
    assert prediction.recommended_action
    driver = next(
        (f for f in prediction.top_factors if f.direction is Direction.TOWARDS_FAILURE), None
    )
    actions = cfg.api.actions
    expected = (
        actions.by_feature.get(driver.feature, actions.critical_default)
        if driver
        else actions.critical_default
    )
    assert prediction.recommended_action == expected


def test_critical_wins_over_preventive(cfg: Config, predictor: Predictor) -> None:
    prediction = predictor.predict(validate_payload({**HIGH_RISK, "tool_wear_min": 250}, cfg))
    assert prediction.decision is Decision.CRITIQUE
    assert prediction.triggered_by is TriggeredBy.BOTH


def test_worn_tool_with_low_probability_is_preventive(cfg: Config, predictor: Predictor) -> None:
    prediction = predictor.predict(validate_payload({**LOW_RISK, "tool_wear_min": 250}, cfg))

    assert prediction.failure_probability < predictor.threshold
    assert prediction.decision is Decision.MAINTENANCE_PREVENTIVE
    assert prediction.triggered_by is TriggeredBy.PREVENTIVE
    assert prediction.recommended_action == cfg.api.actions.preventive


def test_low_probability_is_optimal(cfg: Config, predictor: Predictor) -> None:
    prediction = predictor.predict(validate_payload(LOW_RISK, cfg))
    assert prediction.decision is Decision.OPTIMAL
    assert prediction.recommended_action is None


def test_top_factors_sorted_by_absolute_contribution(cfg: Config, predictor: Predictor) -> None:
    factors = predictor.predict(validate_payload(HIGH_RISK, cfg)).top_factors

    assert 1 <= len(factors) <= 3
    magnitudes = [abs(f.contribution) for f in factors]
    assert magnitudes == sorted(magnitudes, reverse=True)
    for factor in factors:
        expected = (
            Direction.TOWARDS_FAILURE if factor.contribution > 0 else Direction.TOWARDS_NORMAL
        )
        assert factor.direction is expected
        assert factor.unit == FEATURE_UNITS[factor.feature]
    names = set(predictor.pipeline.named_steps["preprocess"].get_feature_names_out())
    assert {f.feature for f in factors} <= names


def test_crafted_contributions_label_order_and_action(cfg: Config, predictor: Predictor) -> None:
    # Contributions choisies à la main : le type produit pèse le plus, mais VERS LE NORMAL.
    payload = validate_payload(HIGH_RISK, cfg)
    features = predictor.pipeline.named_steps["preprocess"].transform(
        predictor.to_model_frame([payload])
    )
    crafted = {"product_type": -3.0, "power_w": 2.0, "torque_nm": 1.0}
    shap_row = np.array([crafted.get(str(c), 0.1) for c in features.columns])

    factors = predictor._top_factors(payload, features, shap_row, 0)

    assert [f.feature for f in factors] == ["product_type", "power_w", "torque_nm"]
    assert [f.direction for f in factors] == [
        Direction.TOWARDS_NORMAL,
        Direction.TOWARDS_FAILURE,
        Direction.TOWARDS_FAILURE,
    ]
    assert factors[0].value == "L"  # libellé, pas le code ordinal
    assert factors[0].unit is None
    assert factors[1].value == pytest.approx(round(115.0 * 1500.0 * 2 * np.pi / 60, 2))
    assert factors[1].unit == "W"
    assert factors[2].value == 115.0
    assert factors[2].unit == "Nm"
    # CRITIQUE : l'action suit le premier facteur « vers_panne » (power_w), pas product_type.
    assert predictor._action(Decision.CRITIQUE, factors) == cfg.api.actions.by_feature["power_w"]


def test_critical_without_failure_driver_uses_default(cfg: Config, predictor: Predictor) -> None:
    payload = validate_payload(HIGH_RISK, cfg)
    features = predictor.pipeline.named_steps["preprocess"].transform(
        predictor.to_model_frame([payload])
    )
    factors = predictor._top_factors(payload, features, -np.ones(len(features.columns)), 0)

    assert all(f.direction is Direction.TOWARDS_NORMAL for f in factors)
    assert predictor._action(Decision.CRITIQUE, factors) == cfg.api.actions.critical_default


def test_batch_matches_single_predictions(cfg: Config, predictor: Predictor) -> None:
    payloads = [validate_payload(p, cfg) for p in (HIGH_RISK, LOW_RISK)]
    batch = predictor.predict_batch(payloads)
    assert batch == [predictor.predict(p) for p in payloads]
    assert predictor.predict_batch([]) == []
