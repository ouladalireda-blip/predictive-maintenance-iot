"""Tests des baselines B1 (préventive), B2 (règles expertes) et B3 (logistique)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from predictive_maintenance_iot.config import Config, load_config
from predictive_maintenance_iot.data.validation import COLUMN_MAP
from predictive_maintenance_iot.models.baseline import (
    ExpertRulesBaseline,
    PreventiveBaseline,
    build_logistic_baseline,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yaml"

TYPE = "Type"
AIR = COLUMN_MAP["air_temperature_k"]
PROCESS = COLUMN_MAP["process_temperature_k"]
SPEED = COLUMN_MAP["rotational_speed_rpm"]
TORQUE = COLUMN_MAP["torque_nm"]
WEAR = COLUMN_MAP["tool_wear_min"]

# Ligne de référence qui ne déclenche aucune règle avec la config actuelle :
# temp_diff = 10 K (> 8,6), puissance ≈ 6 283 W (dans [3 500, 9 000]),
# strain = 4 000 min·N·m (< 12 000 pour M), usure = 100 min (< 200).
NORMAL: dict[str, Any] = {
    TYPE: "M",
    AIR: 300.0,
    PROCESS: 310.0,
    SPEED: 1500,
    TORQUE: 40.0,
    WEAR: 100,
}

# Chaque cas ne déclenche QUE la règle indiquée (valeurs calculées à la main).
RULE_CASES: dict[str, dict[str, Any]] = {
    "HDF": {PROCESS: 308.0, SPEED: 1300},  # diff 8 K < 8,6 et 1 300 < 1 380 rpm
    "PWF": {TORQUE: 20.0},  # 20 × 1 500 × 2π/60 ≈ 3 142 W < 3 500
    "OSF": {TYPE: "L", WEAR: 190, TORQUE: 60.0, SPEED: 1300},  # 11 400 > 11 000 ; P ≈ 8 168 W
    "TWF": {WEAR: 210},  # 210 >= 200 ; strain 8 400 < 12 000
}


def _frame(*overrides: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([{**NORMAL, **o} for o in overrides])


@pytest.fixture(scope="module")
def cfg() -> Config:
    return load_config(CONFIG_PATH)


@pytest.fixture
def expert(cfg: Config) -> ExpertRulesBaseline:
    return ExpertRulesBaseline.from_config(cfg)


@pytest.fixture
def ai4i_df(cfg: Config) -> pd.DataFrame:
    """200 lignes synthétiques au format complet du CSV, avec ~10 % de pannes."""
    rng = np.random.default_rng(0)
    cols = cfg.columns
    n = 200
    data: dict[str, Any] = {
        cols.row_id: np.arange(1, n + 1),
        "Product ID": [f"X{i:05d}" for i in range(n)],
        cols.product_type.column: rng.choice(cols.product_type.allowed, size=n),
    }
    for column in cols.numeric:
        bound = cfg.sensor_bounds[column]
        data[column] = rng.uniform(bound.min, bound.max, size=n).round(1)
    failures = np.zeros(n, dtype=int)
    failures[:20] = 1
    data[cols.target] = failures
    for column in cols.leakage:
        data[column] = failures
    return pd.DataFrame(data)


# --- B1 : préventive --------------------------------------------------------------------


@pytest.mark.parametrize(("wear", "expected"), [(199, 0), (200, 1), (250, 1)])
def test_preventive_threshold(cfg: Config, wear: int, expected: int) -> None:
    model = PreventiveBaseline.from_config(cfg).fit(_frame({WEAR: wear}))
    assert model.predict(_frame({WEAR: wear})).tolist() == [expected]


# --- B2 : règles expertes ---------------------------------------------------------------


def test_normal_row_triggers_nothing(expert: ExpertRulesBaseline) -> None:
    X = _frame({})
    assert expert.fit(X).predict(X).tolist() == [0]
    assert expert.explain(X) == [[]]


@pytest.mark.parametrize("rule", list(RULE_CASES))
def test_each_rule_triggers_alone(expert: ExpertRulesBaseline, rule: str) -> None:
    X = _frame(RULE_CASES[rule])
    assert expert.fit(X).predict(X).tolist() == [1]
    assert expert.explain(X) == [[rule]]


def test_pwf_also_triggers_above_max(expert: ExpertRulesBaseline) -> None:
    X = _frame({TORQUE: 60.0, SPEED: 1500, WEAR: 0})  # 60 × 1 500 × 2π/60 ≈ 9 425 W > 9 000
    assert expert.explain(X) == [["PWF"]]


@pytest.mark.parametrize(("product_type", "expected"), [("L", 1), ("M", 1), ("H", 0)])
def test_osf_threshold_depends_on_product_type(
    expert: ExpertRulesBaseline, product_type: str, expected: int
) -> None:
    # Même strain pour les 3 types : 195 × 64 = 12 480 min·N·m (seuils 11 000 / 12 000 / 13 000).
    # P = 64 × 1 300 × 2π/60 ≈ 8 713 W (pas de PWF), usure 195 < 200 (pas de TWF).
    X = _frame({TYPE: product_type, WEAR: 195, TORQUE: 64.0, SPEED: 1300})
    assert expert.fit(X).predict(X).tolist() == [expected]


def test_explain_lists_all_triggered_rules(expert: ExpertRulesBaseline) -> None:
    X = _frame({}, RULE_CASES["HDF"], {**RULE_CASES["PWF"], WEAR: 210}, RULE_CASES["OSF"])
    assert expert.explain(X) == [[], ["HDF"], ["PWF", "TWF"], ["OSF"]]


def test_unknown_product_type_raises(expert: ExpertRulesBaseline) -> None:
    with pytest.raises(ValueError, match="aucun seuil OSF"):
        expert.predict(_frame({TYPE: "X"}))


def test_missing_column_raises(expert: ExpertRulesBaseline) -> None:
    with pytest.raises(ValueError, match=r"colonne\(s\) manquante\(s\)"):
        expert.fit(_frame({}).drop(columns=[TORQUE]))


# --- predict_proba (les 3 baselines) ----------------------------------------------------


@pytest.mark.parametrize("name", ["preventive", "expert", "logistic"])
def test_predict_proba_shape_and_sum(ai4i_df: pd.DataFrame, cfg: Config, name: str) -> None:
    target = cfg.columns.target
    X, y = ai4i_df.drop(columns=[target]), ai4i_df[target]
    builders = {
        "preventive": PreventiveBaseline.from_config,
        "expert": ExpertRulesBaseline.from_config,
        "logistic": build_logistic_baseline,
    }
    model = builders[name](cfg).fit(X, y)

    proba = model.predict_proba(X)

    assert proba.shape == (len(X), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert ((proba >= 0) & (proba <= 1)).all()


# --- B3 : régression logistique ---------------------------------------------------------


def test_logistic_trains_without_leakage(ai4i_df: pd.DataFrame, cfg: Config) -> None:
    cols = cfg.columns
    X, y = ai4i_df.drop(columns=[cols.target]), ai4i_df[cols.target]

    pipeline = build_logistic_baseline(cfg).fit(X, y)

    used = list(pipeline.named_steps["preprocess"].get_feature_names_out())
    assert not {*cols.leakage, *cols.ids, cols.target} & set(used)
    assert list(pipeline.named_steps["scale"].feature_names_in_) == used
    assert pipeline.named_steps["model"].coef_.shape == (1, len(used))
    assert set(pipeline.predict(X)) <= {0, 1}
