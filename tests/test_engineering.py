"""Tests des variables physiques et du préprocesseur."""

from __future__ import annotations

import math
from pathlib import Path

import joblib
import pandas as pd
import pytest

from predictive_maintenance_iot.config import Config, load_config
from predictive_maintenance_iot.data.validation import COLUMN_MAP
from predictive_maintenance_iot.features.engineering import (
    NEW_FEATURES,
    PRODUCT_TYPE_FEATURE,
    PhysicalFeatures,
    RenameColumns,
    build_preprocessor,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yaml"

AIR = COLUMN_MAP["air_temperature_k"]
PROCESS = COLUMN_MAP["process_temperature_k"]
SPEED = COLUMN_MAP["rotational_speed_rpm"]
TORQUE = COLUMN_MAP["torque_nm"]
WEAR = COLUMN_MAP["tool_wear_min"]


@pytest.fixture(scope="module")
def cfg() -> Config:
    return load_config(CONFIG_PATH)


@pytest.fixture
def features(cfg: Config) -> PhysicalFeatures:
    return PhysicalFeatures.from_config(cfg)


@pytest.fixture
def sensors() -> pd.DataFrame:
    """Deux lignes aux valeurs choisies pour vérifier les formules à la main."""
    return pd.DataFrame(
        {
            AIR: [300.0, 298.0],
            PROCESS: [310.0, 308.5],
            SPEED: [1500, 1500],
            TORQUE: [40.0, 50.0],
            WEAR: [0, 200],
        }
    )


@pytest.fixture
def ai4i_df(cfg: Config) -> pd.DataFrame:
    """Petit extrait synthétique au format complet du CSV AI4I 2020."""
    cols = cfg.columns
    n = 6
    data: dict[str, object] = {
        cols.row_id: list(range(1, n + 1)),
        "Product ID": [f"X{i:05d}" for i in range(n)],
        cols.product_type.column: ["L", "M", "H", "L", "M", "H"],
        AIR: [298.1, 298.2, 298.9, 300.0, 301.5, 302.0],
        PROCESS: [308.6, 308.7, 309.1, 310.0, 311.2, 311.9],
        SPEED: [1551, 1408, 1433, 1500, 1380, 2861],
        TORQUE: [42.8, 46.3, 39.5, 40.0, 60.1, 4.6],
        WEAR: [0, 3, 5, 200, 210, 143],
        cols.target: [0, 0, 0, 0, 1, 1],
    }
    for column in cols.leakage:
        data[column] = [0] * n
    return pd.DataFrame(data)


# --- PhysicalFeatures -------------------------------------------------------------------


def test_from_config_uses_column_map(features: PhysicalFeatures) -> None:
    assert features.air_temp_col == AIR
    assert features.process_temp_col == PROCESS
    assert features.speed_col == SPEED
    assert features.torque_col == TORQUE
    assert features.tool_wear_col == WEAR


def test_known_values(features: PhysicalFeatures, sensors: pd.DataFrame) -> None:
    out = features.fit_transform(sensors)

    assert out.loc[0, "temp_diff_k"] == pytest.approx(10.0)
    assert out.loc[0, "power_w"] == pytest.approx(6283.19, abs=0.01)
    assert out.loc[0, "power_w"] == pytest.approx(40 * 1500 * 2 * math.pi / 60)
    assert out.loc[1, "strain_min_nm"] == pytest.approx(10_000.0)


def test_input_not_modified(features: PhysicalFeatures, sensors: pd.DataFrame) -> None:
    before = sensors.copy()
    out = features.fit_transform(sensors)

    pd.testing.assert_frame_equal(sensors, before)
    assert out is not sensors
    assert not set(NEW_FEATURES) & set(sensors.columns)


def test_transform_single_row(features: PhysicalFeatures, sensors: pd.DataFrame) -> None:
    features.fit(sensors)
    out = features.transform(sensors.iloc[[0]])

    assert len(out) == 1
    assert list(out.columns) == [*sensors.columns, *NEW_FEATURES]


@pytest.mark.parametrize("method", ["fit", "transform"])
def test_missing_column_raises(
    features: PhysicalFeatures, sensors: pd.DataFrame, method: str
) -> None:
    features.fit(sensors)
    incomplete = sensors.drop(columns=[TORQUE])
    with pytest.raises(ValueError, match=r"colonne\(s\) manquante\(s\).*Torque \[Nm\]"):
        getattr(features, method)(incomplete)


def test_get_feature_names_out(features: PhysicalFeatures, sensors: pd.DataFrame) -> None:
    features.fit(sensors)
    assert list(features.get_feature_names_out()) == [*sensors.columns, *NEW_FEATURES]


# --- build_preprocessor -----------------------------------------------------------------


def test_preprocessor_output_columns(ai4i_df: pd.DataFrame, cfg: Config) -> None:
    out = build_preprocessor(cfg).fit_transform(ai4i_df)
    cols = cfg.columns

    assert isinstance(out, pd.DataFrame)
    excluded = {*cols.ids, *cols.leakage, cols.target}
    assert not excluded & set(out.columns)
    assert set(NEW_FEATURES) <= set(out.columns)
    assert list(out.columns) == [
        "product_type",
        "air_temperature_k",
        "process_temperature_k",
        "rotational_speed_rpm",
        "torque_nm",
        "tool_wear_min",
        "temp_diff_k",
        "power_w",
        "strain_min_nm",
    ]


def test_preprocessor_names_are_xgboost_safe(ai4i_df: pd.DataFrame, cfg: Config) -> None:
    preprocessor = build_preprocessor(cfg).fit(ai4i_df)
    names = list(preprocessor.get_feature_names_out())

    assert names == list(preprocessor.transform(ai4i_df).columns)
    offending = [name for name in names if any(char in name for char in "[]<")]
    assert offending == []


def test_preprocessor_encodes_product_type(ai4i_df: pd.DataFrame, cfg: Config) -> None:
    out = build_preprocessor(cfg).fit_transform(ai4i_df)
    column = cfg.columns.product_type.column

    mapping = dict(zip(ai4i_df[column], out[PRODUCT_TYPE_FEATURE], strict=True))
    assert mapping == {"L": 0, "M": 1, "H": 2}


def test_rename_columns_keeps_unmapped_names() -> None:
    df = pd.DataFrame({"a [x]": [1.0], "b": [2.0]})
    renamer = RenameColumns({"a [x]": "a_x"}).fit(df)

    out = renamer.transform(df)

    assert list(out.columns) == ["a_x", "b"]
    assert list(renamer.get_feature_names_out()) == ["a_x", "b"]
    assert list(df.columns) == ["a [x]", "b"]  # entrée non modifiée


def test_preprocessor_joblib_roundtrip(ai4i_df: pd.DataFrame, cfg: Config, tmp_path: Path) -> None:
    pipeline = build_preprocessor(cfg).fit(ai4i_df)
    path = tmp_path / "preprocessor.joblib"

    joblib.dump(pipeline, path)
    reloaded = joblib.load(path)

    pd.testing.assert_frame_equal(reloaded.transform(ai4i_df), pipeline.transform(ai4i_df))


def test_preprocessor_without_derived_features(ai4i_df: pd.DataFrame, cfg: Config) -> None:
    preprocessor = build_preprocessor(cfg, include_derived=False)
    out = preprocessor.fit_transform(ai4i_df)

    assert not set(NEW_FEATURES) & set(out.columns)
    assert list(out.columns) == [
        "product_type",
        "air_temperature_k",
        "process_temperature_k",
        "rotational_speed_rpm",
        "torque_nm",
        "tool_wear_min",
    ]
    assert "physical_features" not in preprocessor.named_steps
    # Par défaut, les variables dérivées restent présentes.
    assert set(NEW_FEATURES) <= set(build_preprocessor(cfg).fit_transform(ai4i_df).columns)
