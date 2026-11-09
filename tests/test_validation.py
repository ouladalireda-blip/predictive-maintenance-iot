"""Tests de la validation des mesures unitaires et du fichier brut."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from pandera.errors import SchemaErrors
from pydantic import ValidationError

from predictive_maintenance_iot.config import Config, load_config
from predictive_maintenance_iot.data.validation import (
    SensorPayload,
    validate_payload,
    validate_raw,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "config.yaml"
RAW_CSV_PATH = ROOT / "data" / "raw" / "ai4i2020.csv"


@pytest.fixture(scope="module")
def cfg() -> Config:
    return load_config(CONFIG_PATH)


@pytest.fixture
def payload() -> dict[str, Any]:
    return {
        "product_type": "M",
        "air_temperature_k": 300.0,
        "process_temperature_k": 310.0,
        "rotational_speed_rpm": 1500.0,
        "torque_nm": 40.0,
        "tool_wear_min": 100.0,
    }


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Petit extrait valide au format du CSV brut (utilisable sans le vrai fichier)."""
    return pd.DataFrame(
        {
            "UDI": [1, 2, 3],
            "Product ID": ["M14860", "L47181", "H29424"],
            "Type": ["M", "L", "H"],
            "Air temperature [K]": [298.1, 298.2, 298.9],
            "Process temperature [K]": [308.6, 308.7, 309.1],
            "Rotational speed [rpm]": [1551, 1408, 1433],
            "Torque [Nm]": [42.8, 46.3, 39.5],
            "Tool wear [min]": [0, 3, 5],
            "Machine failure": [0, 0, 1],
            "TWF": [0, 0, 0],
            "HDF": [0, 0, 1],
            "PWF": [0, 0, 0],
            "OSF": [0, 0, 0],
            "RNF": [0, 0, 0],
        }
    )


# --- SensorPayload ----------------------------------------------------------------------


def test_valid_payload_accepted(payload: dict[str, Any], cfg: Config) -> None:
    result = validate_payload(payload, cfg)
    assert isinstance(result, SensorPayload)
    assert result.torque_nm == 40.0


@pytest.mark.parametrize("side", ["min", "max"])
def test_value_equal_to_bound_accepted(payload: dict[str, Any], cfg: Config, side: str) -> None:
    bound = cfg.sensor_bounds["Torque [Nm]"]
    payload["torque_nm"] = bound.min if side == "min" else bound.max
    assert validate_payload(payload, cfg).torque_nm == payload["torque_nm"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("torque_nm", -5.0),  # couple négatif
        ("process_temperature_k", 500.0),  # température trop haute
    ],
)
def test_out_of_bounds_rejected(
    payload: dict[str, Any], cfg: Config, field: str, value: float
) -> None:
    payload[field] = value
    with pytest.raises(ValidationError, match=rf"{field}={value} hors bornes"):
        validate_payload(payload, cfg)


def test_invalid_product_type_rejected(payload: dict[str, Any], cfg: Config) -> None:
    payload["product_type"] = "X"
    with pytest.raises(ValidationError) as exc_info:
        validate_payload(payload, cfg)
    message = str(exc_info.value)
    assert "product_type='X' non autorisé" in message
    assert str(cfg.columns.product_type.allowed) in message


def test_missing_field_rejected(payload: dict[str, Any], cfg: Config) -> None:
    del payload["tool_wear_min"]
    with pytest.raises(ValidationError, match="tool_wear_min"):
        validate_payload(payload, cfg)


def test_extra_field_rejected(payload: dict[str, Any], cfg: Config) -> None:
    payload["humidity"] = 0.4
    with pytest.raises(ValidationError, match="humidity"):
        validate_payload(payload, cfg)


def test_validation_without_context_rejected(payload: dict[str, Any]) -> None:
    with pytest.raises(ValidationError, match="contexte de validation incomplet"):
        SensorPayload.model_validate(payload)


@pytest.mark.parametrize("missing_key", ["bounds", "product_types"])
def test_validation_with_incomplete_context_rejected(
    payload: dict[str, Any], cfg: Config, missing_key: str
) -> None:
    context: dict[str, Any] = {
        "bounds": cfg.sensor_bounds,
        "product_types": cfg.columns.product_type.allowed,
    }
    del context[missing_key]
    with pytest.raises(ValidationError, match=missing_key):
        SensorPayload.model_validate(payload, context=context)


# --- Fichier brut -----------------------------------------------------------------------


def test_real_csv_passes(cfg: Config) -> None:
    if not RAW_CSV_PATH.is_file():
        pytest.skip(f"{RAW_CSV_PATH} absent (non versionné) : test ignoré")
    df = pd.read_csv(RAW_CSV_PATH)
    validated = validate_raw(df, cfg)
    assert len(validated) == len(df)


def test_valid_extract_passes(raw_df: pd.DataFrame, cfg: Config) -> None:
    validate_raw(raw_df, cfg)


def test_corrupted_df_reports_all_errors(
    raw_df: pd.DataFrame, cfg: Config, caplog: pytest.LogCaptureFixture
) -> None:
    raw_df.loc[0, "Torque [Nm]"] = -5.0
    raw_df.loc[1, "Air temperature [K]"] = math.nan
    raw_df.loc[2, "UDI"] = raw_df.loc[0, "UDI"]

    with pytest.raises(SchemaErrors) as exc_info:
        validate_raw(raw_df, cfg)

    failures = exc_info.value.failure_cases
    failed = set(zip(failures["column"], failures["check"], strict=True))
    assert any(col == "Torque [Nm]" and "in_range" in str(chk) for col, chk in failed)
    assert ("Air temperature [K]", "not_nullable") in failed
    assert ("UDI", "field_uniqueness") in failed
    assert "erreur(s) détectée(s)" in caplog.text
