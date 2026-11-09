"""Tests du chargement et des validations de la configuration."""

from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from predictive_maintenance_iot.config import Config, load_config

REAL_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


@pytest.fixture
def base_config() -> dict[str, Any]:
    """Copie modifiable du contenu réel de config/config.yaml."""
    raw = yaml.safe_load(REAL_CONFIG_PATH.read_text(encoding="utf-8"))
    return copy.deepcopy(raw)


@pytest.fixture
def write_config(tmp_path: Path) -> Callable[[dict[str, Any]], Path]:
    def _write(data: dict[str, Any]) -> Path:
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
        return path

    return _write


WriteConfig = Callable[[dict[str, Any]], Path]


# --- Chargement -------------------------------------------------------------------------


def test_load_real_config() -> None:
    config = load_config(REAL_CONFIG_PATH)

    assert isinstance(config, Config)
    assert config.columns.target == "Machine failure"
    assert set(config.sensor_bounds) == set(config.columns.numeric)
    assert config.model.xgboost.scale_pos_weight == "auto"


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="introuvable"):
        load_config(tmp_path / "absent.yaml")


def test_invalid_yaml_raises(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("project: [non fermé\n", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML invalide"):
        load_config(path)


def test_non_mapping_root_raises(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_config(path)


def test_unknown_key_raises(base_config: dict[str, Any], write_config: WriteConfig) -> None:
    base_config["split"]["shuffle"] = True
    with pytest.raises(ValidationError):
        load_config(write_config(base_config))


# --- split ------------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["test_size", "val_size"])
@pytest.mark.parametrize("value", [0.0, 0.5, 0.9, -0.1])
def test_split_size_out_of_range(
    base_config: dict[str, Any], write_config: WriteConfig, field: str, value: float
) -> None:
    base_config["split"][field] = value
    with pytest.raises(ValidationError):
        load_config(write_config(base_config))


def test_split_sizes_sum_too_large(base_config: dict[str, Any], write_config: WriteConfig) -> None:
    base_config["split"]["test_size"] = 0.3
    base_config["split"]["val_size"] = 0.3
    with pytest.raises(ValidationError, match="< 0.6"):
        load_config(write_config(base_config))


# --- sensor_bounds ----------------------------------------------------------------------


@pytest.mark.parametrize(("low", "high"), [(100.0, 50.0), (10.0, 10.0)])
def test_sensor_bound_min_not_below_max(
    base_config: dict[str, Any], write_config: WriteConfig, low: float, high: float
) -> None:
    base_config["sensor_bounds"]["Torque [Nm]"] = {"min": low, "max": high}
    with pytest.raises(ValidationError, match="borne min"):
        load_config(write_config(base_config))


# --- columns ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("feature_group", "column"),
    [
        ("numeric", "TWF"),  # leakage
        ("categorical", "Product ID"),  # id
        ("numeric", "Machine failure"),  # target
    ],
)
def test_forbidden_column_in_features(
    base_config: dict[str, Any], write_config: WriteConfig, feature_group: str, column: str
) -> None:
    base_config["columns"][feature_group].append(column)
    with pytest.raises(ValidationError, match="colonnes interdites"):
        load_config(write_config(base_config))


# --- model.xgboost.scale_pos_weight -----------------------------------------------------


@pytest.mark.parametrize("value", ["auto", 2.5, 28])
def test_scale_pos_weight_valid(
    base_config: dict[str, Any], write_config: WriteConfig, value: str | float
) -> None:
    base_config["model"]["xgboost"]["scale_pos_weight"] = value
    config = load_config(write_config(base_config))
    assert config.model.xgboost.scale_pos_weight == value


@pytest.mark.parametrize("value", [0, -1.0, "manual"])
def test_scale_pos_weight_invalid(
    base_config: dict[str, Any], write_config: WriteConfig, value: str | float
) -> None:
    base_config["model"]["xgboost"]["scale_pos_weight"] = value
    with pytest.raises(ValidationError):
        load_config(write_config(base_config))


# --- decision ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["cost_missed_failure_eur", "cost_false_alarm_eur"])
@pytest.mark.parametrize("value", [0, -150])
def test_costs_must_be_positive(
    base_config: dict[str, Any], write_config: WriteConfig, field: str, value: float
) -> None:
    base_config["decision"][field] = value
    with pytest.raises(ValidationError):
        load_config(write_config(base_config))


@pytest.mark.parametrize("value", [-0.1, 1.5])
def test_min_precision_out_of_range(
    base_config: dict[str, Any], write_config: WriteConfig, value: float
) -> None:
    base_config["decision"]["min_precision"] = value
    with pytest.raises(ValidationError):
        load_config(write_config(base_config))


# --- mlflow.experiment_name -------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "predictive maintenance",  # espace
        "maintenance-(v2)",  # parenthèses
        "Predictive-Maintenance",  # majuscules
        "pm_iot",  # underscore
        "ab",  # trop court
        "a" * 65,  # trop long
    ],
)
def test_experiment_name_invalid(
    base_config: dict[str, Any], write_config: WriteConfig, value: str
) -> None:
    base_config["mlflow"]["experiment_name"] = value
    with pytest.raises(ValidationError):
        load_config(write_config(base_config))


@pytest.mark.parametrize("value", ["pm-iot", "predictive-maintenance-ai4i", "a" * 64])
def test_experiment_name_valid(
    base_config: dict[str, Any], write_config: WriteConfig, value: str
) -> None:
    base_config["mlflow"]["experiment_name"] = value
    config = load_config(write_config(base_config))
    assert config.mlflow.experiment_name == value
