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
    assert config.columns.row_id == "UDI"
    assert config.columns.product_type.column == "Type"
    assert config.columns.product_type.allowed == ["L", "M", "H"]
    assert config.paths.reports_dir == Path("reports")
    assert config.baselines.preventive_tool_wear_min == 200
    assert config.baselines.expert_rules.osf_strain_max_min_nm == {
        "L": 11000,
        "M": 12000,
        "H": 13000,
    }
    assert config.baselines.logistic.max_iter == 1000


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


def test_row_id_must_be_in_ids(base_config: dict[str, Any], write_config: WriteConfig) -> None:
    base_config["columns"]["row_id"] = "Serial"
    with pytest.raises(ValidationError, match="row_id 'Serial' doit figurer dans ids"):
        load_config(write_config(base_config))


def test_product_type_column_must_be_in_categorical(
    base_config: dict[str, Any], write_config: WriteConfig
) -> None:
    base_config["columns"]["product_type"]["column"] = "Quality"
    with pytest.raises(ValidationError, match="'Quality' doit figurer dans categorical"):
        load_config(write_config(base_config))


def test_product_type_allowed_not_empty(
    base_config: dict[str, Any], write_config: WriteConfig
) -> None:
    base_config["columns"]["product_type"]["allowed"] = []
    with pytest.raises(ValidationError, match="ne doit pas être vide"):
        load_config(write_config(base_config))


def test_product_type_allowed_no_duplicates(
    base_config: dict[str, Any], write_config: WriteConfig
) -> None:
    base_config["columns"]["product_type"]["allowed"] = ["L", "M", "L"]
    with pytest.raises(ValidationError, match=r"doublons : \['L'\]"):
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


# --- baselines --------------------------------------------------------------------------


def _set_nested(data: dict[str, Any], dotted_key: str, value: Any) -> None:
    *parents, last = dotted_key.split(".")
    for key in parents:
        data = data[key]
    data[last] = value


@pytest.mark.parametrize(
    "key",
    [
        "baselines.preventive_tool_wear_min",
        "baselines.expert_rules.hdf_temp_diff_max_k",
        "baselines.expert_rules.hdf_speed_max_rpm",
        "baselines.expert_rules.pwf_power_min_w",
        "baselines.expert_rules.pwf_power_max_w",
        "baselines.expert_rules.osf_strain_max_min_nm.L",
        "baselines.expert_rules.twf_tool_wear_min",
        "baselines.logistic.C",
        "baselines.logistic.max_iter",
    ],
)
@pytest.mark.parametrize("value", [0, -1])
def test_baseline_values_must_be_positive(
    base_config: dict[str, Any], write_config: WriteConfig, key: str, value: float
) -> None:
    _set_nested(base_config, key, value)
    with pytest.raises(ValidationError, match="greater than 0"):
        load_config(write_config(base_config))


@pytest.mark.parametrize(("low", "high"), [(9000, 3500), (5000, 5000)])
def test_pwf_min_below_max(
    base_config: dict[str, Any], write_config: WriteConfig, low: float, high: float
) -> None:
    rules = base_config["baselines"]["expert_rules"]
    rules["pwf_power_min_w"], rules["pwf_power_max_w"] = low, high
    with pytest.raises(ValidationError, match="pwf_power_min_w .* doit être < pwf_power_max_w"):
        load_config(write_config(base_config))


@pytest.mark.parametrize(
    "thresholds",
    [
        {"L": 11000, "M": 12000},  # H manquant
        {"L": 11000, "M": 12000, "H": 13000, "X": 9000},  # type en trop
    ],
)
def test_osf_thresholds_match_product_types(
    base_config: dict[str, Any], write_config: WriteConfig, thresholds: dict[str, float]
) -> None:
    base_config["baselines"]["expert_rules"]["osf_strain_max_min_nm"] = thresholds
    with pytest.raises(ValidationError, match="doit contenir exactement les types"):
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
