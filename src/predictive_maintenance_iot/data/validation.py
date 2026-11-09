"""Validation du schéma et de la qualité des données (contrats de données).

Deux niveaux de contrôle, tous deux pilotés par les bornes de `config.yaml` :
- `SensorPayload` : une mesure unitaire reçue par l'API (noms snake_case) ;
- `build_raw_schema` / `validate_raw` : le fichier brut AI4I 2020 complet (noms du CSV).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Self

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors
from pydantic import BaseModel, ConfigDict, ValidationInfo, model_validator

from predictive_maintenance_iot.config import Config, SensorBound

logger = logging.getLogger(__name__)

# Nom snake_case exposé par l'API → nom de colonne dans le CSV (et dans config.sensor_bounds).
COLUMN_MAP: dict[str, str] = {
    "air_temperature_k": "Air temperature [K]",
    "process_temperature_k": "Process temperature [K]",
    "rotational_speed_rpm": "Rotational speed [rpm]",
    "torque_nm": "Torque [Nm]",
    "tool_wear_min": "Tool wear [min]",
}

BINARY_VALUES: tuple[int, ...] = (0, 1)
REQUIRED_CONTEXT_KEYS: tuple[str, ...] = ("bounds", "product_types")


class SensorPayload(BaseModel):
    """Une mesure capteur unitaire, validée contre les règles de la config.

    Rien n'est codé en dur : bornes et types de produit autorisés doivent être fournis via
    `model_validate(..., context={"bounds": ..., "product_types": ...})`
    (voir `validate_payload`).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    product_type: str
    air_temperature_k: float
    process_temperature_k: float
    rotational_speed_rpm: float
    torque_nm: float
    tool_wear_min: float

    @model_validator(mode="after")
    def _check_against_config(self, info: ValidationInfo) -> Self:
        context: Mapping[str, Any] = info.context if isinstance(info.context, Mapping) else {}
        missing = [key for key in REQUIRED_CONTEXT_KEYS if key not in context]
        if missing:
            raise ValueError(
                f"contexte de validation incomplet, clé(s) manquante(s) : {missing}. "
                "Utiliser validate_payload(data, cfg)"
            )
        bounds: Mapping[str, SensorBound] = context["bounds"]
        product_types: list[str] = list(context["product_types"])

        if self.product_type not in product_types:
            raise ValueError(
                f"product_type={self.product_type!r} non autorisé "
                f"(valeurs admises : {product_types})"
            )

        violations: list[str] = []
        for field, column in COLUMN_MAP.items():
            if column not in bounds:
                raise ValueError(f"borne manquante dans la config pour '{column}' ({field})")
            bound = bounds[column]
            value: float = getattr(self, field)
            if not bound.min <= value <= bound.max:
                violations.append(f"{field}={value} hors bornes [{bound.min}, {bound.max}]")

        if violations:
            raise ValueError("; ".join(violations))
        return self


def validate_payload(data: dict[str, Any], cfg: Config) -> SensorPayload:
    """Valide une mesure brute (dict) avec les bornes et types de produit de la config."""
    return SensorPayload.model_validate(
        data,
        context={
            "bounds": cfg.sensor_bounds,
            "product_types": cfg.columns.product_type.allowed,
        },
    )


def build_raw_schema(cfg: Config) -> pa.DataFrameSchema:
    """Construit le schéma pandera du fichier brut à partir de la config.

    Toutes les colonnes listées dans la config sont obligatoires et non nulles.
    """
    columns: dict[str, pa.Column] = {}

    product_type = cfg.columns.product_type

    for column in cfg.columns.ids:
        columns[column] = pa.Column(nullable=False, unique=column == cfg.columns.row_id)

    for column in cfg.columns.categorical:
        checks = [pa.Check.isin(product_type.allowed)] if column == product_type.column else []
        columns[column] = pa.Column(checks=checks, nullable=False)

    for column in cfg.columns.numeric:
        if column not in cfg.sensor_bounds:
            raise ValueError(f"aucune borne définie dans sensor_bounds pour '{column}'")
        bound = cfg.sensor_bounds[column]
        columns[column] = pa.Column(
            float,
            checks=pa.Check.in_range(bound.min, bound.max),
            nullable=False,
            coerce=True,
        )

    for column in [cfg.columns.target, *cfg.columns.leakage]:
        columns[column] = pa.Column(checks=pa.Check.isin(BINARY_VALUES), nullable=False)

    return pa.DataFrameSchema(columns, name="ai4i2020_raw")


def validate_raw(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Valide le DataFrame brut en collectant toutes les erreurs (lazy=True).

    Raises:
        pandera.errors.SchemaErrors: au moins une règle du schéma est violée.
    """
    schema = build_raw_schema(cfg)
    try:
        return schema.validate(df, lazy=True)
    except SchemaErrors as exc:
        logger.error(
            "Validation du fichier brut échouée : %d erreur(s) détectée(s)",
            len(exc.failure_cases),
        )
        raise
