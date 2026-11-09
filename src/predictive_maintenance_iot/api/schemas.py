"""Schémas de requêtes et de réponses de l'API.

PredictRequest ne contrôle que les types : bornes physiques et types produit autorisés sont
vérifiés ensuite par validate_payload (règles de la config), pour une seule source de vérité.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

DecisionLiteral = Literal["CRITIQUE", "MAINTENANCE_PREVENTIVE", "OPTIMAL"]
TriggeredByLiteral = Literal["model", "preventive", "both", "none"]

REQUEST_EXAMPLE: dict[str, Any] = {
    "product_type": "M",
    "air_temperature_k": 298.9,
    "process_temperature_k": 309.1,
    "rotational_speed_rpm": 1408.0,
    "torque_nm": 46.3,
    "tool_wear_min": 120.0,
}


class PredictRequest(BaseModel):
    """Une mesure capteur (noms snake_case)."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [REQUEST_EXAMPLE]})

    product_type: str = Field(description="Qualité produit (L, M ou H)")
    air_temperature_k: float = Field(description="Température de l'air (K)")
    process_temperature_k: float = Field(description="Température du procédé (K)")
    rotational_speed_rpm: float = Field(description="Vitesse de rotation (tr/min)")
    torque_nm: float = Field(description="Couple (N·m)")
    tool_wear_min: float = Field(description="Usure de l'outil (min)")


class TopFactorSchema(BaseModel):
    feature: str
    value: float | str = Field(
        description="Valeur d'origine : libellé L/M/H pour le type, variables dérivées arrondies"
    )
    contribution: float = Field(description="Contribution SHAP signée (log-odds)")
    direction: Literal["vers_panne", "vers_normal"]
    unit: str | None = Field(description="K, rpm, Nm, min, W, min·Nm ; null si sans unité")


class ModelVersionSchema(BaseModel):
    model_sha256: str
    thresholds_sha256: str
    git_commit: str = Field(description="Commit au moment du gel des seuils")


class PredictResponse(BaseModel):
    request_id: str
    timestamp: datetime
    decision: DecisionLiteral
    failure_probability: float
    threshold: float
    triggered_by: TriggeredByLiteral
    top_factors: list[TopFactorSchema]
    recommended_action: str | None
    model_version: ModelVersionSchema


class BatchRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid", json_schema_extra={"examples": [{"items": [REQUEST_EXAMPLE]}]}
    )

    items: list[PredictRequest] = Field(min_length=1)


class BatchResponse(BaseModel):
    count: int
    predictions: list[PredictResponse]


class HealthResponse(BaseModel):
    status: Literal["ok", "unavailable"]
    model_loaded: bool
    model_version: ModelVersionSchema | None = None
    detail: str | None = None


class ModelInfoResponse(BaseModel):
    primary_approach: str
    secondary_approach: str | None
    threshold: float
    thresholds: dict[str, Any] = Field(description="Seuils figés par approche (thresholds.json)")
    preventive_tool_wear_min: float
    cost_missed_failure_eur: float
    cost_false_alarm_eur: float
    min_precision: float | None
    test_results: list[dict[str, Any]] | None = Field(
        description="Résultats de l'examen final au ratio de la config (None s'il n'a pas eu lieu)"
    )
    model_version: ModelVersionSchema
