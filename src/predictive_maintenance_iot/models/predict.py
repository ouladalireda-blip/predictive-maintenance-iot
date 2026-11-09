"""Inférence : décision de maintenance à partir d'une mesure, avec le modèle et les seuils figés.

Règle de décision (approche hybride, celle retenue au gel des seuils) :
- CRITIQUE               si proba >= seuil hybride figé (même si l'usure est aussi élevée) ;
- MAINTENANCE_PREVENTIVE sinon, si usure outil >= seuil préventif figé ;
- OPTIMAL                sinon.

Tout vient de artifacts/ : modèle (vérifié par son empreinte) et thresholds.json (frozen_for_test).
Rien n'est lu dans la config pour la décision : seuls les textes d'action en proviennent.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from predictive_maintenance_iot.config import ActionsConfig, Config
from predictive_maintenance_iot.data.validation import COLUMN_MAP, SensorPayload
from predictive_maintenance_iot.features.engineering import (
    NEW_FEATURES,
    POWER,
    PRODUCT_TYPE_FEATURE,
    STRAIN,
    TEMP_DIFF,
)
from predictive_maintenance_iot.models.evaluate import THRESHOLDS_FILENAME
from predictive_maintenance_iot.models.fingerprint import (
    FINGERPRINT_FILENAME,
    MODEL_FILENAME,
    sha256_file,
    verify_fingerprint,
)

logger = logging.getLogger(__name__)

N_TOP_FACTORS = 3
# Valeur des colonnes vues à l'entraînement mais absentes d'une mesure (ids, colonnes de fuite).
# Le préprocesseur les supprime avant le modèle : elles n'ont aucun effet sur la probabilité.
_UNUSED_COLUMN_PLACEHOLDER = 0


class Decision(StrEnum):
    CRITIQUE = "CRITIQUE"
    MAINTENANCE_PREVENTIVE = "MAINTENANCE_PREVENTIVE"
    OPTIMAL = "OPTIMAL"


class TriggeredBy(StrEnum):
    MODEL = "model"
    PREVENTIVE = "preventive"
    BOTH = "both"
    NONE = "none"


class Direction(StrEnum):
    TOWARDS_FAILURE = "vers_panne"  # contribution SHAP > 0
    TOWARDS_NORMAL = "vers_normal"  # contribution SHAP < 0


# Unité de chaque variable de sortie du préprocesseur (None : sans unité).
FEATURE_UNITS: dict[str, str | None] = {
    PRODUCT_TYPE_FEATURE: None,
    "air_temperature_k": "K",
    "process_temperature_k": "K",
    "rotational_speed_rpm": "rpm",
    "torque_nm": "Nm",
    "tool_wear_min": "min",
    TEMP_DIFF: "K",
    POWER: "W",
    STRAIN: "min·Nm",
}
DERIVED_DECIMALS = 2


@dataclass(frozen=True)
class TopFactor:
    feature: str
    value: float | str  # valeur d'origine : "L"/"M"/"H" pour le type, dérivées arrondies
    contribution: float  # contribution SHAP (log-odds), signée
    direction: Direction
    unit: str | None


@dataclass(frozen=True)
class ModelVersion:
    model_sha256: str
    thresholds_sha256: str
    git_commit: str


@dataclass(frozen=True)
class Prediction:
    decision: Decision
    failure_probability: float
    threshold: float
    triggered_by: TriggeredBy
    top_factors: list[TopFactor]
    recommended_action: str | None
    derived: dict[str, float]  # variables physiques calculées par le pipeline, arrondies


def decide(model_alert: bool, preventive_alert: bool) -> tuple[Decision, TriggeredBy]:
    """Décision et origine de l'alerte (le modèle l'emporte sur le préventif)."""
    if model_alert and preventive_alert:
        triggered = TriggeredBy.BOTH
    elif model_alert:
        triggered = TriggeredBy.MODEL
    elif preventive_alert:
        triggered = TriggeredBy.PREVENTIVE
    else:
        triggered = TriggeredBy.NONE
    if model_alert:
        return Decision.CRITIQUE, triggered
    if preventive_alert:
        return Decision.MAINTENANCE_PREVENTIVE, triggered
    return Decision.OPTIMAL, triggered


class Predictor:
    """Modèle figé + seuils figés + explicateur SHAP, chargés une seule fois."""

    def __init__(
        self,
        pipeline: Pipeline,
        threshold: float,
        preventive_wear_min: float,
        actions: ActionsConfig,
        version: ModelVersion,
        product_type_column: str,
        frozen: dict[str, Any],
    ) -> None:
        self.pipeline = pipeline
        self.threshold = threshold
        self.preventive_wear_min = preventive_wear_min
        self.actions = actions
        self.version = version
        self.product_type_column = product_type_column
        self.frozen = frozen
        self._explainer = shap.TreeExplainer(pipeline.named_steps["model"])

    @classmethod
    def load(cls, cfg: Config) -> Predictor:
        """Vérifie l'empreinte, charge le modèle et les seuils figés de l'approche principale."""
        verify_fingerprint(cfg)
        artifacts = cfg.paths.artifacts_dir
        thresholds_path = artifacts / THRESHOLDS_FILENAME
        if not thresholds_path.is_file():
            raise FileNotFoundError(f"{thresholds_path} introuvable : seuils non figés")
        frozen: dict[str, Any] = json.loads(thresholds_path.read_text(encoding="utf-8"))
        if frozen.get("frozen_for_test") is not True:
            raise ValueError(f"{thresholds_path} n'est pas marqué frozen_for_test=true")
        primary = frozen.get("primary_approach")
        if primary != "hybrid":
            raise ValueError(
                f"approche principale {primary!r} non prise en charge : la règle de décision "
                "de l'API est celle de l'hybride"
            )

        fingerprint = json.loads((artifacts / FINGERPRINT_FILENAME).read_text(encoding="utf-8"))
        version = ModelVersion(
            model_sha256=str(fingerprint["sha256"]),
            thresholds_sha256=sha256_file(thresholds_path),
            git_commit=str(frozen.get("git_commit", "unknown")),
        )
        predictor = cls(
            pipeline=joblib.load(artifacts / MODEL_FILENAME),
            threshold=float(frozen["thresholds"][primary]["threshold"]),
            preventive_wear_min=float(frozen["preventive_tool_wear_min"]),
            actions=cfg.api.actions,
            version=version,
            product_type_column=cfg.columns.product_type.column,
            frozen=frozen,
        )
        logger.info(
            "Modèle %s chargé (seuil %.4f, usure préventive %.0f min)",
            version.model_sha256[:12],
            predictor.threshold,
            predictor.preventive_wear_min,
        )
        return predictor

    def to_model_frame(self, payloads: Sequence[SensorPayload]) -> pd.DataFrame:
        """Mesures (noms API) → DataFrame aux colonnes CSV attendues par le pipeline.

        Le pipeline a été entraîné sur toutes les colonnes du CSV hors cible : les colonnes
        absentes d'une mesure (ids, colonnes de fuite) sont remplies d'une valeur neutre,
        supprimée par le préprocesseur avant le modèle.
        """
        rows = []
        for payload in payloads:
            row: dict[str, Any] = dict.fromkeys(
                self.pipeline.feature_names_in_, _UNUSED_COLUMN_PLACEHOLDER
            )
            row.update({csv: getattr(payload, api) for api, csv in COLUMN_MAP.items()})
            row[self.product_type_column] = payload.product_type
            rows.append(row)
        return pd.DataFrame(rows, columns=list(self.pipeline.feature_names_in_))

    @staticmethod
    def _original_value(
        feature: str, payload: SensorPayload, features: pd.DataFrame, i: int
    ) -> float | str:
        """Valeur lisible : libellé du type produit, mesure brute, ou dérivée arrondie."""
        if feature == PRODUCT_TYPE_FEATURE:
            return payload.product_type
        if feature in COLUMN_MAP:
            return float(getattr(payload, feature))
        return round(float(features[feature].iloc[i]), DERIVED_DECIMALS)

    def _top_factors(
        self, payload: SensorPayload, features: pd.DataFrame, shap_row: np.ndarray, i: int
    ) -> list[TopFactor]:
        """Les N_TOP_FACTORS plus fortes contributions en valeur absolue, triées décroissantes.

        Les contributions nulles sont ignorées (elles n'expliquent rien)."""
        order = np.argsort(-np.abs(shap_row), kind="stable")
        factors = []
        for j in order[:N_TOP_FACTORS]:
            contribution = float(shap_row[j])
            if contribution == 0:
                continue
            feature = str(features.columns[j])
            factors.append(
                TopFactor(
                    feature=feature,
                    value=self._original_value(feature, payload, features, i),
                    contribution=contribution,
                    direction=(
                        Direction.TOWARDS_FAILURE if contribution > 0 else Direction.TOWARDS_NORMAL
                    ),
                    unit=FEATURE_UNITS.get(feature),
                )
            )
        return factors

    def _action(self, decision: Decision, factors: list[TopFactor]) -> str | None:
        """CRITIQUE : action du facteur « vers_panne » le plus fort (texte par défaut sinon)."""
        if decision is Decision.CRITIQUE:
            driver = next((f for f in factors if f.direction is Direction.TOWARDS_FAILURE), None)
            if driver is not None and driver.feature in self.actions.by_feature:
                return self.actions.by_feature[driver.feature]
            return self.actions.critical_default
        if decision is Decision.MAINTENANCE_PREVENTIVE:
            return self.actions.preventive
        return None

    def predict_batch(self, payloads: Sequence[SensorPayload]) -> list[Prediction]:
        if not payloads:
            return []
        X = self.to_model_frame(payloads)
        proba = self.pipeline.predict_proba(X)[:, 1]
        features = self.pipeline.named_steps["preprocess"].transform(X)
        contributions = np.asarray(self._explainer.shap_values(features))

        predictions = []
        for i, payload in enumerate(payloads):
            p = float(proba[i])
            decision, triggered = decide(
                model_alert=p >= self.threshold,
                preventive_alert=payload.tool_wear_min >= self.preventive_wear_min,
            )
            factors = self._top_factors(payload, features, contributions[i], i)
            predictions.append(
                Prediction(
                    decision=decision,
                    failure_probability=p,
                    threshold=self.threshold,
                    triggered_by=triggered,
                    top_factors=factors,
                    recommended_action=self._action(decision, factors),
                    derived={
                        name: round(float(features[name].iloc[i]), DERIVED_DECIMALS)
                        for name in NEW_FEATURES
                    },
                )
            )
        return predictions

    def predict(self, payload: SensorPayload) -> Prediction:
        return self.predict_batch([payload])[0]
