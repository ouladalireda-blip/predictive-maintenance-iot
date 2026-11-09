"""Feature engineering : variables physiques dérivées et préprocesseur scikit-learn.

Variables ajoutées par `PhysicalFeatures` (une par mode de panne du jeu AI4I 2020) :

- ``temp_diff_k`` = température process − température air, en kelvins (K).
  Mode associé : HDF (Heat Dissipation Failure). La dissipation thermique devient insuffisante
  quand l'écart process/air est faible et que la vitesse de rotation est basse.

- ``power_w`` = couple × vitesse × 2π / 60, en watts (W) (couple en N·m, vitesse en tr/min
  convertie en rad/s). Mode associé : PWF (Power Failure). La panne survient quand la puissance
  mécanique sort de sa plage de fonctionnement (trop faible ou trop élevée).

- ``strain_min_nm`` = usure outil × couple, en min·N·m. Mode associé : OSF (Overstrain Failure).
  La surcharge survient quand le produit usure × couple dépasse un seuil dépendant
  de la qualité produit (L, M, H).

Aucune de ces variables n'apprend quoi que ce soit des données : ce sont des formules physiques
fixes, sans risque de fuite entre train et test.
"""

from __future__ import annotations

import math
from typing import Any, Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from predictive_maintenance_iot.config import Config
from predictive_maintenance_iot.data.validation import COLUMN_MAP

TEMP_DIFF = "temp_diff_k"
POWER = "power_w"
STRAIN = "strain_min_nm"
NEW_FEATURES: tuple[str, str, str] = (TEMP_DIFF, POWER, STRAIN)

RPM_TO_RAD_PER_S = 2 * math.pi / 60


class PhysicalFeatures(BaseEstimator, TransformerMixin):
    """Ajoute temp_diff_k, power_w et strain_min_nm (voir la docstring du module).

    Transformation sans état : `fit` ne fait que vérifier la présence des colonnes et
    retenir les noms d'entrée (nécessaires à `get_feature_names_out`).
    """

    def __init__(
        self,
        air_temp_col: str,
        process_temp_col: str,
        speed_col: str,
        torque_col: str,
        tool_wear_col: str,
    ) -> None:
        self.air_temp_col = air_temp_col
        self.process_temp_col = process_temp_col
        self.speed_col = speed_col
        self.torque_col = torque_col
        self.tool_wear_col = tool_wear_col

    @classmethod
    def from_config(cls, cfg: Config) -> PhysicalFeatures:
        """Construit le transformeur à partir des colonnes numériques de la config."""
        missing = [col for col in COLUMN_MAP.values() if col not in cfg.columns.numeric]
        if missing:
            raise ValueError(f"colonnes de COLUMN_MAP absentes de cfg.columns.numeric : {missing}")
        return cls(
            air_temp_col=COLUMN_MAP["air_temperature_k"],
            process_temp_col=COLUMN_MAP["process_temperature_k"],
            speed_col=COLUMN_MAP["rotational_speed_rpm"],
            torque_col=COLUMN_MAP["torque_nm"],
            tool_wear_col=COLUMN_MAP["tool_wear_min"],
        )

    @property
    def _required_columns(self) -> list[str]:
        return [
            self.air_temp_col,
            self.process_temp_col,
            self.speed_col,
            self.torque_col,
            self.tool_wear_col,
        ]

    def _check_columns(self, X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"PhysicalFeatures attend un pandas.DataFrame, reçu {type(X).__name__}")
        missing = [col for col in self._required_columns if col not in X.columns]
        if missing:
            raise ValueError(f"PhysicalFeatures : colonne(s) manquante(s) dans X : {missing}")

    def fit(self, X: pd.DataFrame, y: Any = None) -> Self:
        self._check_columns(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Retourne une copie de X avec les 3 variables physiques ajoutées (X n'est pas modifié)."""
        self._check_columns(X)
        out = X.copy()
        out[TEMP_DIFF] = X[self.process_temp_col] - X[self.air_temp_col]
        out[POWER] = X[self.torque_col] * X[self.speed_col] * RPM_TO_RAD_PER_S
        out[STRAIN] = X[self.tool_wear_col] * X[self.torque_col]
        return out

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Colonnes d'entrée suivies des 3 nouvelles variables."""
        if input_features is None:
            if not hasattr(self, "feature_names_in_"):
                raise ValueError("PhysicalFeatures n'est pas encore ajusté (appeler fit d'abord)")
            input_features = self.feature_names_in_
        return np.asarray([*input_features, *NEW_FEATURES], dtype=object)


def build_preprocessor(cfg: Config) -> Pipeline:
    """Pipeline : variables physiques → encodage du type produit + passthrough numérique.

    Seules les variables explicatives sortent du pipeline : ids, colonnes de fuite et cible
    sont exclues (remainder="drop"). La sortie est un DataFrame pandas avec des noms de
    colonnes lisibles (utile pour SHAP).
    """
    product_type = cfg.columns.product_type
    encoder = ColumnTransformer(
        transformers=[
            (
                "product_type",
                OrdinalEncoder(categories=[product_type.allowed]),
                [product_type.column],
            ),
            ("numeric", "passthrough", [*cfg.columns.numeric, *NEW_FEATURES]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    pipeline = Pipeline(
        steps=[
            ("physical_features", PhysicalFeatures.from_config(cfg)),
            ("encode", encoder),
        ]
    )
    return pipeline.set_output(transform="pandas")
