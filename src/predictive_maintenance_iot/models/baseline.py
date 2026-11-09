"""Baselines : les références que le modèle ML doit battre pour justifier sa complexité.

- B1 `PreventiveBaseline` : maintenance préventive, alerte dès que l'usure outil atteint un seuil.
- B2 `ExpertRulesBaseline` : règles physiques, une par mode de panne (HDF, PWF, OSF, TWF).
- B3 `build_logistic_baseline` : régression logistique sur les variables du préprocesseur.

Les baselines B1 et B2 sont des règles déterministes : `fit` n'apprend rien et `predict_proba`
renvoie uniquement 0 ou 1 (ce n'est PAS une probabilité calibrée). Les métriques fondées sur un
classement des scores (ROC-AUC, PR-AUC) n'ont donc pas de sens pour elles.

Ce module n'utilise que train.csv et val.csv : le jeu de test est réservé à l'évaluation finale.
"""

from __future__ import annotations

import logging
from typing import Any, Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from predictive_maintenance_iot.config import Config, ExpertRulesConfig, load_config
from predictive_maintenance_iot.data.validation import COLUMN_MAP
from predictive_maintenance_iot.features.engineering import (
    POWER,
    STRAIN,
    TEMP_DIFF,
    PhysicalFeatures,
    build_preprocessor,
)

logger = logging.getLogger(__name__)

RULES: tuple[str, ...] = ("HDF", "PWF", "OSF", "TWF")
CLASSES = np.array([0, 1])


class _RuleBaseline(BaseEstimator, ClassifierMixin):
    """Base commune des baselines à règles : pas d'apprentissage, sortie binaire."""

    def _required_columns(self) -> list[str]:
        raise NotImplementedError

    def _alerts(self, X: pd.DataFrame) -> pd.Series:
        """Série booléenne : True si la ligne déclenche une alerte."""
        raise NotImplementedError

    def _check_columns(self, X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"{type(self).__name__} attend un pandas.DataFrame, reçu {type(X).__name__}"
            )
        missing = [col for col in self._required_columns() if col not in X.columns]
        if missing:
            raise ValueError(f"{type(self).__name__} : colonne(s) manquante(s) dans X : {missing}")

    def fit(self, X: pd.DataFrame, y: Any = None) -> Self:
        """N'apprend rien : vérifie les colonnes et fixe les classes (0 = normal, 1 = panne)."""
        self._check_columns(X)
        self.classes_ = CLASSES
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_columns(X)
        return self._alerts(X).to_numpy(dtype=int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Tableau (n, 2) : [1 - alerte, alerte], avec alerte ∈ {0, 1} (pas une probabilité)."""
        alert = self.predict(X).astype(float)
        return np.column_stack([1.0 - alert, alert])


class PreventiveBaseline(_RuleBaseline):
    """B1 : alerte si usure outil >= seuil de remplacement systématique."""

    def __init__(self, tool_wear_col: str, threshold_min: float) -> None:
        self.tool_wear_col = tool_wear_col
        self.threshold_min = threshold_min

    @classmethod
    def from_config(cls, cfg: Config) -> PreventiveBaseline:
        return cls(
            tool_wear_col=COLUMN_MAP["tool_wear_min"],
            threshold_min=cfg.baselines.preventive_tool_wear_min,
        )

    def _required_columns(self) -> list[str]:
        return [self.tool_wear_col]

    def _alerts(self, X: pd.DataFrame) -> pd.Series:
        return X[self.tool_wear_col] >= self.threshold_min


class ExpertRulesBaseline(_RuleBaseline):
    """B2 : alerte si au moins une règle physique est déclenchée.

    - HDF : temp_diff_k < hdf_temp_diff_max_k ET vitesse < hdf_speed_max_rpm
    - PWF : power_w hors [pwf_power_min_w, pwf_power_max_w]
    - OSF : strain_min_nm > seuil du type produit (osf_strain_max_min_nm[type])
    - TWF : usure outil >= twf_tool_wear_min
    """

    def __init__(
        self,
        rules: ExpertRulesConfig,
        features: PhysicalFeatures,
        product_type_col: str,
    ) -> None:
        self.rules = rules
        self.features = features
        self.product_type_col = product_type_col

    @classmethod
    def from_config(cls, cfg: Config) -> ExpertRulesBaseline:
        return cls(
            rules=cfg.baselines.expert_rules,
            features=PhysicalFeatures.from_config(cfg),
            product_type_col=cfg.columns.product_type.column,
        )

    def _required_columns(self) -> list[str]:
        f = self.features
        return [
            f.air_temp_col,
            f.process_temp_col,
            f.speed_col,
            f.torque_col,
            f.tool_wear_col,
            self.product_type_col,
        ]

    def rule_flags(self, X: pd.DataFrame) -> pd.DataFrame:
        """DataFrame booléen (une colonne par règle de RULES), même index que X."""
        self._check_columns(X)
        r = self.rules
        enriched = self.features.transform(X)

        osf_threshold = X[self.product_type_col].map(r.osf_strain_max_min_nm)
        unknown = sorted(X.loc[osf_threshold.isna(), self.product_type_col].unique())
        if unknown:
            raise ValueError(f"aucun seuil OSF pour le(s) type(s) produit : {unknown}")

        return pd.DataFrame(
            {
                "HDF": (enriched[TEMP_DIFF] < r.hdf_temp_diff_max_k)
                & (X[self.features.speed_col] < r.hdf_speed_max_rpm),
                "PWF": (enriched[POWER] < r.pwf_power_min_w)
                | (enriched[POWER] > r.pwf_power_max_w),
                "OSF": enriched[STRAIN] > osf_threshold,
                "TWF": X[self.features.tool_wear_col] >= r.twf_tool_wear_min,
            },
            index=X.index,
        )[list(RULES)]

    def _alerts(self, X: pd.DataFrame) -> pd.Series:
        return self.rule_flags(X).any(axis=1)

    def explain(self, X: pd.DataFrame) -> list[list[str]]:
        """Pour chaque ligne, la liste des règles déclenchées (vide si aucune)."""
        flags = self.rule_flags(X)
        return [[rule for rule in RULES if row[rule]] for _, row in flags.iterrows()]


def build_logistic_baseline(cfg: Config) -> Pipeline:
    """B3 : préprocesseur + standardisation + régression logistique pondérée."""
    params = cfg.baselines.logistic
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(cfg)),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=params.C,
                    max_iter=params.max_iter,
                    class_weight="balanced",
                    random_state=cfg.project.random_seed,
                ),
            ),
        ]
    )


def _summary(name: str, y_true: pd.Series, y_pred: np.ndarray) -> dict[str, Any]:
    truth = y_true.to_numpy(dtype=int)
    return {
        "baseline": name,
        "détectées": int(((y_pred == 1) & (truth == 1)).sum()),
        "pannes": int(truth.sum()),
        "fausses alertes": int(((y_pred == 1) & (truth == 0)).sum()),
        "ratées": int(((y_pred == 0) & (truth == 1)).sum()),
    }


def main() -> None:
    cfg = load_config()
    target = cfg.columns.target
    processed = cfg.paths.processed_dir
    train = pd.read_csv(processed / "train.csv")
    val = pd.read_csv(processed / "val.csv")

    X_train, y_train = train.drop(columns=[target]), train[target]
    X_val, y_val = val.drop(columns=[target]), val[target]

    baselines: dict[str, Any] = {
        "B1 préventive": PreventiveBaseline.from_config(cfg),
        "B2 règles expertes": ExpertRulesBaseline.from_config(cfg),
        "B3 logistique (seuil 0,5)": build_logistic_baseline(cfg),
    }
    rows = []
    for name, model in baselines.items():
        model.fit(X_train, y_train)
        rows.append(_summary(name, y_val, model.predict(X_val)))

    report = pd.DataFrame(rows).set_index("baseline")
    report["détectées"] = report["détectées"].astype(str) + " / " + report["pannes"].astype(str)
    print(f"\nBaselines sur la VALIDATION ({len(val)} lignes) :\n")
    print(report.drop(columns=["pannes"]).to_string())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    main()
