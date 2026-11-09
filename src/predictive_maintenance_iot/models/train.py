"""Entraînement des modèles et suivi des expériences avec MLflow.

Les modèles sont entraînés sur train.csv et évalués sur val.csv uniquement.
Le jeu de test n'est jamais lu ici : il est réservé à l'évaluation finale (evaluate.py).
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from xgboost import Booster, XGBClassifier

from predictive_maintenance_iot.config import Config, load_config
from predictive_maintenance_iot.features.engineering import (
    PhysicalFeatures,
    RenameColumns,
    build_preprocessor,
)
from predictive_maintenance_iot.models.baseline import build_logistic_baseline

logger = logging.getLogger(__name__)


def _qualified_name(obj: Any) -> str:
    return f"{obj.__module__}.{obj.__qualname__}"


# MLflow sérialise les modèles sklearn avec skops, qui refuse par défaut les types inconnus.
# Liste explicite (et non « tout ce qui est trouvé ») : nos transformeurs + les classes XGBoost.
SKOPS_TRUSTED_TYPES: list[str] = [
    _qualified_name(obj) for obj in (PhysicalFeatures, RenameColumns, XGBClassifier, Booster)
]


def compute_scale_pos_weight(y: pd.Series | np.ndarray) -> float:
    """Ratio nb de lignes normales (0) / nb de pannes (1)."""
    labels = np.asarray(y)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0:
        raise ValueError("impossible de calculer scale_pos_weight : aucune panne (classe 1) dans y")
    return n_neg / n_pos


def build_xgb_pipeline(cfg: Config, y_train: pd.Series | np.ndarray) -> Pipeline:
    """Préprocesseur + XGBClassifier (hyperparamètres de la config, sans early stopping)."""
    params = cfg.model.xgboost
    scale_pos_weight = (
        compute_scale_pos_weight(y_train)
        if params.scale_pos_weight == "auto"
        else float(params.scale_pos_weight)
    )
    model = XGBClassifier(
        n_estimators=params.n_estimators,
        max_depth=params.max_depth,
        learning_rate=params.learning_rate,
        subsample=params.subsample,
        colsample_bytree=params.colsample_bytree,
        scale_pos_weight=scale_pos_weight,
        random_state=cfg.project.random_seed,
        eval_metric="aucpr",
        n_jobs=-1,
    )
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(cfg)),
            ("model", model),
        ]
    )


def _git_commit() -> str:
    """Hash du commit du code (exécuté dans le dossier du module, pas le dossier courant)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
            cwd=Path(__file__).resolve().parent,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _loggable_params(estimator: Any) -> dict[str, Any]:
    """Hyperparamètres scalaires non nuls de l'estimateur final (pour mlflow.log_params)."""
    return {
        key: value
        for key, value in estimator.get_params(deep=False).items()
        if isinstance(value, str | int | float | bool)
    }


def validation_metrics(y_true: pd.Series, proba: np.ndarray) -> dict[str, float]:
    return {
        "pr_auc": float(average_precision_score(y_true, proba)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "brier_score": float(brier_score_loss(y_true, proba)),
    }


def train_model(
    name: str,
    pipeline: Pipeline,
    cfg: Config,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> Pipeline:
    """Entraîne sur train, évalue sur val, journalise dans MLflow et sauvegarde en .joblib."""
    target = cfg.columns.target
    X_train, y_train = train_df.drop(columns=[target]), train_df[target]
    X_val, y_val = val_df.drop(columns=[target]), val_df[target]

    pipeline.fit(X_train, y_train)
    proba_val = pipeline.predict_proba(X_val)[:, 1]
    metrics = validation_metrics(y_val, proba_val)

    estimator = pipeline.steps[-1][1]
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    with mlflow.start_run(run_name=name):
        mlflow.log_params(_loggable_params(estimator))
        mlflow.log_params(
            {
                "random_seed": cfg.project.random_seed,
                "n_train": len(train_df),
                "n_val": len(val_df),
                "train_failure_rate": float(y_train.mean()),
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.set_tags({"model_type": type(estimator).__name__, "git_commit": _git_commit()})

        input_example = X_train.head(2)
        signature = infer_signature(input_example, pipeline.predict_proba(input_example))
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            signature=signature,
            input_example=input_example,
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )

        if isinstance(estimator, XGBClassifier):
            importance = estimator.get_booster().get_score(importance_type="gain")
            ranked = dict(sorted(importance.items(), key=lambda kv: kv[1], reverse=True))
            mlflow.log_dict(ranked, "feature_importance_gain.json")

    out_dir = cfg.paths.artifacts_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out_dir / f"{name}.joblib")

    logger.info(
        "%s : PR-AUC=%.3f, ROC-AUC=%.3f, Brier=%.4f (validation)",
        name,
        metrics["pr_auc"],
        metrics["roc_auc"],
        metrics["brier_score"],
    )
    return pipeline


def main() -> None:
    cfg = load_config()
    target = cfg.columns.target
    processed = cfg.paths.processed_dir
    train_df = pd.read_csv(processed / "train.csv")
    val_df = pd.read_csv(processed / "val.csv")

    candidates = {
        "xgboost": build_xgb_pipeline(cfg, train_df[target]),
        "logistic": build_logistic_baseline(cfg),
    }
    X_val, y_val = val_df.drop(columns=[target]), val_df[target]
    rows = []
    for name, pipeline in candidates.items():
        fitted = train_model(name, pipeline, cfg, train_df, val_df)
        metrics = validation_metrics(y_val, fitted.predict_proba(X_val)[:, 1])
        rows.append({"modèle": name, **metrics})

    report = pd.DataFrame(rows).set_index("modèle")
    report.columns = ["PR-AUC", "ROC-AUC", "Brier"]
    print(f"\nModèles sur la VALIDATION ({len(val_df)} lignes) :\n")
    print(report.round(4).to_string())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    main()
