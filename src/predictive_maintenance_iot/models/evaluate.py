"""Évaluation sur la VALIDATION : coût métier, choix du seuil et comparaison aux baselines.

Le seuil d'alerte n'est pas fixé à 0,5 : il minimise le coût total
    coût = nb pannes ratées × cost_missed_failure_eur + nb fausses alertes × cost_false_alarm_eur
sous la contrainte de précision minimale de la config (decision.min_precision).

Ce module ne lit que train.csv et val.csv : le jeu de test est réservé à l'évaluation finale,
une seule fois, avec les seuils figés ici (artifacts/thresholds.json).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

import joblib
import mlflow
import numpy as np
import pandas as pd

from predictive_maintenance_iot.config import Config, load_config
from predictive_maintenance_iot.models.baseline import ExpertRulesBaseline, PreventiveBaseline
from predictive_maintenance_iot.utils import git_commit

logger = logging.getLogger(__name__)

THRESHOLDS_FILENAME = "thresholds.json"
MLFLOW_RUN_NAME = "evaluation-validation"

B0 = "B0 aucune alerte"
B1 = "B1 preventive"
B2 = "B2 regles expertes"
B3 = "B3 logistique"
XGB = "XGBoost"
XGB_FREE = "XGBoost (sans contrainte)"
HYBRID = "Hybride XGBoost + preventif"
HYBRID_FREE = "Hybride XGBoost + preventif (sans contrainte)"
# Approche à seuil optimisé → nom du pipeline sauvegardé par train.py ({artifacts_dir}/{nom}.joblib)
SCORED_MODELS: dict[str, str] = {B3: "logistic", XGB: "xgboost"}

ArrayLike = pd.Series | np.ndarray


@dataclass(frozen=True)
class _Approach:
    """Sorties d'une approche sur la validation.

    - kind="binary" : `values` contient des alertes 0/1 (règles), pas de seuil à choisir.
    - kind="proba" : `values` contient des scores ; le seuil est optimisé sur le coût.
      `forced_alert` : lignes alertées quel que soit le seuil (approche hybride).
      `constrained` : applique decision.min_precision lors du choix du seuil.
    """

    kind: Literal["binary", "proba"]
    values: np.ndarray
    forced_alert: np.ndarray | None = None
    constrained: bool = True


# --- Coût et seuil ----------------------------------------------------------------------


def cost_at_threshold(
    y_true: ArrayLike,
    proba: ArrayLike,
    threshold: float,
    cost_fn: float,
    cost_fp: float,
    forced_alert: ArrayLike | None = None,
) -> dict[str, Any]:
    """Matrice de confusion et coût total en alertant quand proba >= threshold.

    `forced_alert` (optionnel) : masque booléen des lignes alertées quel que soit le seuil
    (ex. remplacement préventif de l'outil dans l'approche hybride).

    Convention : precision = 0 quand aucune alerte n'est émise (un seuil qui n'alerte jamais
    ne peut donc pas satisfaire une contrainte de précision minimale strictement positive).
    """
    truth = np.asarray(y_true).astype(int)
    alert = np.asarray(proba) >= threshold
    if forced_alert is not None:
        alert = alert | np.asarray(forced_alert, dtype=bool)
    tp = int((alert & (truth == 1)).sum())
    fp = int((alert & (truth == 0)).sum())
    fn = int((~alert & (truth == 1)).sum())
    tn = int((~alert & (truth == 0)).sum())
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "total_cost_eur": fn * cost_fn + fp * cost_fp,
    }


def find_best_threshold(
    y_true: ArrayLike,
    proba: ArrayLike,
    cost_fn: float,
    cost_fp: float,
    min_precision: float,
    forced_alert: ArrayLike | None = None,
) -> dict[str, Any]:
    """Seuil de coût minimal parmi les valeurs uniques de proba (+ 0 et 1).

    Seuls les seuils de précision >= min_precision sont retenus ; à coût égal, le plus élevé
    (moins d'alertes) l'emporte. Si aucun seuil n'est admissible, renvoie le seuil de coût
    minimal sans contrainte avec constraint_satisfied=False.
    Avec `forced_alert`, coût et précision sont ceux de la combinaison (score OU forcé).
    """
    scores = np.asarray(proba, dtype=float)
    candidates = np.unique(np.concatenate([scores, [0.0, 1.0]]))
    results = [
        {
            "threshold": float(t),
            **cost_at_threshold(y_true, scores, t, cost_fn, cost_fp, forced_alert),
        }
        for t in candidates
    ]
    admissible = [r for r in results if r["precision"] >= min_precision]
    constraint_satisfied = bool(admissible)
    if not constraint_satisfied:
        logger.warning(
            "Aucun seuil n'atteint la précision minimale %.2f : coût minimal sans contrainte",
            min_precision,
        )
    pool = admissible or results
    best = min(pool, key=lambda r: (r["total_cost_eur"], -r["threshold"]))
    return {**best, "constraint_satisfied": constraint_satisfied}


# --- Comparaison des approches ----------------------------------------------------------


def _load_validation_outputs(cfg: Config) -> tuple[pd.Series, dict[str, _Approach]]:
    """Cible de validation et sorties de chaque approche (voir `_Approach`)."""
    target = cfg.columns.target
    processed = cfg.paths.processed_dir
    train = pd.read_csv(processed / "train.csv")
    val = pd.read_csv(processed / "val.csv")
    X_train, y_train = train.drop(columns=[target]), train[target]
    X_val, y_val = val.drop(columns=[target]), val[target]

    # Alerte préventive (usure >= preventive_tool_wear_min) : B1 seul, ou forcée dans l'hybride.
    preventive = PreventiveBaseline.from_config(cfg).fit(X_train, y_train).predict(X_val)
    expert = ExpertRulesBaseline.from_config(cfg).fit(X_train, y_train).predict(X_val)
    approaches: dict[str, _Approach] = {
        B0: _Approach("binary", np.zeros(len(val), dtype=int)),
        B1: _Approach("binary", preventive),
        B2: _Approach("binary", expert),
    }

    scores: dict[str, np.ndarray] = {}
    for approach, filename in SCORED_MODELS.items():
        path = cfg.paths.artifacts_dir / f"{filename}.joblib"
        if not path.is_file():
            raise FileNotFoundError(f"{path} introuvable : lancer d'abord models/train.py")
        scores[approach] = joblib.load(path).predict_proba(X_val)[:, 1]
        approaches[approach] = _Approach("proba", scores[approach])

    forced = preventive.astype(bool)
    approaches[XGB_FREE] = _Approach("proba", scores[XGB], constrained=False)
    approaches[HYBRID] = _Approach("proba", scores[XGB], forced_alert=forced)
    approaches[HYBRID_FREE] = _Approach(
        "proba", scores[XGB], forced_alert=forced, constrained=False
    )
    return y_val, approaches


def _evaluate_approaches(
    y_val: pd.Series,
    approaches: dict[str, _Approach],
    cost_fn: float,
    cost_fp: float,
    min_precision: float,
) -> pd.DataFrame:
    """Une ligne par approche. precision_ok compare toujours à min_precision (garde-fou métier),
    y compris pour les règles et les variantes optimisées sans contrainte."""
    rows = []
    for approach, spec in approaches.items():
        if spec.kind == "proba":
            result = find_best_threshold(
                y_val,
                spec.values,
                cost_fn,
                cost_fp,
                min_precision if spec.constrained else 0.0,
                spec.forced_alert,
            )
            if not spec.constrained:
                result["constraint_satisfied"] = np.nan  # contrainte non appliquée
        else:
            # Sorties 0/1 : alerte quand la sortie vaut 1 (pas de seuil à choisir).
            result = {
                "threshold": np.nan,
                **cost_at_threshold(y_val, spec.values, 1, cost_fn, cost_fp),
                "constraint_satisfied": np.nan,
            }
        rows.append(
            {
                "approach": approach,
                "threshold": result["threshold"],
                "detected": result["tp"],
                "false_alarms": result["fp"],
                "missed": result["fn"],
                "precision": result["precision"],
                "precision_ok": bool(result["precision"] >= min_precision),
                "recall": result["recall"],
                "total_cost_eur": result["total_cost_eur"],
                "constraint_satisfied": result["constraint_satisfied"],
            }
        )
    b0_cost = float(next(r["total_cost_eur"] for r in rows if r["approach"] == B0))
    df = pd.DataFrame(rows).set_index("approach")
    df["gain_vs_b0_eur"] = b0_cost - df["total_cost_eur"]
    df["gain_vs_b0_pct"] = 100 * df["gain_vs_b0_eur"] / b0_cost if b0_cost else 0.0
    return df


def _cell(df: pd.DataFrame, approach: str, column: str) -> Any:
    """Valeur d'une cellule (ligne = approche), typée Any (évite les unions de pandas-stubs)."""
    return df[column].to_dict()[approach]


def compare_on_validation(cfg: Config) -> pd.DataFrame:
    """Une ligne par approche (B0 à B3, XGBoost, hybride, variantes sans contrainte)."""
    y_val, outputs = _load_validation_outputs(cfg)
    d = cfg.decision
    return _evaluate_approaches(
        y_val, outputs, d.cost_missed_failure_eur, d.cost_false_alarm_eur, d.min_precision
    )


def sensitivity_analysis(cfg: Config, ratios: Iterable[float] = (5, 10, 20, 50)) -> pd.DataFrame:
    """Coût de chaque approche quand cost_fn = ratio × cost_fp (cost_fp fixé par la config)."""
    y_val, outputs = _load_validation_outputs(cfg)
    cost_fp = cfg.decision.cost_false_alarm_eur
    frames = []
    for ratio in ratios:
        df = _evaluate_approaches(
            y_val, outputs, ratio * cost_fp, cost_fp, cfg.decision.min_precision
        ).reset_index()
        df.insert(0, "ratio_fn_fp", ratio)
        df.insert(1, "cost_fn_eur", ratio * cost_fp)
        df["rank"] = df["total_cost_eur"].rank(method="min").astype(int)
        frames.append(df)
    columns = [
        "ratio_fn_fp",
        "cost_fn_eur",
        "approach",
        "threshold",
        "precision",
        "precision_ok",
        "total_cost_eur",
        "gain_vs_b0_eur",
        "gain_vs_b0_pct",
        "rank",
    ]
    return pd.concat(frames, ignore_index=True)[columns]


# --- Point d'entrée ---------------------------------------------------------------------


def _thresholds_payload(comparison: pd.DataFrame, cfg: Config) -> dict[str, Any]:
    d = cfg.decision
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "cost_missed_failure_eur": d.cost_missed_failure_eur,
        "cost_false_alarm_eur": d.cost_false_alarm_eur,
        "min_precision": d.min_precision,
        "thresholds": {
            filename: {
                "threshold": float(_cell(comparison, approach, "threshold")),
                "constraint_satisfied": bool(_cell(comparison, approach, "constraint_satisfied")),
                "validation_cost_eur": float(_cell(comparison, approach, "total_cost_eur")),
            }
            for approach, filename in SCORED_MODELS.items()
        },
    }


def _rounded(df: pd.DataFrame) -> pd.DataFrame:
    """Montants arrondis à l'euro, ratios à 3 décimales (affichage uniquement)."""
    out = df.copy()
    for column in out.columns:
        if column.endswith("_eur"):
            out[column] = out[column].round(0).astype("Int64")
        elif out[column].dtype.kind == "f":
            out[column] = out[column].round(3)
    return out


def main() -> None:
    cfg = load_config()
    comparison = compare_on_validation(cfg)
    sensitivity = sensitivity_analysis(cfg)

    print("\nComparaison sur la VALIDATION (coûts de la config) :\n")
    print(_rounded(comparison).to_string())
    print("\nAnalyse de sensibilité (cost_fp fixe, cost_fn = ratio × cost_fp) :\n")
    print(_rounded(sensitivity).to_string(index=False))

    reports_dir = cfg.paths.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = reports_dir / "validation_comparison.csv"
    sensitivity_path = reports_dir / "sensitivity.csv"
    comparison.to_csv(comparison_path)
    sensitivity.to_csv(sensitivity_path, index=False)

    cfg.paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
    thresholds_path = cfg.paths.artifacts_dir / THRESHOLDS_FILENAME
    thresholds_path.write_text(
        json.dumps(_thresholds_payload(comparison, cfg), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    with mlflow.start_run(run_name=MLFLOW_RUN_NAME):
        mlflow.set_tag("git_commit", git_commit())
        mlflow.log_metrics(
            {
                "xgb_val_cost_eur": float(_cell(comparison, XGB, "total_cost_eur")),
                "xgb_threshold": float(_cell(comparison, XGB, "threshold")),
                "b2_val_cost_eur": float(_cell(comparison, B2, "total_cost_eur")),
            }
        )
        for path in (comparison_path, sensitivity_path, thresholds_path):
            mlflow.log_artifact(str(path))
    logger.info("Rapports écrits dans %s, seuils dans %s", reports_dir, thresholds_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    main()
