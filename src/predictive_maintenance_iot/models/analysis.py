"""Analyses complémentaires sur TRAIN et VAL : stabilité, incertitude, erreurs, ablation, SHAP.

Règles :
- le jeu de test n'est jamais lu ;
- artifacts/thresholds.json est lu, jamais modifié : les seuils figés (dont l'usure du
  préventif de l'hybride) sont réutilisés tels quels ;
- l'empreinte du modèle (artifacts/model_fingerprint.json) est vérifiée au démarrage ;
- aucun hyperparamètre ni seuil n'est modifié ici.

Sorties : {reports_dir}/analysis/*.csv et *.png, et un run MLflow "analysis-validation".
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
import shap
from matplotlib.figure import Figure
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from predictive_maintenance_iot.config import Config, load_config
from predictive_maintenance_iot.data.validation import COLUMN_MAP
from predictive_maintenance_iot.features.engineering import build_preprocessor
from predictive_maintenance_iot.models.baseline import (
    ExpertRulesBaseline,
    PreventiveBaseline,
    build_logistic_baseline,
)
from predictive_maintenance_iot.models.evaluate import B2, HYBRID, THRESHOLDS_FILENAME, XGB
from predictive_maintenance_iot.models.fingerprint import verify_fingerprint
from predictive_maintenance_iot.models.train import build_xgb_pipeline
from predictive_maintenance_iot.utils import git_commit

logger = logging.getLogger(__name__)

MLFLOW_RUN_NAME = "analysis-validation"
ANALYSIS_SUBDIR = "analysis"
XGB_NO_DERIVED = "xgboost_sans_derivees"
NO_MODE = "(aucun mode)"

PipelineBuilder = Callable[[Config, pd.Series], Pipeline]


def build_xgb_without_derived(cfg: Config, y_train: pd.Series) -> Pipeline:
    """XGBoost aux hyperparamètres de la config, préprocesseur SANS les 3 variables dérivées."""
    pipeline = build_xgb_pipeline(cfg, y_train)
    return pipeline.set_params(preprocess=build_preprocessor(cfg, include_derived=False))


CV_MODELS: dict[str, PipelineBuilder] = {
    "xgboost": build_xgb_pipeline,
    "logistic": lambda cfg, _y: build_logistic_baseline(cfg),
    XGB_NO_DERIVED: build_xgb_without_derived,
}


def _split_xy(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.Series]:
    target = cfg.columns.target
    return df.drop(columns=[target]), df[target]


# --- 1. Stabilité : validation croisée sur TRAIN ----------------------------------------


def cross_validate_models(
    cfg: Config,
    train_df: pd.DataFrame,
    models: Mapping[str, PipelineBuilder] = CV_MODELS,
) -> pd.DataFrame:
    """PR-AUC et ROC-AUC par pli (validation croisée stratifiée sur TRAIN uniquement).

    Chaque pipeline est reconstruit dans chaque pli : pour XGBoost, scale_pos_weight ("auto")
    est donc recalculé sur la partie entraînement du pli.
    """
    X, y = _split_xy(train_df, cfg)
    folds = StratifiedKFold(
        n_splits=cfg.analysis.cv_folds, shuffle=True, random_state=cfg.project.random_seed
    )
    rows = []
    for fold, (fit_idx, eval_idx) in enumerate(folds.split(X, y), start=1):
        X_fit, y_fit = X.iloc[fit_idx], y.iloc[fit_idx]
        X_eval, y_eval = X.iloc[eval_idx], y.iloc[eval_idx]
        for name, builder in models.items():
            proba = builder(cfg, y_fit).fit(X_fit, y_fit).predict_proba(X_eval)[:, 1]
            rows.append(
                {
                    "model": name,
                    "fold": fold,
                    "n_eval": len(eval_idx),
                    "pr_auc": float(average_precision_score(y_eval, proba)),
                    "roc_auc": float(roc_auc_score(y_eval, proba)),
                }
            )
    return pd.DataFrame(rows)


def summarize_cv(folds: pd.DataFrame) -> pd.DataFrame:
    """Moyenne et écart-type (ddof=1) des métriques par modèle."""
    summary = folds.groupby("model")[["pr_auc", "roc_auc"]].agg(["mean", "std"])
    summary.columns = ["_".join(map(str, column)) for column in summary.columns.tolist()]
    return summary


# --- 2. Incertitude : bootstrap sur VAL -------------------------------------------------


def bootstrap_costs(
    y_true: pd.Series | np.ndarray,
    alerts: Mapping[str, np.ndarray],
    cost_fn: float,
    cost_fp: float,
    n_iterations: int,
    seed: int,
    reference: str = B2,
) -> pd.DataFrame:
    """Médiane et intervalle à 95 % (percentiles 2,5 et 97,5) du coût de chaque approche.

    Les lignes de VAL sont rééchantillonnées avec remise ; les alertes (seuils figés) ne
    changent pas. Métriques : total_cost_eur, gain_vs_b0_eur (B0 = aucune alerte) et
    share_of_reference_gain (gain / gain de `reference`, B2 par défaut).

    ATTENTION : l'intervalle reflète l'incertitude d'ÉCHANTILLONNAGE de la validation. Il ne
    corrige pas l'optimisme dû au choix du seuil sur ce même jeu : seul le jeu de test, lu
    une seule fois avec les seuils figés, donne une estimation non biaisée.
    """
    truth = np.asarray(y_true).astype(bool)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, truth.size, size=(n_iterations, truth.size))
    sampled_truth = truth[idx]
    b0_cost = sampled_truth.sum(axis=1) * cost_fn

    costs: dict[str, np.ndarray] = {}
    for name, alert in alerts.items():
        sampled_alert = np.asarray(alert).astype(bool)[idx]
        fn = (sampled_truth & ~sampled_alert).sum(axis=1)
        fp = (~sampled_truth & sampled_alert).sum(axis=1)
        costs[name] = fn * cost_fn + fp * cost_fp

    reference_gain = b0_cost - costs[reference]
    rows = []
    for name, cost in costs.items():
        gain = b0_cost - cost
        with np.errstate(divide="ignore", invalid="ignore"):
            share = np.where(reference_gain != 0, gain / reference_gain, np.nan)
        for metric, values in (
            ("total_cost_eur", cost),
            ("gain_vs_b0_eur", gain),
            ("share_of_reference_gain", share),
        ):
            low, median, high = np.nanpercentile(values, [2.5, 50, 97.5])
            rows.append(
                {
                    "approach": name,
                    "metric": metric,
                    "median": float(median),
                    "ci95_low": float(low),
                    "ci95_high": float(high),
                }
            )
    return pd.DataFrame(rows)


# --- 3. Analyse d'erreurs par mode de panne ---------------------------------------------


def error_analysis(df: pd.DataFrame, alerts: Mapping[str, np.ndarray], cfg: Config) -> pd.DataFrame:
    """Pannes détectées / ratées par mode (TWF, HDF, PWF, OSF, RNF) et par approche.

    - une panne à plusieurs modes est comptée dans chacun (colonne multi_mode) ;
    - la ligne NO_MODE regroupe les pannes sans aucun mode actif ;
    - mode_active_target_0 compte les lignes où le mode est actif mais la cible vaut 0
      (anomalies connues d'AI4I, notamment pour RNF).
    """
    target = df[cfg.columns.target].to_numpy().astype(bool)
    modes = df[cfg.columns.leakage].to_numpy().astype(bool)
    n_active = modes.sum(axis=1)

    groups: dict[str, np.ndarray] = {
        mode: target & modes[:, i] for i, mode in enumerate(cfg.columns.leakage)
    }
    groups[NO_MODE] = target & (n_active == 0)

    rows = []
    for i, (group, mask) in enumerate(groups.items()):
        is_mode = group != NO_MODE
        row: dict[str, Any] = {
            "mode": group,
            "failures": int(mask.sum()),
            "multi_mode": int((mask & (n_active > 1)).sum()),
            "mode_active_target_0": int((~target & modes[:, i]).sum()) if is_mode else np.nan,
        }
        for name, alert in alerts.items():
            alert_bool = np.asarray(alert).astype(bool)
            row[f"detected_{name}"] = int((mask & alert_bool).sum())
            row[f"missed_{name}"] = int((mask & ~alert_bool).sum())
        rows.append(row)
    return pd.DataFrame(rows).set_index("mode")


# --- 4. Ablation des variables dérivées -------------------------------------------------


def ablation(
    cfg: Config, train_df: pd.DataFrame, val_df: pd.DataFrame, cv_summary: pd.DataFrame
) -> pd.DataFrame:
    """PR-AUC sur VAL et en CV (train) : XGBoost complet vs sans les 3 variables dérivées."""
    X_train, y_train = _split_xy(train_df, cfg)
    X_val, y_val = _split_xy(val_df, cfg)
    cv_mean = cv_summary["pr_auc_mean"].to_dict()
    cv_std = cv_summary["pr_auc_std"].to_dict()
    rows: list[dict[str, Any]] = []
    for name, builder in (
        ("xgboost", build_xgb_pipeline),
        (XGB_NO_DERIVED, build_xgb_without_derived),
    ):
        proba = builder(cfg, y_train).fit(X_train, y_train).predict_proba(X_val)[:, 1]
        rows.append(
            {
                "model": name,
                "val_pr_auc": float(average_precision_score(y_val, proba)),
                "cv_pr_auc_mean": float(cv_mean[name]),
                "cv_pr_auc_std": float(cv_std[name]),
            }
        )
    full_val_pr_auc = rows[0]["val_pr_auc"]
    for row in rows:
        row["val_pr_auc_delta_vs_full"] = row["val_pr_auc"] - full_val_pr_auc
    return pd.DataFrame(rows).set_index("model")


# --- 5. SHAP ----------------------------------------------------------------------------


def shap_analysis(
    pipeline: Pipeline,
    X_val: pd.DataFrame,
    detected: np.ndarray,
    proba: np.ndarray,
    n_examples: int,
    row_ids: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Importance moyenne |SHAP| par variable, et top 3 des contributions vers « panne »
    pour les n_examples pannes détectées de plus forte probabilité."""
    X_t = pipeline.named_steps["preprocess"].transform(X_val)
    explainer = shap.TreeExplainer(pipeline.named_steps["model"])
    values = np.asarray(explainer.shap_values(X_t))

    importance = (
        pd.DataFrame({"feature": X_t.columns, "mean_abs_shap": np.abs(values).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    candidates = np.flatnonzero(np.asarray(detected).astype(bool))
    chosen = candidates[np.argsort(-proba[candidates], kind="stable")][:n_examples]
    examples = []
    for position in chosen:
        top = np.argsort(-values[position], kind="stable")[:3]
        for rank, feature_idx in enumerate(top, start=1):
            examples.append(
                {
                    "row_id": row_ids.iloc[position],
                    "proba": float(proba[position]),
                    "rank": rank,
                    "feature": X_t.columns[feature_idx],
                    "value": float(X_t.iloc[position, feature_idx]),
                    "shap": float(values[position, feature_idx]),
                }
            )
    return importance, pd.DataFrame(examples)


def save_shap_bar(importance: pd.DataFrame, path: Path) -> None:
    """Graphique en barres horizontales de l'importance moyenne |SHAP| (PNG)."""
    ordered = importance.sort_values("mean_abs_shap")
    fig = Figure(figsize=(7, 0.45 * len(ordered) + 1.2), layout="constrained")
    ax = fig.subplots()
    ax.barh(ordered["feature"], ordered["mean_abs_shap"], color="#3b6ea5")
    ax.set_xlabel("Importance moyenne |SHAP| (log-odds)")
    ax.set_title("XGBoost : importance des variables sur la validation")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=150)


# --- 6. Point d'entrée ------------------------------------------------------------------


def _load_frozen_thresholds(cfg: Config) -> dict[str, Any]:
    path = cfg.paths.artifacts_dir / THRESHOLDS_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"{path} introuvable : lancer d'abord models/evaluate.py")
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("frozen_for_test"):
        raise ValueError(f"{path} n'est pas marqué frozen_for_test : seuils non figés")
    return payload


def _validation_alerts(
    cfg: Config,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    pipeline: Pipeline,
    thresholds: dict[str, Any],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Probabilités XGBoost sur VAL et alertes (seuils figés) de XGBoost, hybride et B2.

    Le seuil d'usure de l'alerte préventive de l'hybride est lu dans thresholds.json
    (valeur figée), pas dans la config : modifier la config ne change pas l'hybride évalué.
    """
    if "preventive_tool_wear_min" not in thresholds:
        raise KeyError("thresholds.json ne contient pas preventive_tool_wear_min (valeur figée)")
    X_train, y_train = _split_xy(train_df, cfg)
    X_val, _ = _split_xy(val_df, cfg)
    proba = pipeline.predict_proba(X_val)[:, 1]
    preventive = (
        PreventiveBaseline(
            tool_wear_col=COLUMN_MAP["tool_wear_min"],
            threshold_min=float(thresholds["preventive_tool_wear_min"]),
        )
        .fit(X_train, y_train)
        .predict(X_val)
    )
    expert = ExpertRulesBaseline.from_config(cfg).fit(X_train, y_train).predict(X_val)
    frozen = thresholds["thresholds"]
    alerts = {
        XGB: proba >= frozen["xgboost"]["threshold"],
        HYBRID: (proba >= frozen["hybrid"]["threshold"]) | preventive.astype(bool),
        B2: expert.astype(bool),
    }
    return proba, alerts


def main() -> None:
    cfg = load_config()
    verify_fingerprint(cfg)  # refuse de tourner si le modèle n'est plus celui des seuils figés
    processed = cfg.paths.processed_dir
    train_df = pd.read_csv(processed / "train.csv")
    val_df = pd.read_csv(processed / "val.csv")
    thresholds = _load_frozen_thresholds(cfg)
    pipeline = joblib.load(cfg.paths.artifacts_dir / "xgboost.joblib")

    out_dir = cfg.paths.reports_dir / ANALYSIS_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    d = cfg.decision

    proba, alerts = _validation_alerts(cfg, train_df, val_df, pipeline, thresholds)
    X_val, y_val = _split_xy(val_df, cfg)

    cv_folds = cross_validate_models(cfg, train_df)
    cv_summary = summarize_cv(cv_folds)
    boot = bootstrap_costs(
        y_val,
        alerts,
        d.cost_missed_failure_eur,
        d.cost_false_alarm_eur,
        cfg.analysis.bootstrap_iterations,
        cfg.project.random_seed,
    )
    errors = error_analysis(val_df, alerts, cfg)
    abl = ablation(cfg, train_df, val_df, cv_summary)
    importance, examples = shap_analysis(
        pipeline,
        X_val,
        alerts[XGB] & y_val.to_numpy().astype(bool),
        proba,
        cfg.analysis.shap_examples,
        val_df[cfg.columns.row_id],
    )

    outputs = {
        "cv_folds.csv": cv_folds,
        "cv_summary.csv": cv_summary,
        "bootstrap_costs.csv": boot,
        "error_analysis.csv": errors,
        "ablation.csv": abl,
        "shap_importance.csv": importance,
        "shap_examples.csv": examples,
    }
    for filename, frame in outputs.items():
        frame.to_csv(out_dir / filename, index=frame.index.name is not None)
    save_shap_bar(importance, out_dir / "shap_importance.png")

    pd.set_option("display.width", 140)
    print("\n=== Stabilité (CV stratifiée sur TRAIN) ===\n")
    print(cv_summary.round(4).to_string())
    print("\n=== Coût sur VAL : médiane [IC 95 %] (bootstrap, seuils figés) ===\n")
    print(boot.round(2).to_string(index=False))
    print("  (incertitude d'échantillonnage seulement ; seuils choisis sur ce même jeu)")
    print("\n=== Pannes par mode (VAL) ===\n")
    print(errors.to_string())
    print("\n=== Ablation des variables dérivées ===\n")
    print(abl.round(4).to_string())
    print("\n=== SHAP : importance moyenne |SHAP| ===\n")
    print(importance.round(4).to_string(index=False))
    print("\n=== SHAP : pannes détectées expliquées ===\n")
    print(examples.round(4).to_string(index=False))

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    with mlflow.start_run(run_name=MLFLOW_RUN_NAME):
        mlflow.set_tag("git_commit", git_commit())
        mlflow.log_params(
            {
                "cv_folds": cfg.analysis.cv_folds,
                "bootstrap_iterations": cfg.analysis.bootstrap_iterations,
                "random_seed": cfg.project.random_seed,
            }
        )
        mlflow.log_artifacts(str(out_dir), artifact_path=ANALYSIS_SUBDIR)
    logger.info("Analyses écrites dans %s", out_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    main()
