"""Examen final sur le jeu de TEST : une seule fois, avec les seuils figés, sans rien optimiser.

C'est le SEUL module autorisé à lire test.csv. Garde-fous, vérifiés AVANT toute lecture :
1. l'empreinte du modèle correspond à celle enregistrée au gel des seuils ;
2. thresholds.json est marqué frozen_for_test ;
3. artifacts/test_results.json n'existe pas encore (l'examen ne se passe qu'une fois) ;
   l'option --force permet de relancer, avec un avertissement journalisé.

Aucun seuil n'est choisi ici : seuils, coûts, contrainte et usure préventive viennent tous de
thresholds.json. Les règles expertes (B2) viennent de la config, qui ne contient aucun
paramètre appris.

Usage : uv run python -m predictive_maintenance_iot.models.final_test [--force]
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd

from predictive_maintenance_iot.config import Config, load_config
from predictive_maintenance_iot.data.validation import COLUMN_MAP
from predictive_maintenance_iot.models.analysis import bootstrap_costs, error_analysis
from predictive_maintenance_iot.models.baseline import ExpertRulesBaseline, PreventiveBaseline
from predictive_maintenance_iot.models.evaluate import (
    B0,
    B1,
    B2,
    HYBRID,
    THRESHOLDS_FILENAME,
    XGB,
    cost_at_threshold,
)
from predictive_maintenance_iot.models.fingerprint import (
    MODEL_FILENAME,
    sha256_file,
    verify_fingerprint,
)
from predictive_maintenance_iot.utils import git_commit

logger = logging.getLogger(__name__)

RESULTS_FILENAME = "test_results.json"
REPORT_FILENAME = "final_test.md"
MLFLOW_RUN_NAME = "final-test"
APPROACHES: tuple[str, ...] = (B0, B1, B2, XGB, HYBRID)


class TestAlreadyRunError(RuntimeError):
    """L'examen final a déjà été passé (test_results.json existe)."""

    __test__ = False  # pas une classe de test pytest malgré son nom


# --- Garde-fous -------------------------------------------------------------------------


def _check_preconditions(cfg: Config, force: bool) -> dict[str, Any]:
    """Empreinte, seuils figés, examen non encore passé. Renvoie le contenu de thresholds.json."""
    verify_fingerprint(cfg)

    thresholds_path = cfg.paths.artifacts_dir / THRESHOLDS_FILENAME
    if not thresholds_path.is_file():
        raise FileNotFoundError(f"{thresholds_path} introuvable : lancer d'abord evaluate.py")
    frozen: dict[str, Any] = json.loads(thresholds_path.read_text(encoding="utf-8"))
    if frozen.get("frozen_for_test") is not True:
        raise ValueError(f"{thresholds_path} n'est pas marqué frozen_for_test=true : examen refusé")

    results_path = cfg.paths.artifacts_dir / RESULTS_FILENAME
    if results_path.exists():
        if not force:
            raise TestAlreadyRunError(
                f"L'examen final a déjà été passé ({results_path} existe). Le relancer reviendrait "
                "à regarder le test plusieurs fois. Utiliser --force uniquement en connaissance "
                "de cause (ex. correction d'un bug sans rapport avec les seuils)."
            )
        logger.warning(
            "--force : %s existe déjà et va être ÉCRASÉ. Ce second passage sur le test doit "
            "être justifié et signalé dans le rapport.",
            results_path,
        )
    return frozen


# --- Calculs à seuils figés -------------------------------------------------------------

FROZEN_KEYS: dict[str, str] = {XGB: "xgboost", HYBRID: "hybrid"}


def frozen_threshold(frozen: Mapping[str, Any], key: str, ratio: str) -> float:
    """Seuil figé de `key` ("xgboost" / "hybrid") : celui du ratio de la config si
    ratio == "config", sinon celui de thresholds_by_ratio[ratio]. Lecture seule."""
    entry = frozen["thresholds"][key]
    return float(entry["threshold"] if ratio == "config" else entry["thresholds_by_ratio"][ratio])


def _test_alerts(
    cfg: Config, X_test: pd.DataFrame, proba: np.ndarray, frozen: Mapping[str, Any], ratio: str
) -> dict[str, np.ndarray]:
    """Alertes de chaque approche sur le test, aux seuils figés pour `ratio` ("config", "5"…)."""
    # fit n'apprend rien pour les règles : il vérifie seulement la présence des colonnes.
    preventive = (
        PreventiveBaseline(
            tool_wear_col=COLUMN_MAP["tool_wear_min"],
            threshold_min=float(frozen["preventive_tool_wear_min"]),
        )
        .fit(X_test)
        .predict(X_test)
        .astype(bool)
    )
    expert = ExpertRulesBaseline.from_config(cfg).fit(X_test).predict(X_test).astype(bool)
    return {
        B0: np.zeros(len(X_test), dtype=bool),
        B1: preventive,
        B2: expert,
        XGB: proba >= frozen_threshold(frozen, "xgboost", ratio),
        HYBRID: (proba >= frozen_threshold(frozen, "hybrid", ratio)) | preventive,
    }


def _results_table(
    y_test: pd.Series,
    proba: np.ndarray,
    alerts: Mapping[str, np.ndarray],
    frozen: Mapping[str, Any],
    ratio: str,
    cost_fn: float,
    cost_fp: float,
) -> pd.DataFrame:
    """Une ligne par approche ; aucun choix de seuil (cost_at_threshold à seuil fixé)."""
    preventive = alerts[B1]
    rows = []
    for approach in APPROACHES:
        if approach in FROZEN_KEYS:
            t = frozen_threshold(frozen, FROZEN_KEYS[approach], ratio)
            forced = preventive if approach == HYBRID else None
            result = cost_at_threshold(y_test, proba, t, cost_fn, cost_fp, forced)
        else:
            t = np.nan
            result = cost_at_threshold(y_test, alerts[approach].astype(int), 1, cost_fn, cost_fp)
        rows.append(
            {
                "ratio_fn_fp": ratio,
                "approach": approach,
                "threshold": t,
                "detected": result["tp"],
                "false_alarms": result["fp"],
                "missed": result["fn"],
                "precision": result["precision"],
                "model_precision": result["model_precision"],
                "total_cost_eur": result["total_cost_eur"],
            }
        )
    df = pd.DataFrame(rows)
    costs = dict(zip(df["approach"], df["total_cost_eur"], strict=True))
    df["gain_vs_b0_eur"] = costs[B0] - df["total_cost_eur"]
    b2_gain = costs[B0] - costs[B2]
    df["share_of_b2_gain"] = df["gain_vs_b0_eur"] / b2_gain if b2_gain else np.nan
    return df


# --- Rapport ----------------------------------------------------------------------------


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Lignes JSON-compatibles (NaN → null, types numpy → types Python)."""
    records: list[dict[str, Any]] = json.loads(df.to_json(orient="records", force_ascii=False))
    return records


def _markdown_table(df: pd.DataFrame, decimals: int = 3) -> str:
    def fmt(value: Any) -> str:
        if isinstance(value, float):
            return "—" if np.isnan(value) else f"{value:,.{decimals}f}".replace(",", " ")
        return str(value)

    header = "| " + " | ".join(map(str, df.columns)) + " |"
    separator = "|" + "|".join("---" for _ in df.columns) + "|"
    lines = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, separator, *lines])


def _write_report(
    path: Path, payload: Mapping[str, Any], tables: Mapping[str, pd.DataFrame]
) -> None:
    sections = [
        "# Examen final sur le jeu de test",
        "",
        f"- Date : {payload['generated_at']}",
        f"- Commit : `{payload['git_commit']}`",
        f"- Modèle (SHA-256) : `{payload['model_sha256']}`",
        f"- Seuils figés (SHA-256 de thresholds.json) : `{payload['thresholds_sha256']}`",
        f"- Lignes de test : {payload['n_test']} dont {payload['n_failures']} pannes",
        f"- Relancé avec --force : {'oui' if payload['forced_rerun'] else 'non'}",
        "",
        "Aucun seuil n'a été optimisé sur ce jeu : tous proviennent de thresholds.json.",
    ]
    for title, table in tables.items():
        sections += ["", f"## {title}", "", _markdown_table(table)]
    path.write_text("\n".join(sections) + "\n", encoding="utf-8")


# --- Point d'entrée ---------------------------------------------------------------------


def run_final_test(cfg: Config, force: bool = False) -> dict[str, Any]:
    """Passe l'examen final et renvoie le contenu écrit dans test_results.json."""
    frozen = _check_preconditions(cfg, force)  # AVANT toute lecture du test

    target = cfg.columns.target
    test_df = pd.read_csv(cfg.paths.processed_dir / "test.csv")
    X_test, y_test = test_df.drop(columns=[target]), test_df[target]
    model_path = cfg.paths.artifacts_dir / MODEL_FILENAME
    proba = joblib.load(model_path).predict_proba(X_test)[:, 1]

    cost_fp = float(frozen["cost_false_alarm_eur"])
    cost_fn_config = float(frozen["cost_missed_failure_eur"])
    ratios = list(frozen["thresholds"]["hybrid"]["thresholds_by_ratio"])

    config_alerts = _test_alerts(cfg, X_test, proba, frozen, "config")
    at_config = _results_table(
        y_test, proba, config_alerts, frozen, "config", cost_fn_config, cost_fp
    )
    by_ratio = pd.concat(
        [
            _results_table(
                y_test,
                proba,
                _test_alerts(cfg, X_test, proba, frozen, r),
                frozen,
                r,
                float(r) * cost_fp,
                cost_fp,
            )
            for r in ratios
        ],
        ignore_index=True,
    )
    boot = bootstrap_costs(
        y_test,
        config_alerts,
        cost_fn_config,
        cost_fp,
        cfg.analysis.bootstrap_iterations,
        cfg.project.random_seed,
    )
    boot = boot[boot["metric"].isin(["total_cost_eur", "share_of_reference_gain"])]
    errors = error_analysis(
        test_df, {k: config_alerts[k] for k in (XGB, HYBRID, B2)}, cfg
    ).reset_index()

    payload: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "forced_rerun": force,
        "model_sha256": sha256_file(model_path),
        "thresholds_sha256": sha256_file(cfg.paths.artifacts_dir / THRESHOLDS_FILENAME),
        "n_test": len(test_df),
        "n_failures": int(y_test.sum()),
        "cost_missed_failure_eur": cost_fn_config,
        "cost_false_alarm_eur": cost_fp,
        "primary_approach": frozen.get("primary_approach"),
        "secondary_approach": frozen.get("secondary_approach"),
        "results_at_config_ratio": _records(at_config),
        "results_by_ratio": _records(by_ratio),
        "bootstrap_config_ratio": _records(boot),
        "error_analysis": _records(errors),
    }

    results_path = cfg.paths.artifacts_dir / RESULTS_FILENAME
    results_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    cfg.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = cfg.paths.reports_dir / REPORT_FILENAME
    _write_report(
        report_path,
        payload,
        {
            "Résultats au ratio de la config": at_config.drop(columns=["ratio_fn_fp"]),
            "Résultats par ratio (seuils figés par ratio)": by_ratio,
            "Bootstrap au ratio de la config (médiane, IC 95 %)": boot,
            "Pannes par mode": errors,
        },
    )

    costs = dict(zip(at_config["approach"], at_config["total_cost_eur"], strict=True))
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    with mlflow.start_run(run_name=MLFLOW_RUN_NAME):
        mlflow.set_tags({"git_commit": payload["git_commit"], "forced_rerun": str(force)})
        mlflow.log_metrics(
            {
                "test_hybrid_cost_eur": float(costs[HYBRID]),
                "test_xgb_cost_eur": float(costs[XGB]),
                "test_b2_cost_eur": float(costs[B2]),
                "test_b0_cost_eur": float(costs[B0]),
            }
        )
        mlflow.log_artifact(str(results_path))
        mlflow.log_artifact(str(report_path))

    logger.info("Examen final écrit dans %s et %s", results_path, report_path)
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Examen final sur le jeu de test (une fois).")
    parser.add_argument(
        "--force",
        action="store_true",
        help="relancer malgré un test_results.json existant (journalise un avertissement)",
    )
    args = parser.parse_args(argv)
    payload = run_final_test(load_config(), force=args.force)

    at_config = pd.DataFrame(payload["results_at_config_ratio"])
    print("\nExamen final (TEST, seuils figés, ratio de la config) :\n")
    print(at_config.drop(columns=["ratio_fn_fp"]).round(3).to_string(index=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    try:
        main()
    except TestAlreadyRunError as exc:
        raise SystemExit(f"ERREUR : {exc}") from exc
