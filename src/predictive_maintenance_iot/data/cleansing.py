"""Nettoyage des données brutes et découpage train / validation / test vers data/processed/.

Aucune transformation apprise ici (ni normalisation, ni encodage) : seulement la suppression
des doublons exacts et le découpage, pour éviter toute fuite d'information du test vers le train.
Toutes les colonnes sont conservées ; la sélection des variables se fait à l'entraînement.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from predictive_maintenance_iot.config import Config, load_config
from predictive_maintenance_iot.data.validation import validate_raw

logger = logging.getLogger(__name__)

SPLIT_NAMES: tuple[str, str, str] = ("train", "val", "test")
REPORT_FILENAME = "split_report.json"


def split_data(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Supprime les doublons exacts puis découpe en (train, val, test).

    Fonction pure : aucune lecture ni écriture de fichier, le DataFrame d'entrée n'est pas modifié.
    Le découpage se fait en deux temps pour que val représente bien `val_size` du total :
    test = test_size du total, puis val = val_size / (1 - test_size) du reste.
    """
    target = cfg.columns.target
    seed = cfg.project.random_seed
    test_size = cfg.split.test_size
    val_size = cfg.split.val_size

    deduplicated = df.drop_duplicates(ignore_index=True)
    n_duplicates = len(df) - len(deduplicated)
    logger.info("Doublons exacts supprimés : %d", n_duplicates)

    def _stratify(frame: pd.DataFrame) -> pd.Series | None:
        return frame[target] if cfg.split.stratify else None

    train_val, test = train_test_split(
        deduplicated,
        test_size=test_size,
        random_state=seed,
        stratify=_stratify(deduplicated),
    )
    train, val = train_test_split(
        train_val,
        test_size=val_size / (1 - test_size),
        random_state=seed,
        stratify=_stratify(train_val),
    )

    logger.info(
        "Découpage (graine %d) : train=%d, val=%d, test=%d", seed, len(train), len(val), len(test)
    )
    return train, val, test


def _split_stats(df: pd.DataFrame, target: str) -> dict[str, Any]:
    n_rows = len(df)
    n_failures = int(df[target].sum())
    return {
        "rows": n_rows,
        "failures": n_failures,
        "failure_rate": n_failures / n_rows if n_rows else 0.0,
    }


def run_cleansing(cfg: Config) -> None:
    """Lit le CSV brut, le valide, le découpe et écrit les 3 lots + un rapport JSON."""
    raw_path = cfg.paths.raw_data
    out_dir = cfg.paths.processed_dir

    logger.info("Lecture de %s", raw_path)
    df = pd.read_csv(raw_path)
    # Contrôle seulement : le DataFrame renvoyé par pandera (colonnes numériques converties
    # en float) est ignoré, pour conserver les types d'origine du CSV dans les lots écrits.
    validate_raw(df, cfg)
    splits = dict(zip(SPLIT_NAMES, split_data(df, cfg), strict=True))

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, split in splits.items():
        split.to_csv(out_dir / f"{name}.csv", index=False)

    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "random_seed": cfg.project.random_seed,
        "source": str(raw_path),
        "splits": {name: _split_stats(split, cfg.columns.target) for name, split in splits.items()},
    }
    (out_dir / REPORT_FILENAME).write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Lots et rapport écrits dans %s", out_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    run_cleansing(load_config())
