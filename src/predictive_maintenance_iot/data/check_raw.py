"""Téléchargement des données brutes vers data/raw/."""
"""Contrôle du jeu de données brut AI4I 2020, téléchargé manuellement depuis l'UCI."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

EXPECTED_ROWS = 10_000
EXPECTED_COLUMNS = [
    "UDI", "Product ID", "Type", "Air temperature [K]", "Process temperature [K]",
    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]", "Machine failure",
    "TWF", "HDF", "PWF", "OSF", "RNF",
]


def check_dataset(csv_path: Path) -> pd.DataFrame:
    """Lève une erreur si le fichier n'est pas le jeu AI4I 2020 attendu."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} introuvable : télécharge le jeu depuis l'UCI (voir README)."
        )
    df = pd.read_csv(csv_path)
    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"Colonnes inattendues : {list(df.columns)}")
    if len(df) != EXPECTED_ROWS:
        raise ValueError(f"{len(df)} lignes au lieu de {EXPECTED_ROWS}")
    logger.info("Jeu valide : %d lignes, %d pannes", len(df), int(df["Machine failure"].sum()))
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    check_dataset(Path("data/raw/ai4i2020.csv"))