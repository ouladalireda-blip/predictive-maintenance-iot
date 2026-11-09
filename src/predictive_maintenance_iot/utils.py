"""Utilitaires transverses (traçabilité des exécutions)."""

from __future__ import annotations

import subprocess
from pathlib import Path


def git_commit() -> str:
    """Hash du commit du code, ou "unknown" si git ou le dépôt est indisponible.

    La commande est exécutée dans le dossier du package, pas dans le dossier courant :
    le hash reste correct quel que soit l'endroit d'où le script est lancé.
    """
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
