"""Empreinte (SHA-256) du modèle figé, pour garantir que les analyses utilisent le bon fichier.

Le modèle {artifacts_dir}/xgboost.joblib n'est pas versionné (trop lourd, régénérable), mais ses
seuils figés le sont (thresholds.json). L'empreinte, versionnée elle aussi, relie les deux :
une analyse refuse de tourner si le modèle présent n'est plus celui dont les seuils ont été figés.

Usage (une fois, après l'entraînement et le gel des seuils) :
    uv run python -m predictive_maintenance_iot.models.fingerprint
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from predictive_maintenance_iot.config import Config, load_config
from predictive_maintenance_iot.utils import git_commit

logger = logging.getLogger(__name__)

MODEL_FILENAME = "xgboost.joblib"
FINGERPRINT_FILENAME = "model_fingerprint.json"
_CHUNK_SIZE = 1 << 20  # 1 Mio


class FingerprintMismatchError(RuntimeError):
    """Le modèle présent ne correspond pas à l'empreinte enregistrée."""


def sha256_file(path: Path) -> str:
    """SHA-256 hexadécimal du contenu du fichier (lu par blocs)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _paths(cfg: Config) -> tuple[Path, Path]:
    artifacts = cfg.paths.artifacts_dir
    return artifacts / MODEL_FILENAME, artifacts / FINGERPRINT_FILENAME


def write_fingerprint(cfg: Config) -> Path:
    """Calcule l'empreinte du modèle et l'écrit dans {artifacts_dir}/model_fingerprint.json."""
    model_path, fingerprint_path = _paths(cfg)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Modèle introuvable : {model_path}. Lancer d'abord models/train.py."
        )
    payload: dict[str, Any] = {
        "model_path": model_path.as_posix(),
        "sha256": sha256_file(model_path),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
    }
    fingerprint_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Empreinte %s écrite dans %s", payload["sha256"][:12], fingerprint_path)
    return fingerprint_path


def verify_fingerprint(cfg: Config) -> None:
    """Lève une erreur explicite si le modèle ou l'empreinte manque, ou s'ils divergent."""
    model_path, fingerprint_path = _paths(cfg)
    if not fingerprint_path.is_file():
        raise FileNotFoundError(
            f"Empreinte introuvable : {fingerprint_path}. Lancer "
            "`uv run python -m predictive_maintenance_iot.models.fingerprint` "
            "après l'entraînement et le gel des seuils."
        )
    if not model_path.is_file():
        raise FileNotFoundError(f"Modèle introuvable : {model_path}.")

    expected = json.loads(fingerprint_path.read_text(encoding="utf-8"))["sha256"]
    actual = sha256_file(model_path)
    if actual != expected:
        raise FingerprintMismatchError(
            f"{model_path} ne correspond pas à l'empreinte enregistrée "
            f"(attendu {expected[:12]}…, obtenu {actual[:12]}…). Le modèle a changé depuis "
            "le gel des seuils : les résultats ne seraient pas ceux du modèle figé."
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    write_fingerprint(load_config())
