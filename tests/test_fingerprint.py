"""Tests de l'empreinte du modèle figé."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from predictive_maintenance_iot.config import Config, load_config
from predictive_maintenance_iot.models.fingerprint import (
    FINGERPRINT_FILENAME,
    MODEL_FILENAME,
    FingerprintMismatchError,
    sha256_file,
    verify_fingerprint,
    write_fingerprint,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    base = load_config(CONFIG_PATH)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / MODEL_FILENAME).write_bytes(b"modele-factice" * 1000)
    return base.model_copy(
        update={"paths": base.paths.model_copy(update={"artifacts_dir": artifacts})}
    )


def test_sha256_is_stable_for_same_file(cfg: Config) -> None:
    model = cfg.paths.artifacts_dir / MODEL_FILENAME
    first, second = sha256_file(model), sha256_file(model)

    assert first == second
    assert re.fullmatch(r"[0-9a-f]{64}", first)


def test_write_then_verify(cfg: Config) -> None:
    path = write_fingerprint(cfg)

    assert path == cfg.paths.artifacts_dir / FINGERPRINT_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["sha256"] == sha256_file(cfg.paths.artifacts_dir / MODEL_FILENAME)
    assert payload["model_path"].endswith(f"artifacts/{MODEL_FILENAME}")
    assert {"generated_at", "git_commit"} <= set(payload)
    verify_fingerprint(cfg)  # ne lève rien

    write_fingerprint(cfg)  # réécriture : même empreinte pour le même fichier
    assert json.loads(path.read_text(encoding="utf-8"))["sha256"] == payload["sha256"]


def test_modified_model_raises(cfg: Config) -> None:
    write_fingerprint(cfg)
    model = cfg.paths.artifacts_dir / MODEL_FILENAME
    model.write_bytes(model.read_bytes() + b"x")

    with pytest.raises(FingerprintMismatchError, match="ne correspond pas"):
        verify_fingerprint(cfg)


def test_missing_fingerprint_raises_clear_error(cfg: Config) -> None:
    with pytest.raises(FileNotFoundError, match="Empreinte introuvable.*fingerprint"):
        verify_fingerprint(cfg)


def test_missing_model_raises_clear_error(cfg: Config) -> None:
    write_fingerprint(cfg)
    (cfg.paths.artifacts_dir / MODEL_FILENAME).unlink()

    with pytest.raises(FileNotFoundError, match="Modèle introuvable"):
        verify_fingerprint(cfg)
    with pytest.raises(FileNotFoundError, match="Modèle introuvable"):
        write_fingerprint(cfg)
