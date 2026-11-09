"""Tests des utilitaires transverses."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from predictive_maintenance_iot import utils
from predictive_maintenance_iot.utils import git_commit

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_git_commit_format() -> None:
    assert re.fullmatch(r"[0-9a-f]{40}|unknown", git_commit())


@pytest.mark.skipif(shutil.which("git") is None, reason="git non installé")
def test_git_commit_matches_repository_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT, check=False
    )
    if head.returncode != 0:
        pytest.skip("le dossier du projet n'est pas un dépôt git")

    monkeypatch.chdir(tmp_path)  # dossier courant hors du dépôt : le hash doit rester correct
    assert git_commit() == head.stdout.strip()


def test_git_commit_unknown_when_git_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_git(*args: Any, **kwargs: Any) -> Any:
        raise FileNotFoundError("git")

    monkeypatch.setattr(utils.subprocess, "run", missing_git)
    assert git_commit() == "unknown"
