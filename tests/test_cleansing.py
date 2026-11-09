"""Tests du découpage train / validation / test et de l'écriture des lots."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from predictive_maintenance_iot.config import Config, load_config
from predictive_maintenance_iot.data.cleansing import REPORT_FILENAME, run_cleansing, split_data

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
N_ROWS = 1_000
N_FAILURES = 34  # 3,4 %, comme le vrai jeu AI4I 2020


@pytest.fixture(scope="module")
def cfg() -> Config:
    return load_config(CONFIG_PATH)


def _with_seed(cfg: Config, seed: int) -> Config:
    return cfg.model_copy(update={"project": cfg.project.model_copy(update={"random_seed": seed})})


@pytest.fixture
def synthetic_df(cfg: Config) -> pd.DataFrame:
    """1 000 lignes au format du CSV brut, valides vis-à-vis de la config, ~3,4 % de pannes."""
    rng = np.random.default_rng(0)
    cols = cfg.columns
    product_types = rng.choice(cols.product_type.allowed, size=N_ROWS)
    failures = np.zeros(N_ROWS, dtype=int)
    failures[rng.choice(N_ROWS, size=N_FAILURES, replace=False)] = 1

    data: dict[str, object] = {
        cols.row_id: np.arange(1, N_ROWS + 1),
        "Product ID": [f"{t}{i:05d}" for i, t in enumerate(product_types)],
        cols.product_type.column: product_types,
    }
    for column in cols.numeric:
        bound = cfg.sensor_bounds[column]
        data[column] = rng.uniform(bound.min, bound.max, size=N_ROWS).round(1)
    data[cols.target] = failures
    for column in cols.leakage:
        data[column] = np.zeros(N_ROWS, dtype=int)
    data[cols.leakage[1]] = failures  # une cause de panne cohérente avec la cible
    return pd.DataFrame(data)


def _ids(df: pd.DataFrame, cfg: Config) -> set[int]:
    return set(df[cfg.columns.row_id])


# --- split_data -------------------------------------------------------------------------


def test_split_sizes_close_to_70_15_15(synthetic_df: pd.DataFrame, cfg: Config) -> None:
    train, val, test = split_data(synthetic_df, cfg)
    total = len(synthetic_df)

    assert len(train) / total == pytest.approx(0.70, abs=0.01)
    assert len(val) / total == pytest.approx(cfg.split.val_size, abs=0.01)
    assert len(test) / total == pytest.approx(cfg.split.test_size, abs=0.01)


def test_failure_rate_preserved_in_each_split(synthetic_df: pd.DataFrame, cfg: Config) -> None:
    target = cfg.columns.target
    global_rate = synthetic_df[target].mean()

    for split in split_data(synthetic_df, cfg):
        assert split[target].mean() == pytest.approx(global_rate, abs=0.005)


def test_splits_are_disjoint_and_complete(synthetic_df: pd.DataFrame, cfg: Config) -> None:
    train, val, test = (_ids(s, cfg) for s in split_data(synthetic_df, cfg))

    assert not train & val
    assert not train & test
    assert not val & test
    assert train | val | test == _ids(synthetic_df, cfg)


def test_same_seed_gives_identical_splits(synthetic_df: pd.DataFrame, cfg: Config) -> None:
    first = split_data(synthetic_df, cfg)
    second = split_data(synthetic_df, cfg)

    for a, b in zip(first, second, strict=True):
        pd.testing.assert_frame_equal(a, b)


def test_different_seed_gives_different_splits(synthetic_df: pd.DataFrame, cfg: Config) -> None:
    _, _, test_a = split_data(synthetic_df, cfg)
    _, _, test_b = split_data(synthetic_df, _with_seed(cfg, cfg.project.random_seed + 1))

    assert _ids(test_a, cfg) != _ids(test_b, cfg)


def test_exact_duplicates_removed_before_split(
    synthetic_df: pd.DataFrame, cfg: Config, caplog: pytest.LogCaptureFixture
) -> None:
    duplicated = pd.concat([synthetic_df, synthetic_df.iloc[:25]], ignore_index=True)
    caplog.set_level("INFO")

    train, val, test = split_data(duplicated, cfg)

    assert len(train) + len(val) + len(test) == N_ROWS
    combined = pd.concat([train, val, test])
    assert not combined[cfg.columns.row_id].duplicated().any()
    assert "Doublons exacts supprimés : 25" in caplog.text


def test_split_data_does_not_modify_input(synthetic_df: pd.DataFrame, cfg: Config) -> None:
    before = synthetic_df.copy()
    split_data(synthetic_df, cfg)
    pd.testing.assert_frame_equal(synthetic_df, before)


# --- run_cleansing ----------------------------------------------------------------------


def _cfg_in_tmp(cfg: Config, df: pd.DataFrame, tmp_path: Path) -> tuple[Config, Path]:
    """Écrit `df` comme CSV brut dans tmp_path et renvoie une config qui pointe dessus."""
    raw_path = tmp_path / "raw" / "ai4i2020.csv"
    raw_path.parent.mkdir()
    df.to_csv(raw_path, index=False)
    out_dir = tmp_path / "processed"  # volontairement absent : doit être créé
    tmp_cfg = cfg.model_copy(
        update={
            "paths": cfg.paths.model_copy(update={"raw_data": raw_path, "processed_dir": out_dir})
        }
    )
    return tmp_cfg, out_dir


def test_run_cleansing_writes_all_files(
    synthetic_df: pd.DataFrame, cfg: Config, tmp_path: Path
) -> None:
    tmp_cfg, out_dir = _cfg_in_tmp(cfg, synthetic_df, tmp_path)

    run_cleansing(tmp_cfg)

    for name in ("train.csv", "val.csv", "test.csv", REPORT_FILENAME):
        assert (out_dir / name).is_file(), f"{name} manquant"

    report = json.loads((out_dir / REPORT_FILENAME).read_text(encoding="utf-8"))
    assert report["random_seed"] == cfg.project.random_seed
    assert "generated_at" in report
    assert set(report["splits"]) == {"train", "val", "test"}
    assert sum(s["rows"] for s in report["splits"].values()) == N_ROWS
    assert sum(s["failures"] for s in report["splits"].values()) == N_FAILURES
    for name, stats in report["splits"].items():
        written = pd.read_csv(out_dir / f"{name}.csv")
        assert stats["rows"] == len(written)
        assert stats["failure_rate"] == pytest.approx(stats["failures"] / stats["rows"])


def test_run_cleansing_keeps_integer_columns(
    synthetic_df: pd.DataFrame, cfg: Config, tmp_path: Path
) -> None:
    # Colonne numérique que pandera convertit en float lors de la validation.
    column = "Rotational speed [rpm]"
    synthetic_df[column] = synthetic_df[column].round().astype(int)
    tmp_cfg, out_dir = _cfg_in_tmp(cfg, synthetic_df, tmp_path)
    assert pd.api.types.is_integer_dtype(pd.read_csv(tmp_cfg.paths.raw_data)[column])

    run_cleansing(tmp_cfg)

    train = pd.read_csv(out_dir / "train.csv")
    assert pd.api.types.is_integer_dtype(train[column])
    raw_text = (out_dir / "train.csv").read_text(encoding="utf-8")
    first_value = raw_text.splitlines()[1].split(",")[synthetic_df.columns.get_loc(column)]
    assert "." not in first_value
