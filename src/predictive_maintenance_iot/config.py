"""Chargement et validation de la configuration du projet (config/config.yaml).

Chaque section du YAML est représentée par un modèle Pydantic v2 strict (clés inconnues
refusées, objets immuables). `load_config` est le seul point d'entrée pour lire la config.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

DEFAULT_CONFIG_PATH = Path("config/config.yaml")


class _Section(BaseModel):
    """Base commune : clés inconnues interdites, valeurs non modifiables après chargement."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ProjectConfig(_Section):
    random_seed: int


class PathsConfig(_Section):
    raw_data: Path
    processed_dir: Path
    artifacts_dir: Path


class ProductTypeConfig(_Section):
    column: str
    allowed: list[str]

    @model_validator(mode="after")
    def _allowed_non_empty_unique(self) -> Self:
        if not self.allowed:
            raise ValueError("product_type.allowed ne doit pas être vide")
        duplicates = sorted({v for v in self.allowed if self.allowed.count(v) > 1})
        if duplicates:
            raise ValueError(f"product_type.allowed contient des doublons : {duplicates}")
        return self


class ColumnsConfig(_Section):
    target: str
    ids: list[str]
    row_id: str
    product_type: ProductTypeConfig
    leakage: list[str]
    categorical: list[str]
    numeric: list[str]

    @model_validator(mode="after")
    def _row_id_in_ids(self) -> Self:
        if self.row_id not in self.ids:
            raise ValueError(f"row_id '{self.row_id}' doit figurer dans ids {self.ids}")
        return self

    @model_validator(mode="after")
    def _product_type_in_categorical(self) -> Self:
        column = self.product_type.column
        if column not in self.categorical:
            raise ValueError(
                f"product_type.column '{column}' doit figurer dans categorical {self.categorical}"
            )
        return self

    @model_validator(mode="after")
    def _no_forbidden_features(self) -> Self:
        forbidden = {self.target, *self.ids, *self.leakage}
        features = [*self.numeric, *self.categorical]
        overlap = sorted(col for col in features if col in forbidden)
        if overlap:
            raise ValueError(
                "colonnes interdites comme variables explicatives "
                f"(target / ids / leakage) présentes dans numeric ou categorical : {overlap}"
            )
        return self


class SensorBound(_Section):
    min: float
    max: float

    @model_validator(mode="after")
    def _min_below_max(self) -> Self:
        if self.min >= self.max:
            raise ValueError(f"borne min ({self.min}) doit être < borne max ({self.max})")
        return self


class SplitConfig(_Section):
    test_size: Annotated[float, Field(gt=0, lt=0.5)]
    val_size: Annotated[float, Field(gt=0, lt=0.5)]
    stratify: bool

    @model_validator(mode="after")
    def _sizes_leave_enough_train(self) -> Self:
        total = self.test_size + self.val_size
        if total >= 0.6:
            raise ValueError(f"test_size + val_size doit être < 0.6 (reçu {total:.2f})")
        return self


class XGBoostConfig(_Section):
    n_estimators: int
    max_depth: int
    learning_rate: float
    subsample: float
    colsample_bytree: float
    scale_pos_weight: Literal["auto"] | Annotated[float, Field(gt=0)]


class ModelConfig(_Section):
    xgboost: XGBoostConfig


class DecisionConfig(_Section):
    cost_missed_failure_eur: Annotated[float, Field(gt=0)]
    cost_false_alarm_eur: Annotated[float, Field(gt=0)]
    min_precision: Annotated[float, Field(ge=0, le=1)]


class MlflowConfig(_Section):
    tracking_uri: str
    experiment_name: Annotated[str, Field(pattern=r"^[a-z0-9-]+$", min_length=3, max_length=64)]


class Config(_Section):
    project: ProjectConfig
    paths: PathsConfig
    columns: ColumnsConfig
    sensor_bounds: dict[str, SensorBound]
    split: SplitConfig
    model: ModelConfig
    decision: DecisionConfig
    mlflow: MlflowConfig


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> Config:
    """Lit et valide le fichier YAML de configuration.

    Raises:
        FileNotFoundError: le fichier n'existe pas.
        ValueError: le YAML est syntaxiquement invalide ou n'est pas un mapping.
        pydantic.ValidationError: le contenu ne respecte pas le schéma ou les règles métier.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Fichier de configuration introuvable : {path.resolve()}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"YAML invalide dans {path} : {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError(
            f"{path} doit contenir un mapping YAML à la racine, reçu : {type(raw).__name__}"
        )

    try:
        return Config.model_validate(raw)
    except ValidationError as exc:
        exc.add_note(f"Fichier de configuration : {path.resolve()}")
        raise
