"""API d'inférence (FastAPI).

Le modèle est chargé UNE fois au démarrage (lifespan). S'il ne peut pas l'être (empreinte qui ne
correspond pas, seuils non figés, fichier absent...), l'API démarre quand même pour l'exposer :
/health renvoie 503 avec la cause, et les routes de prédiction renvoient 503.

Chaque prédiction est journalisée dans SQLite ; un échec du journal n'échoue jamais la réponse.

Lancement : uv run uvicorn predictive_maintenance_iot.api.main:app
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import ValidationError

from predictive_maintenance_iot.api.schemas import (
    BatchRequest,
    BatchResponse,
    DerivedSchema,
    HealthResponse,
    ModelInfoResponse,
    ModelVersionSchema,
    PredictRequest,
    PredictResponse,
    TopFactorSchema,
)
from predictive_maintenance_iot.config import Config, load_config
from predictive_maintenance_iot.data.validation import SensorPayload, validate_payload
from predictive_maintenance_iot.models.predict import Prediction, Predictor

logger = logging.getLogger(__name__)

TEST_RESULTS_FILENAME = "test_results.json"


# --- Journal des inférences -------------------------------------------------------------


class InferenceLog:
    """Journal SQLite des prédictions. Toute erreur est journalisée, jamais propagée."""

    def __init__(self, path: Path) -> None:
        self.path = path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(path) as conn:
                conn.execute(
                    """CREATE TABLE IF NOT EXISTS inference_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        request_id TEXT NOT NULL,
                        inputs TEXT NOT NULL,
                        probability REAL NOT NULL,
                        decision TEXT NOT NULL,
                        model_sha256 TEXT NOT NULL
                    )"""
                )
        except Exception:
            logger.exception("Journal d'inférence indisponible (%s)", path)

    def write(self, rows: Sequence[tuple[str, str, str, float, str, str]]) -> None:
        try:
            with sqlite3.connect(self.path) as conn:
                conn.executemany(
                    "INSERT INTO inference_log (timestamp, request_id, inputs, probability, "
                    "decision, model_sha256) VALUES (?, ?, ?, ?, ?, ?)",
                    rows,
                )
        except Exception:
            logger.exception("Échec d'écriture dans le journal d'inférence (%s)", self.path)


# --- État de l'application --------------------------------------------------------------


@dataclass
class ApiState:
    cfg: Config | None = None
    predictor: Predictor | None = None
    inference_log: InferenceLog | None = None
    load_error: str | None = None


def get_state(request: Request) -> ApiState:
    state: ApiState = request.app.state.api
    return state


StateDep = Annotated[ApiState, Depends(get_state)]


def _ready(state: ApiState) -> tuple[Config, Predictor]:
    if state.cfg is None or state.predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Modèle non chargé : {state.load_error or 'cause inconnue'}",
        )
    return state.cfg, state.predictor


_PYDANTIC_VALUE_ERROR_PREFIX = "Value error, "


def _clean_message(message: str) -> str:
    """Retire le préfixe « Value error, » que Pydantic ajoute aux ValueError des validateurs."""
    return message.removeprefix(_PYDANTIC_VALUE_ERROR_PREFIX)


def _validate(request: PredictRequest, cfg: Config, index: int | None = None) -> SensorPayload:
    """Bornes physiques et types produit de la config → 422 explicite en cas d'échec."""
    try:
        return validate_payload(request.model_dump(), cfg)
    except ValidationError as exc:
        errors = exc.errors(include_url=False, include_context=False, include_input=False)
        for error in errors:
            error["msg"] = _clean_message(str(error["msg"]))
        where = "" if index is None else f"élément {index} : "
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "message": where + "; ".join(str(e["msg"]) for e in errors),
                "index": index,
                "errors": errors,
            },
        ) from exc


def _respond(
    state: ApiState,
    predictor: Predictor,
    requests: Sequence[PredictRequest],
    predictions: Sequence[Prediction],
) -> list[PredictResponse]:
    """Construit les réponses et journalise une ligne par prédiction."""
    version = ModelVersionSchema(**asdict(predictor.version))
    responses, rows = [], []
    for request, prediction in zip(requests, predictions, strict=True):
        response = PredictResponse(
            request_id=uuid.uuid4().hex,
            timestamp=datetime.now(UTC),
            decision=prediction.decision.value,
            failure_probability=prediction.failure_probability,
            threshold=prediction.threshold,
            triggered_by=prediction.triggered_by.value,
            top_factors=[TopFactorSchema(**asdict(f)) for f in prediction.top_factors],
            recommended_action=prediction.recommended_action,
            derived=DerivedSchema(**prediction.derived),
            model_version=version,
        )
        responses.append(response)
        rows.append(
            (
                response.timestamp.isoformat(),
                response.request_id,
                json.dumps(request.model_dump()),
                response.failure_probability,
                response.decision,
                version.model_sha256,
            )
        )
    if state.inference_log is not None:
        state.inference_log.write(rows)
    return responses


# --- Application ------------------------------------------------------------------------


def create_app(config_loader: Callable[[], Config] = load_config) -> FastAPI:
    """Fabrique l'application ; `config_loader` est injectable (tests)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        state = ApiState()
        try:
            state.cfg = config_loader()
            state.inference_log = InferenceLog(state.cfg.api.inference_log_path)
            state.predictor = Predictor.load(state.cfg)
        except Exception as exc:
            state.load_error = f"{type(exc).__name__}: {exc}"
            logger.exception("Démarrage sans modèle : %s", state.load_error)
        app.state.api = state
        yield

    app = FastAPI(
        title="Maintenance prédictive AI4I",
        description="Décision de maintenance (CRITIQUE / MAINTENANCE_PREVENTIVE / OPTIMAL) "
        "à partir d'une mesure capteur, avec le modèle et les seuils figés.",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/docs", status_code=status.HTTP_307_TEMPORARY_REDIRECT)

    @app.get("/health", response_model=HealthResponse)
    def health(state: StateDep) -> Any:
        if state.predictor is None:
            body = HealthResponse(status="unavailable", model_loaded=False, detail=state.load_error)
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content=body.model_dump()
            )
        return HealthResponse(
            status="ok",
            model_loaded=True,
            model_version=ModelVersionSchema(**asdict(state.predictor.version)),
        )

    @app.get("/model/info", response_model=ModelInfoResponse)
    def model_info(state: StateDep) -> ModelInfoResponse:
        cfg, predictor = _ready(state)
        frozen = predictor.frozen
        results_path = cfg.paths.artifacts_dir / TEST_RESULTS_FILENAME
        test_results = None
        if results_path.is_file():
            payload = json.loads(results_path.read_text(encoding="utf-8"))
            test_results = payload.get("results_at_config_ratio")
        return ModelInfoResponse(
            primary_approach=str(frozen["primary_approach"]),
            secondary_approach=frozen.get("secondary_approach"),
            threshold=predictor.threshold,
            thresholds=frozen["thresholds"],
            preventive_tool_wear_min=predictor.preventive_wear_min,
            cost_missed_failure_eur=float(frozen["cost_missed_failure_eur"]),
            cost_false_alarm_eur=float(frozen["cost_false_alarm_eur"]),
            min_precision=frozen.get("min_precision"),
            test_results=test_results,
            model_version=ModelVersionSchema(**asdict(predictor.version)),
        )

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest, state: StateDep) -> PredictResponse:
        cfg, predictor = _ready(state)
        payload = _validate(request, cfg)
        return _respond(state, predictor, [request], [predictor.predict(payload)])[0]

    @app.post("/predict/batch", response_model=BatchResponse)
    def predict_batch(request: BatchRequest, state: StateDep) -> BatchResponse:
        cfg, predictor = _ready(state)
        limit = cfg.api.max_batch_size
        if len(request.items) > limit:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=f"{len(request.items)} mesures reçues, maximum {limit} par requête",
            )
        payloads = [_validate(item, cfg, index=i) for i, item in enumerate(request.items)]
        predictions = predictor.predict_batch(payloads)
        responses = _respond(state, predictor, request.items, predictions)
        return BatchResponse(count=len(responses), predictions=responses)

    return app


app = create_app()
