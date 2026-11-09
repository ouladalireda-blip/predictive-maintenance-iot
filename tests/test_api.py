"""Tests de l'API (FastAPI TestClient, artefacts construits dans tmp_path)."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from predictive_maintenance_iot.api.main import create_app
from predictive_maintenance_iot.config import Config
from tests.test_predict import HIGH_RISK, LOW_RISK, build_artifacts

MAX_BATCH = 3


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return build_artifacts(tmp_path, max_batch_size=MAX_BATCH)


@pytest.fixture
def client(cfg: Config) -> Iterator[TestClient]:
    with TestClient(create_app(lambda: cfg)) as test_client:  # déclenche le lifespan
        yield test_client


def _log_rows(cfg: Config) -> list[tuple[Any, ...]]:
    with sqlite3.connect(cfg.api.inference_log_path) as conn:
        return conn.execute(
            "SELECT request_id, inputs, probability, decision, model_sha256 FROM inference_log"
        ).fetchall()


# --- Santé et informations --------------------------------------------------------------


def test_health_ok(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert len(body["model_version"]["model_sha256"]) == 64


def test_model_info(cfg: Config, client: TestClient) -> None:
    body = client.get("/model/info").json()
    assert body["primary_approach"] == "hybrid"
    assert body["preventive_tool_wear_min"] == 200
    assert body["cost_missed_failure_eur"] == 3000
    assert body["test_results"] is None  # examen final pas encore passé

    results = [{"approach": "Hybride XGBoost + preventif", "total_cost_eur": 1234}]
    (cfg.paths.artifacts_dir / "test_results.json").write_text(
        json.dumps({"results_at_config_ratio": results}), encoding="utf-8"
    )
    assert client.get("/model/info").json()["test_results"] == results


def test_openapi_has_realistic_example(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()["components"]["schemas"]["PredictRequest"]
    assert schema["examples"][0]["product_type"] == "M"


# --- /predict ---------------------------------------------------------------------------


def test_predict_returns_all_fields(client: TestClient) -> None:
    response = client.post("/predict", json=HIGH_RISK)

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "request_id",
        "timestamp",
        "decision",
        "failure_probability",
        "threshold",
        "triggered_by",
        "top_factors",
        "recommended_action",
        "derived",
        "model_version",
    }
    assert body["derived"]["temp_diff_k"] == 10.0
    assert body["derived"]["strain_min_nm"] == 50.0 * 115.0
    assert set(body["derived"]) == {"temp_diff_k", "power_w", "strain_min_nm"}
    assert body["decision"] == "CRITIQUE"
    assert body["recommended_action"]
    assert body["failure_probability"] >= body["threshold"]
    assert body["timestamp"].endswith("Z") or body["timestamp"].endswith("+00:00")
    assert {"feature", "value", "contribution", "direction", "unit"} == set(body["top_factors"][0])


def test_preventive_decision(client: TestClient) -> None:
    body = client.post("/predict", json={**LOW_RISK, "tool_wear_min": 250}).json()
    assert body["decision"] == "MAINTENANCE_PREVENTIVE"
    assert body["triggered_by"] == "preventive"


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ({"torque_nm": -5.0}, "torque_nm"),  # hors bornes physiques
        ({"process_temperature_k": 500.0}, "process_temperature_k"),
        ({"product_type": "X"}, "product_type"),  # type inconnu
    ],
)
def test_invalid_measure_returns_clear_422(
    client: TestClient, override: dict[str, Any], expected: str
) -> None:
    response = client.post("/predict", json={**HIGH_RISK, **override})

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert expected in detail["message"]
    assert "Value error" not in detail["message"]
    assert all(not e["msg"].startswith("Value error, ") for e in detail["errors"])


def test_extra_field_returns_422(client: TestClient) -> None:
    response = client.post("/predict", json={**HIGH_RISK, "humidity": 0.4})
    assert response.status_code == 422
    assert "humidity" in json.dumps(response.json())


def test_wrong_type_returns_422(client: TestClient) -> None:
    response = client.post("/predict", json={**HIGH_RISK, "torque_nm": "beaucoup"})
    assert response.status_code == 422


# --- /predict/batch ---------------------------------------------------------------------


def test_batch_ok(client: TestClient) -> None:
    response = client.post("/predict/batch", json={"items": [HIGH_RISK, LOW_RISK]})

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert [p["decision"] for p in body["predictions"]] == ["CRITIQUE", "OPTIMAL"]


def test_batch_too_large_returns_413(client: TestClient) -> None:
    response = client.post("/predict/batch", json={"items": [LOW_RISK] * (MAX_BATCH + 1)})
    assert response.status_code == 413


def test_batch_invalid_item_reports_index(client: TestClient) -> None:
    items = [LOW_RISK, {**LOW_RISK, "torque_nm": -1.0}]
    response = client.post("/predict/batch", json={"items": items})

    assert response.status_code == 422
    assert response.json()["detail"]["index"] == 1


# --- Journal des inférences -------------------------------------------------------------


def test_one_log_row_per_prediction(cfg: Config, client: TestClient) -> None:
    single = client.post("/predict", json=HIGH_RISK).json()
    batch = client.post("/predict/batch", json={"items": [LOW_RISK, HIGH_RISK]}).json()
    client.post("/predict", json={**HIGH_RISK, "torque_nm": -5.0})  # 422 : pas journalisé

    rows = _log_rows(cfg)
    assert len(rows) == 3
    ids = {row[0] for row in rows}
    assert ids == {single["request_id"], *(p["request_id"] for p in batch["predictions"])}
    first = next(row for row in rows if row[0] == single["request_id"])
    assert json.loads(first[1]) == HIGH_RISK
    assert first[2] == pytest.approx(single["failure_probability"])
    assert first[3] == "CRITIQUE"
    assert first[4] == single["model_version"]["model_sha256"]


def test_log_failure_never_breaks_prediction(tmp_path: Path) -> None:
    cfg = build_artifacts(tmp_path)
    blocked = tmp_path / "pas_un_fichier"
    blocked.mkdir()  # un dossier à la place de la base : SQLite ne peut pas l'ouvrir
    cfg = cfg.model_copy(update={"api": cfg.api.model_copy(update={"inference_log_path": blocked})})
    with TestClient(create_app(lambda: cfg)) as client:
        response = client.post("/predict", json=HIGH_RISK)
    assert response.status_code == 200


# --- Modèle indisponible ----------------------------------------------------------------


def test_fingerprint_mismatch_returns_503(cfg: Config) -> None:
    model = cfg.paths.artifacts_dir / "xgboost.joblib"
    model.write_bytes(model.read_bytes() + b"\0")

    with TestClient(create_app(lambda: cfg)) as client:
        health = client.get("/health")
        predict = client.post("/predict", json=HIGH_RISK)
        info = client.get("/model/info")

    assert health.status_code == 503
    assert health.json()["model_loaded"] is False
    assert "FingerprintMismatchError" in health.json()["detail"]
    assert predict.status_code == 503
    assert info.status_code == 503


def test_422_message_is_clean_for_out_of_bounds(client: TestClient) -> None:
    message = client.post("/predict", json={**HIGH_RISK, "torque_nm": -5.0}).json()["detail"][
        "message"
    ]
    assert message.startswith("torque_nm=-5.0 hors bornes")


def test_product_type_label_in_api_response(client: TestClient) -> None:
    for product_type in ("L", "M", "H"):
        body = client.post("/predict", json={**HIGH_RISK, "product_type": product_type}).json()
        for factor in body["top_factors"]:
            if factor["feature"] == "product_type":
                assert factor["value"] == product_type
            else:
                assert isinstance(factor["value"], float)


def test_root_redirects_to_docs(client: TestClient) -> None:
    response = client.get("/", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "/docs"
    assert "/" not in client.get("/openapi.json").json()["paths"]
