"""Tests de l'interface technicien (AppTest) avec une API simulée (httpx.MockTransport)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import app_streamlit as ui

APP_PATH = str(Path(__file__).resolve().parents[1] / "app_streamlit.py")
Handler = Callable[[httpx.Request], httpx.Response]
NBSP, THIN, MINUS = " ", " ", "−"
FORBIDDEN_WORDS = ("shap", "probabilit", "log-odds", "feature")
PREVENTIVE_WEAR_FROM_API = 180  # volontairement ≠ 200 : prouve que la valeur vient de l'API

FACTORS = [
    {
        "feature": "tool_wear_min",
        "value": 215.0,
        "contribution": 2.1,
        "direction": "vers_panne",
        "unit": "min",
    },
    {
        "feature": "power_w",
        "value": 5681.2,
        "contribution": -0.8,
        "direction": "vers_normal",
        "unit": "W",
    },
    {
        "feature": "product_type",
        "value": "M",
        "contribution": 0.1,
        "direction": "vers_panne",
        "unit": None,
    },
]


def _prediction(decision: str, risk: float) -> dict[str, Any]:
    """Réponse /predict au format de l'API."""
    actions = {
        "OPTIMAL": None,
        "MAINTENANCE_PREVENTIVE": "Planifier le remplacement de l'outil.",
        "CRITIQUE": "Remplacer l'outil : usure élevée.",
    }
    return {
        "request_id": "abc",
        "timestamp": "2026-11-09T12:00:00Z",
        "decision": decision,
        "failure_probability": risk,
        "threshold": 0.4126,
        "triggered_by": "model",
        "top_factors": FACTORS,
        "recommended_action": actions[decision],
        "derived": {"temp_diff_k": 10.04, "power_w": 5681.2, "strain_min_nm": 13650.0},
        "model_version": {
            "model_sha256": "84b0026a" + "0" * 56,
            "thresholds_sha256": "1" * 64,
            "git_commit": "abcdef12",
        },
    }


TEST_RESULTS = [
    {
        "approach": "B0 aucune alerte",
        "detected": 0,
        "missed": 51,
        "false_alarms": 0,
        "total_cost_eur": 153000.0,
    },
    {
        "approach": "B1 preventive",
        "detected": 19,
        "missed": 32,
        "false_alarms": 119,
        "total_cost_eur": 113850.0,
    },
    {
        "approach": "Hybride XGBoost + preventif",
        "detected": 49,
        "missed": 2,
        "false_alarms": 120,
        "total_cost_eur": 24000.0,
    },
]
MODEL_INFO = {
    "primary_approach": "hybrid",
    "secondary_approach": "xgboost",
    "threshold": 0.4126,
    "thresholds": {"hybrid": {"threshold": 0.4126}},
    "preventive_tool_wear_min": PREVENTIVE_WEAR_FROM_API,
    "cost_missed_failure_eur": 3000,
    "cost_false_alarm_eur": 150,
    "min_precision": 0.5,
    "test_results": TEST_RESULTS,
    "model_version": {"model_sha256": "8" * 64, "thresholds_sha256": "1" * 64, "git_commit": "x"},
}


def fake_api(seen: list[dict[str, Any]] | None = None) -> Handler:
    """CRITIQUE si couple >= 60 Nm ; sinon PRÉVENTIVE si usure >= 200 min ; sinon OPTIMAL."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/predict":
            body = json.loads(request.content)
            if seen is not None:
                seen.append(body)
            if body["torque_nm"] >= 60:
                return httpx.Response(200, json=_prediction("CRITIQUE", 0.9996))
            if body["tool_wear_min"] >= 200:
                return httpx.Response(200, json=_prediction("MAINTENANCE_PREVENTIVE", 0.16))
            return httpx.Response(200, json=_prediction("OPTIMAL", 0.02))
        if request.url.path == "/model/info":
            return httpx.Response(200, json=MODEL_INFO)
        return httpx.Response(404, json={"detail": "Not Found"})

    return handler


@pytest.fixture(autouse=True)
def clear_streamlit_cache() -> None:
    """Le cache st.cache_data vit dans le processus : on le vide entre deux tests."""
    st.cache_data.clear()


@pytest.fixture
def use_api(monkeypatch: pytest.MonkeyPatch) -> Callable[[Handler], None]:
    """Remplace httpx.Client par un client branché sur un handler simulé."""
    real_client = httpx.Client

    def install(handler: Handler) -> None:
        def client(**kwargs: Any) -> httpx.Client:
            return real_client(transport=httpx.MockTransport(handler), **kwargs)

        monkeypatch.setattr(httpx, "Client", client)

    return install


def _run() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    return at


def _click(at: AppTest, label: str) -> None:
    next(b for b in at.button if b.label == label).click().run()


def _visible_text(at: AppTest) -> str:
    """Tout le texte affiché (markdown, légendes, indicateurs, boutons, onglets)."""
    parts = [h.value for h in at.header] + [h.value for h in at.subheader]
    parts += [m.value for m in at.markdown] + [c.value for c in at.caption]
    parts += [f"{m.label} {m.value}" for m in at.metric]
    parts += [b.label for b in at.button] + [t.label for t in at.tabs]
    return "\n".join(parts)


# --- Structure --------------------------------------------------------------------------


def test_app_renders(use_api: Callable[[Handler], None]) -> None:
    use_api(fake_api())
    at = _run()

    assert not at.exception
    text = _visible_text(at)
    assert "Surveillance machine — risque de panne" in text
    assert "le verdict se met à jour automatiquement" in text
    assert [t.label for t in at.tabs] == ["Machine", "Lot", "Le modèle"]
    subheaders = [h.value for h in at.subheader]
    for group in ("Scénarios", "Thermique", "Mécanique", "Outil", "Pourquoi ce verdict"):
        assert group in subheaders
    assert at.header[0].value == "🛠️ Surveillance machine — risque de panne"
    assert "Scénario : Machine saine" in text


# --- Les trois états --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("scenario", "decision", "has_action"),
    [
        ("Machine saine", "OPTIMAL", False),
        ("Usure ≥ 200 min", "MAINTENANCE_PREVENTIVE", True),
        ("Surcharge outil", "CRITIQUE", True),
    ],
)
def test_each_state_has_label_sentence_and_action(
    use_api: Callable[[Handler], None], scenario: str, decision: str, has_action: bool
) -> None:
    use_api(fake_api())
    at = _run()
    _click(at, scenario)

    assert not at.exception
    text = _visible_text(at)
    state = ui.STATES[decision]
    assert state["label"] in text and state["icon"] in text  # icône + libellé, pas la couleur seule
    assert state["sentence"] in text
    assert ("À faire :" in text) is has_action
    assert f"Scénario : {scenario}" in text


def test_slider_change_switches_to_custom(use_api: Callable[[Handler], None]) -> None:
    use_api(fake_api())
    at = _run()
    _click(at, "Surcharge outil")

    at.slider(key="torque_nm").set_value(30.0).run()

    assert "Scénario : Personnalisé" in _visible_text(at)


def test_risk_gauge_text(use_api: Callable[[Handler], None]) -> None:
    use_api(fake_api())
    at = _run()
    _click(at, "Usure ≥ 200 min")

    assert f"Risque de panne : 16{NBSP}%" in _visible_text(at)


# --- Format et vocabulaire --------------------------------------------------------------


def test_celsius_conversion_and_french_format() -> None:
    assert ui.format_temperature(298.0) == f"298,0{NBSP}K (24,9{NBSP}°C)"
    assert ui.format_temperature(310.0) == f"310,0{NBSP}K (36,9{NBSP}°C)"
    assert ui.fmt_number(5681.2) == f"5{THIN}681"
    assert ui.fmt_number(1234.56, 1) == f"1{THIN}234,6"
    assert ui.fmt_number(153000) == f"153{THIN}000"
    assert ui.fmt_pct(0.163) == f"16{NBSP}%"
    assert ui.fmt_smart(-10.0) == f"{MINUS}10"
    assert ui.fmt_smart(2.5) == "2,5"


def test_temperatures_shown_in_kelvin_and_celsius(use_api: Callable[[Handler], None]) -> None:
    seen: list[dict[str, Any]] = []
    use_api(fake_api(seen))
    at = _run()
    _click(at, "Usure ≥ 200 min")  # air 298,0 K

    assert f"Température air : **298,0{NBSP}K (24,9{NBSP}°C)**" in _visible_text(at)
    assert seen[-1]["air_temperature_k"] == 298.0  # l'API reçoit toujours des kelvins


def test_causes_are_sentences_without_jargon(use_api: Callable[[Handler], None]) -> None:
    use_api(fake_api())
    at = _run()
    _click(at, "Surcharge outil")

    text = _visible_text(at)
    assert f"↑ L'usure de l'outil (215{NBSP}min) augmente le risque." in text
    assert f"↓ La puissance (5{THIN}681{NBSP}W) réduit le risque." in text
    lowered = text.lower()
    for word in FORBIDDEN_WORDS:
        assert word not in lowered, f"jargon affiché : {word}"


def test_technical_details_are_formatted(use_api: Callable[[Handler], None]) -> None:
    use_api(fake_api())
    at = _run()

    text = _visible_text(at)
    assert f"Écart thermique 10,0{NBSP}K" in text  # 1 décimale
    assert f"Puissance 5{THIN}681{NBSP}W" in text  # à l'unité
    assert f"Contrainte d'usure 13{THIN}650{NBSP}min·Nm" in text
    assert at.expander[0].label == "Détails techniques"


# --- Seuil préventif lu dans l'API ------------------------------------------------------


def test_preventive_hint_comes_from_model_info(use_api: Callable[[Handler], None]) -> None:
    use_api(fake_api())
    at = _run()

    text = _visible_text(at)
    assert f"Remplacement préventif à partir de {PREVENTIVE_WEAR_FROM_API}{NBSP}min" in text
    assert "à partir de 200" not in text


# --- Erreurs ----------------------------------------------------------------------------


def test_api_unavailable_shows_clear_message(use_api: Callable[[Handler], None]) -> None:
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    use_api(down)
    at = _run()

    assert not at.exception  # aucune trace Python à l'écran
    text = _visible_text(at)
    assert "Service d'analyse indisponible" in text
    assert "Vérifiez qu'il est démarré" in text
    assert "Remplacement préventif" not in text  # pas de repère inventé sans l'API


def test_rejected_measure_is_rephrased(use_api: Callable[[Handler], None]) -> None:
    body = {"detail": {"message": "torque_nm=-10.0 hors bornes [0.0, 120.0]"}}
    use_api(lambda request: httpx.Response(422, json=body))
    at = _run()

    assert not at.exception
    text = _visible_text(at)
    assert "Mesure refusée" in text
    assert (
        f"Couple : {MINUS}10{NBSP}Nm est hors de la plage plausible "
        f"(0 à 120{NBSP}Nm). Vérifiez le capteur."
    ) in text


def test_model_not_ready(use_api: Callable[[Handler], None]) -> None:
    use_api(lambda request: httpx.Response(503, json={"detail": "Modèle non chargé"}))
    at = _run()

    assert not at.exception
    text = _visible_text(at)
    assert "Le modèle n'est pas chargé" in text
    assert "Réessayez dans quelques instants" in text


def test_rephrase_unknown_message_is_kept() -> None:
    assert ui.rephrase_rejection("message inconnu") == "message inconnu"
    assert "Type de produit « X » inconnu" in ui.rephrase_rejection(
        "product_type='X' non autorisé (valeurs admises : ['L', 'M', 'H'])"
    )


# --- Onglet Le modèle -------------------------------------------------------------------


def test_model_tab_explains_and_reports_reliability(use_api: Callable[[Handler], None]) -> None:
    use_api(fake_api())
    at = _run()

    text = _visible_text(at)
    assert {"Comment lire le résultat", "Fiabilité mesurée", "Version"} <= {
        h.value for h in at.subheader
    }
    for state in ui.STATES.values():
        assert state["sentence"] in text
    assert "49 pannes sur 51 anticipées." in text
    assert (
        "1 seule fausse alerte urgente, et 119 remplacements d'outils planifiés "
        "au titre du préventif."
    ) in text
    assert f"24{THIN}000{NBSP}€" in text and f"153{THIN}000{NBSP}€" in text
    assert "Coûts estimés" in text
    assert "données synthétiques (AI4I 2020)" in text


# --- Cache ------------------------------------------------------------------------------


def test_model_info_is_cached_but_predict_is_not(use_api: Callable[[Handler], None]) -> None:
    calls: list[str] = []
    handler = fake_api()

    def counting(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return handler(request)

    use_api(counting)
    at = _run()
    at.slider(key="torque_nm").set_value(30.0).run()
    at.run()

    assert not at.exception
    assert calls.count("/model/info") == 1  # servi par le cache aux rendus suivants
    assert calls.count("/predict") == 3  # appelé à chaque rendu


# --- Lot (fonctions pures) --------------------------------------------------------------

CSV_HEADER = ",".join(ui.FIELDS)


def test_example_csv_is_valid() -> None:
    items = ui.parse_batch_csv(ui.example_csv())
    assert len(items) == 5
    assert {item["product_type"] for item in items} == {"L", "M", "H"}


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"product_type,torque_nm\nL,40\n", "Colonnes manquantes"),
        (f"{CSV_HEADER}\n".encode(), "aucune ligne"),
        (f"{CSV_HEADER}\nL,300,,1400,65,210\n".encode(), "Valeurs manquantes"),
    ],
)
def test_parse_batch_csv_errors(content: bytes, message: str) -> None:
    with pytest.raises(ui.MeasureRejected, match=message):
        ui.parse_batch_csv(content)


def test_parse_batch_csv_too_many_rows() -> None:
    rows = "\n".join(["M,298.9,309.1,1500,40,50"] * (ui.MAX_BATCH + 1))
    with pytest.raises(ui.MeasureRejected, match=f"maximum {ui.MAX_BATCH}"):
        ui.parse_batch_csv(f"{CSV_HEADER}\n{rows}\n".encode())


def test_batch_results_sorted_to_handle_first() -> None:
    items = [{"id": "a"}, {"id": "b"}, {"id": "c"}, {"id": "d"}]
    predictions = [
        _prediction("OPTIMAL", 0.01),
        _prediction("CRITIQUE", 0.70),
        _prediction("MAINTENANCE_PREVENTIVE", 0.20),
        _prediction("CRITIQUE", 0.95),
    ]

    result = ui.batch_results(items, predictions)

    assert result["id"].tolist() == ["d", "b", "c", "a"]  # CRITIQUE (risque ↓), PRÉVENTIVE, OPTIMAL
    assert result["main_cause"].iloc[0] == f"Usure de l'outil (215{NBSP}min)"
    assert result["recommended_action"].iloc[-1] == ""  # OPTIMAL : pas d'action


# --- Risque affiché ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("risk", "expected"),
    [
        (0.0, f"<{NBSP}1{NBSP}%"),
        (0.004, f"<{NBSP}1{NBSP}%"),
        (0.01, f"1{NBSP}%"),
        (0.163, f"16{NBSP}%"),
        (0.99, f"99{NBSP}%"),
        (0.9996, f">{NBSP}99{NBSP}%"),
        (1.0, f">{NBSP}99{NBSP}%"),
    ],
)
def test_fmt_risk_bounds(risk: float, expected: str) -> None:
    assert ui.fmt_risk(risk) == expected


def test_extreme_risk_never_shows_100(use_api: Callable[[Handler], None]) -> None:
    use_api(fake_api())
    at = _run()
    _click(at, "Surcharge outil")  # risque simulé 0,9996

    text = _visible_text(at)
    assert f"Risque de panne : >{NBSP}99{NBSP}%" in text
    assert f"100{NBSP}%" not in text


# --- Graphique divergent des causes -----------------------------------------------------


def _bars_data(chart: Any) -> list[dict[str, Any]]:
    spec = chart.to_dict()
    bars = spec["layer"][0]
    rows: list[dict[str, Any]] = spec["datasets"][bars["data"]["name"]]
    return rows


def test_causes_chart_diverges_around_zero() -> None:
    chart = ui.causes_chart(FACTORS)
    spec = chart.to_dict()
    rows = {r["cause"]: r for r in _bars_data(chart)}

    # Signe → côté : vers la panne à droite (> 0), vers le normal à gauche (< 0).
    assert rows[f"Usure de l'outil (215{NBSP}min)"]["poids"] > 0
    assert rows[f"Puissance (5{THIN}681{NBSP}W)"]["poids"] < 0
    # Axe central à 0 : échelle symétrique et trait vertical en x = 0.
    bars, axis, left, right = spec["layer"]
    low, high = bars["encoding"]["x"]["scale"]["domain"]
    assert low == -high
    assert bars["encoding"]["x"]["axis"] is None  # aucune graduation numérique
    assert axis["mark"]["type"] == "rule"
    assert spec["datasets"][axis["data"]["name"]] == [{"x": 0.0}]
    # Légende alignée sur l'axe : « ← vers le normal » à gauche, « vers la panne → » à droite.
    assert spec["datasets"][left["data"]["name"]][0]["text"] == "← vers le normal"
    assert left["mark"]["align"] == "right"
    assert spec["datasets"][right["data"]["name"]][0]["text"] == "vers la panne →"
    assert right["mark"]["align"] == "left"
    # Couleurs : rouge pour la panne, vert pour le normal.
    color = bars["encoding"]["color"]["scale"]
    assert dict(zip(color["domain"], color["range"], strict=True)) == {
        "vers_panne": ui.STATES["CRITIQUE"]["color"],
        "vers_normal": ui.STATES["OPTIMAL"]["color"],
    }


def test_side_follows_direction_not_only_sign() -> None:
    # Même si le signe reçu était incohérent, le côté suit la direction annoncée par l'API.
    odd = [{**FACTORS[0], "contribution": -2.1}, {**FACTORS[1], "contribution": 0.8}]
    rows = _bars_data(ui.causes_chart(odd))
    assert rows[0]["poids"] > 0 and rows[0]["sens"] == "vers_panne"
    assert rows[1]["poids"] < 0 and rows[1]["sens"] == "vers_normal"


# --- Fiabilité : valeurs calculées ------------------------------------------------------


def test_reliability_values_are_computed() -> None:
    results = [
        {"approach": "B0 aucune alerte", "detected": 0, "missed": 30, "false_alarms": 0,
         "total_cost_eur": 90000.0},
        {"approach": "B1 preventive", "detected": 10, "missed": 20, "false_alarms": 100,
         "total_cost_eur": 75000.0},
        {"approach": "Hybride XGBoost + preventif", "detected": 28, "missed": 2,
         "false_alarms": 104, "total_cost_eur": 21600.0},
    ]  # fmt: skip

    text = " ".join(ui.reliability_lines(results))

    assert "28 pannes sur 30 anticipées." in text
    assert (
        "4 fausses alertes urgentes, et 100 remplacements d'outils planifiés au titre du préventif."
    ) in text
    assert f"21{THIN}600{NBSP}€" in text and f"90{THIN}000{NBSP}€" in text


def test_reliability_without_urgent_false_alarm() -> None:
    results = [
        {"approach": "B1 preventive", "detected": 1, "missed": 0, "false_alarms": 1,
         "total_cost_eur": 150.0},
        {"approach": "Hybride XGBoost + preventif", "detected": 1, "missed": 0,
         "false_alarms": 1, "total_cost_eur": 150.0},
    ]  # fmt: skip
    text = " ".join(ui.reliability_lines(results))
    assert "Aucune fausse alerte urgente, et 1 remplacement d'outil planifié" in text
