"""Interface technicien (Streamlit) : client de l'API de maintenance prédictive.

L'interface ne prédit et ne recalcule RIEN : verdict, risque, seuil, causes et variables
physiques viennent de l'API. Seules des conversions d'affichage sont faites ici
(kelvins → degrés Celsius, format français des nombres).

URL de l'API : variable d'environnement API_URL (défaut http://127.0.0.1:8000).
Lancement : uv run streamlit run app_streamlit.py
"""

from __future__ import annotations

import html
import io
import os
import re
from typing import Any

import altair as alt
import httpx
import pandas as pd
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")
MAX_BATCH = int(os.environ.get("UI_MAX_BATCH", "100"))
TIMEOUT_S = 15.0

FIELDS = (
    "product_type",
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
)
PRODUCT_TYPES = ("L", "M", "H")
CUSTOM = "Personnalisé"

# --- Palette et états (couleur + icône + libellé, jamais la couleur seule) --------------

NEUTRAL = "#5F6368"
NEUTRAL_BG = "#F1F3F4"
STATES: dict[str, dict[str, str]] = {
    "OPTIMAL": {
        "icon": "🟢",
        "label": "OPTIMAL",
        "color": "#1E8E3E",
        "background": "#E6F4EA",
        "sentence": "Aucune panne attendue. Aucune action requise.",
    },
    "MAINTENANCE_PREVENTIVE": {
        "icon": "🟠",
        "label": "MAINTENANCE PRÉVENTIVE",
        "color": "#E37400",
        "background": "#FEF3E2",
        "sentence": "Pas de risque immédiat. Outil à remplacer au prochain arrêt programmé.",
    },
    "CRITIQUE": {
        "icon": "🔴",
        "label": "CRITIQUE",
        "color": "#C5221F",
        "background": "#FCE8E6",
        "sentence": "Risque de panne élevé. Intervenir maintenant.",
    },
}
STATE_ORDER = ("CRITIQUE", "MAINTENANCE_PREVENTIVE", "OPTIMAL")  # « à traiter d'abord »

# --- Mesures : (libellé, unité, min, max, pas, décimales) -------------------------------

MEASURES: dict[str, tuple[str, str, float, float, float, int]] = {
    "air_temperature_k": ("Température air", "K", 290.0, 310.0, 0.1, 1),
    "process_temperature_k": ("Température procédé", "K", 300.0, 320.0, 0.1, 1),
    "rotational_speed_rpm": ("Vitesse de rotation", "rpm", 1000, 3000, 10, 0),
    "torque_nm": ("Couple", "Nm", 0.0, 100.0, 0.1, 1),
    "tool_wear_min": ("Usure de l'outil", "min", 0, 300, 1, 0),
}
GROUPS: dict[str, tuple[str, ...]] = {
    "Thermique": ("air_temperature_k", "process_temperature_k"),
    "Mécanique": ("rotational_speed_rpm", "torque_nm"),
    "Outil": ("product_type", "tool_wear_min"),
}

# Causes : (sujet de phrase, libellé court, décimales d'affichage).
CAUSES: dict[str, tuple[str, str, int]] = {
    "product_type": ("Le type de produit", "Type de produit", 0),
    "air_temperature_k": ("La température de l'air", "Température air", 1),
    "process_temperature_k": ("La température du procédé", "Température procédé", 1),
    "rotational_speed_rpm": ("La vitesse de rotation", "Vitesse de rotation", 0),
    "torque_nm": ("Le couple", "Couple", 1),
    "tool_wear_min": ("L'usure de l'outil", "Usure de l'outil", 0),
    "temp_diff_k": ("L'écart thermique", "Écart thermique", 1),
    "power_w": ("La puissance", "Puissance", 0),
    "strain_min_nm": ("La contrainte d'usure", "Contrainte d'usure", 0),
}
DERIVED: dict[str, tuple[str, str, int]] = {
    "temp_diff_k": ("Écart thermique", "K", 1),
    "power_w": ("Puissance", "W", 0),
    "strain_min_nm": ("Contrainte d'usure", "min·Nm", 0),
}

SCENARIOS: dict[str, dict[str, Any]] = {
    "Machine saine": {
        "product_type": "M",
        "air_temperature_k": 298.9,
        "process_temperature_k": 309.1,
        "rotational_speed_rpm": 1500,
        "torque_nm": 40.0,
        "tool_wear_min": 50,
    },
    "Surcharge outil": {
        "product_type": "L",
        "air_temperature_k": 300.0,
        "process_temperature_k": 310.0,
        "rotational_speed_rpm": 1400,
        "torque_nm": 65.0,
        "tool_wear_min": 210,
    },
    "Surchauffe": {
        "product_type": "M",
        "air_temperature_k": 302.5,
        "process_temperature_k": 310.5,
        "rotational_speed_rpm": 1300,
        "torque_nm": 55.0,
        "tool_wear_min": 100,
    },
    "Usure ≥ 200 min": {
        "product_type": "M",
        "air_temperature_k": 298.0,
        "process_temperature_k": 308.5,
        "rotational_speed_rpm": 1550,
        "torque_nm": 35.0,
        "tool_wear_min": 215,
    },
}


# --- Format français (affichage uniquement) ---------------------------------------------

THOUSANDS = " "  # espace fine insécable
NBSP = " "
MINUS = "−"


def fmt_number(value: float, decimals: int = 0) -> str:
    """1234.5 → « 1 234,5 » (espace des milliers, virgule décimale, vrai signe moins)."""
    text = f"{abs(value):,.{decimals}f}".replace(",", THOUSANDS).replace(".", ",")
    is_zero = not text.strip("0," + THOUSANDS)
    return text if value >= 0 or is_zero else f"{MINUS}{text}"


def fmt_smart(value: float) -> str:
    """Entier sans décimale, sinon une décimale : 120.0 → « 120 », −2.5 → « −2,5 »."""
    return fmt_number(value, 0 if float(value).is_integer() else 1)


def fmt_pct(fraction: float) -> str:
    """0.163 → « 16 % »."""
    return f"{fmt_number(fraction * 100, 0)}{NBSP}%"


def fmt_risk(risk: float) -> str:
    """Risque affiché : « < 1 % » sous 0,01, « > 99 % » au-delà de 0,99, sinon à l'unité.

    Évite d'afficher « 0 % » (jamais certain) ou « 100 % » (jamais certain non plus).
    """
    if risk < 0.01:
        return f"<{NBSP}1{NBSP}%"
    if risk > 0.99:
        return f">{NBSP}99{NBSP}%"
    return fmt_pct(risk)


def fmt_value(value: float | str, unit: str | None, decimals: int) -> str:
    if isinstance(value, str):
        return value
    text = fmt_number(value, decimals)
    return f"{text}{NBSP}{unit}" if unit else text


def format_temperature(kelvin: float) -> str:
    """298.0 → « 298,0 K (24,9 °C) » (conversion d'affichage ; l'API reçoit des kelvins)."""
    return f"{fmt_number(kelvin, 1)}{NBSP}K ({fmt_number(kelvin - 273.15, 1)}{NBSP}°C)"


def format_measure(field: str, value: float) -> str:
    if field.endswith("temperature_k"):
        return format_temperature(value)
    _, unit, *_, decimals = MEASURES[field]
    return fmt_value(value, unit, decimals)


def cause_sentence(factor: dict[str, Any]) -> str:
    """« ↑ L'usure de l'outil (215 min) augmente le risque. »"""
    subject, _, decimals = CAUSES.get(factor["feature"], (factor["feature"], "", 1))
    value = fmt_value(factor["value"], factor.get("unit"), decimals)
    if factor["direction"] == "vers_panne":
        return f"↑ {subject} ({value}) augmente le risque."
    return f"↓ {subject} ({value}) réduit le risque."


def cause_label(factor: dict[str, Any]) -> str:
    """« Usure de l'outil (215 min) » : libellé court, avec unité."""
    _, short, decimals = CAUSES.get(factor["feature"], ("", factor["feature"], 1))
    return f"{short} ({fmt_value(factor['value'], factor.get('unit'), decimals)})"


def main_cause(factors: list[dict[str, Any]]) -> str:
    """Cause principale « vers la panne » en phrase courte (lot)."""
    driver = next((f for f in factors if f["direction"] == "vers_panne"), None)
    return cause_label(driver) if driver else "—"


# --- Accès à l'API (sans affichage, testable) -------------------------------------------


class ApiError(Exception):
    """Erreur présentable à l'utilisateur (jamais de trace Python à l'écran)."""

    title = "Erreur du service d'analyse"
    kind = "neutral"


class ApiUnavailable(ApiError):
    title = "Service d'analyse indisponible"


class MeasureRejected(ApiError):
    title = "Mesure refusée"
    kind = "warning"


class ModelNotReady(ApiError):
    title = "Le modèle n'est pas chargé"


_OUT_OF_BOUNDS = re.compile(
    r"(?P<field>\w+)=(?P<value>-?[\d.]+) hors bornes \[(?P<low>-?[\d.]+), (?P<high>-?[\d.]+)\]"
)
_BAD_PRODUCT = re.compile(r"product_type='(?P<value>[^']*)' non autorisé")


def rephrase_rejection(message: str) -> str:
    """Message technique de l'API → phrase pour le technicien (inchangé s'il est inconnu)."""
    sentences = []
    for part in message.split("; "):
        if match := _OUT_OF_BOUNDS.search(part):
            field = match["field"]
            label, unit = MEASURES[field][:2] if field in MEASURES else (field, "")
            value, low, high = (float(match[k]) for k in ("value", "low", "high"))
            sentences.append(
                f"{label} : {fmt_smart(value)}{NBSP}{unit} est hors de la plage plausible "
                f"({fmt_smart(low)} à {fmt_smart(high)}{NBSP}{unit}). Vérifiez le capteur."
            )
        elif match := _BAD_PRODUCT.search(part):
            sentences.append(
                f"Type de produit « {match['value']} » inconnu : choisissez L, M ou H."
            )
        else:
            sentences.append(part)
    return " ".join(sentences)


def _detail_message(response: httpx.Response) -> str:
    try:
        detail = response.json().get("detail")
    except ValueError:
        return response.text[:200] or "réponse vide"
    if isinstance(detail, dict):
        return str(detail.get("message", detail))
    if isinstance(detail, list):  # erreurs de type renvoyées directement par FastAPI
        return "; ".join(str(e.get("msg", e)) for e in detail if isinstance(e, dict))
    return str(detail)


def call_api(method: str, path: str, payload: Any = None) -> dict[str, Any]:
    """Appelle l'API et renvoie le JSON, ou lève une ApiError au message clair."""
    try:
        with httpx.Client(base_url=API_URL, timeout=TIMEOUT_S) as client:
            response = client.request(method, path, json=payload)
    except httpx.HTTPError as exc:
        raise ApiUnavailable("Vérifiez qu'il est démarré, puis réessayez.") from exc

    if response.status_code == 422:
        raise MeasureRejected(rephrase_rejection(_detail_message(response)))
    if response.status_code == 503:
        raise ModelNotReady("Réessayez dans quelques instants.")
    if response.status_code == 413:
        raise MeasureRejected(f"Lot trop volumineux : {_detail_message(response)}")
    if response.status_code >= 400:
        raise ApiError(f"Réponse inattendue (code {response.status_code}). Réessayez plus tard.")
    try:
        body: dict[str, Any] = response.json()
    except ValueError as exc:
        raise ApiError("Réponse illisible du service.") from exc
    return body


@st.cache_data(ttl=300, show_spinner=False)
def fetch_model_info() -> dict[str, Any]:
    """/model/info, mis en cache 5 min : le modèle et les seuils sont figés.

    Une ApiError n'est jamais mise en cache (Streamlit ne mémorise que les résultats).
    /predict, lui, n'est jamais mis en cache.
    """
    return call_api("GET", "/model/info")


def parse_batch_csv(content: bytes) -> list[dict[str, Any]]:
    """CSV (colonnes de l'API) → liste de mesures, ou MeasureRejected au message clair."""
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise MeasureRejected("Fichier CSV illisible.") from exc
    missing = [c for c in FIELDS if c not in df.columns]
    if missing:
        raise MeasureRejected(
            f"Colonnes manquantes : {', '.join(missing)}. Attendues : {', '.join(FIELDS)}."
        )
    if df.empty:
        raise MeasureRejected("Le fichier ne contient aucune ligne.")
    if len(df) > MAX_BATCH:
        raise MeasureRejected(
            f"{len(df)} lignes : maximum {MAX_BATCH} par lot. Découpez le fichier."
        )
    df = df[list(FIELDS)]
    if df.isna().any().any():
        rows = (df.isna().any(axis=1).to_numpy().nonzero()[0] + 1).tolist()
        raise MeasureRejected(f"Valeurs manquantes aux lignes {rows[:10]}.")
    df["product_type"] = df["product_type"].astype(str).str.strip()
    return [{str(key): value for key, value in row.items()} for row in df.to_dict("records")]


def example_csv() -> bytes:
    """CSV d'exemple : 5 machines réalistes (les 4 scénarios + une machine de type H)."""
    rows = [
        *SCENARIOS.values(),
        {
            "product_type": "H",
            "air_temperature_k": 297.4,
            "process_temperature_k": 308.2,
            "rotational_speed_rpm": 1720,
            "torque_nm": 31.5,
            "tool_wear_min": 140,
        },
    ]
    return pd.DataFrame(rows, columns=list(FIELDS)).to_csv(index=False).encode("utf-8")


def batch_results(items: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> pd.DataFrame:
    """Mesures + verdict de l'API, triées « à traiter d'abord » (aucune valeur recalculée)."""
    result = pd.DataFrame(items)
    result["decision"] = [p["decision"] for p in predictions]
    result["risk"] = [p["failure_probability"] for p in predictions]
    result["recommended_action"] = [p["recommended_action"] or "" for p in predictions]
    result["main_cause"] = [main_cause(p["top_factors"]) for p in predictions]
    priority = {state: rank for rank, state in enumerate(STATE_ORDER)}
    result["_priority"] = result["decision"].map(priority)
    result = result.sort_values(["_priority", "risk"], ascending=[True, False], kind="stable")
    return result.drop(columns="_priority").reset_index(drop=True)


def state_text(decision: str) -> str:
    state = STATES[decision]
    return f"{state['icon']} {state['label']}"


def _plural(count: int, singular: str, plural: str) -> str:
    return f"{fmt_number(count)} {singular if count == 1 else plural}"


def reliability_lines(results: list[dict[str, Any]]) -> list[str]:
    """Résultats du test (lus dans /model/info) → phrases simples, valeurs calculées.

    Les alertes de l'hybride = alertes du modèle OU remplacement préventif (règle B1).
    Remplacements planifiés = fausses alertes de B1 ; fausses alertes urgentes du modèle =
    fausses alertes de l'hybride − celles de B1 (les alertes de B1 sont incluses dans l'hybride).
    """
    by_prefix = {str(r["approach"]).split()[0]: r for r in results}
    hybrid, preventive, baseline = (by_prefix.get(k) for k in ("Hybride", "B1", "B0"))
    if hybrid is None:
        return []
    failures = hybrid["detected"] + hybrid["missed"]
    first = f"**{hybrid['detected']} pannes sur {failures} anticipées.**"
    if preventive is not None:
        planned = int(preventive["false_alarms"])
        urgent = int(hybrid["false_alarms"]) - planned
        if urgent == 0:
            urgent_text = "Aucune fausse alerte urgente"
        elif urgent == 1:
            urgent_text = "1 seule fausse alerte urgente"
        else:
            urgent_text = _plural(urgent, "fausse alerte urgente", "fausses alertes urgentes")
        planned_text = _plural(
            planned, "remplacement d'outil planifié", "remplacements d'outils planifiés"
        )
        first += f" {urgent_text}, et {planned_text} au titre du préventif."
    else:
        first += f" {_plural(int(hybrid['false_alarms']), 'fausse alerte', 'fausses alertes')}."
    lines = [first]
    if baseline is not None and baseline["total_cost_eur"]:
        saving = 1 - hybrid["total_cost_eur"] / baseline["total_cost_eur"]
        lines.append(
            f"Coût estimé : **{fmt_number(hybrid['total_cost_eur'])}{NBSP}€** contre "
            f"{fmt_number(baseline['total_cost_eur'])}{NBSP}€ sans détection "
            f"({MINUS}{fmt_number(saving * 100)}{NBSP}%)."
        )
    lines.append("Résultats mesurés sur des cycles jamais vus pendant la mise au point.")
    return lines


# --- Composants d'affichage -------------------------------------------------------------


def _safe(text: str) -> str:
    """Échappe < > & pour l'insertion dans du HTML (les apostrophes restent lisibles)."""
    return html.escape(text, quote=False)


def card(title: str, text: str, color: str, background: str, icon: str = "") -> None:
    """Carte à fond pâle et bordure gauche épaisse (titre en gros, texte dessous)."""
    st.markdown(
        f'<div style="background:{background};border-left:8px solid {color};'
        f'border-radius:6px;padding:14px 18px;margin:4px 0 12px 0;">'
        f'<div style="font-size:1.55rem;font-weight:700;color:{color};">'
        f"{icon} {_safe(title)}</div>"
        f'<div style="font-size:1.05rem;color:#202124;margin-top:4px;">{_safe(text)}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def error_card(error: ApiError) -> None:
    if error.kind == "warning":
        warning = STATES["MAINTENANCE_PREVENTIVE"]
        card(error.title, str(error), warning["color"], warning["background"], "⚠️")
    else:
        card(error.title, str(error), NEUTRAL, NEUTRAL_BG, "⚪")


def status_card(decision: str) -> None:
    state = STATES[decision]
    card(state["label"], state["sentence"], state["color"], state["background"], state["icon"])


def action_box(action: str, decision: str) -> None:
    color = STATES[decision]["color"]
    st.markdown(
        f'<div style="border:2px solid {color};border-radius:6px;padding:10px 16px;'
        f'margin-bottom:12px;font-size:1.05rem;"><b>À faire :</b> {_safe(action)}</div>',
        unsafe_allow_html=True,
    )


def risk_gauge(risk: float, threshold: float, decision: str) -> None:
    color = STATES[decision]["color"]
    st.markdown(f"**Risque de panne : {fmt_risk(risk)}**")
    scale = alt.Scale(domain=[0, 100])
    track = (
        alt.Chart(pd.DataFrame({"start": [0], "end": [100]}))
        .mark_bar(color="#E8EAED", height=28, cornerRadius=4)
        .encode(x=alt.X("start:Q", scale=scale, axis=None), x2="end:Q")
    )
    fill = (
        alt.Chart(pd.DataFrame({"start": [0], "end": [risk * 100]}))
        .mark_bar(color=color, height=28, cornerRadius=4)
        .encode(x=alt.X("start:Q", scale=scale, axis=None), x2="end:Q")
    )
    marker = pd.DataFrame(
        {"at": [threshold * 100], "text": [f"Seuil d'alerte {fmt_pct(threshold)}"]}
    )
    rule = alt.Chart(marker).mark_rule(color="#202124", strokeWidth=2).encode(x="at:Q")
    label = (
        alt.Chart(marker)
        .mark_text(align="left", dx=5, dy=-24, color="#202124", fontSize=12)
        .encode(x="at:Q", text="text:N")
    )
    chart = alt.layer(track, fill, rule, label).properties(height=75)
    st.altair_chart(chart.configure_view(stroke=None), width="stretch")


TOWARDS_FAILURE, TOWARDS_NORMAL = "vers_panne", "vers_normal"


def causes_chart(factors: list[dict[str, Any]]) -> alt.LayerChart:
    """Barres divergentes autour d'un axe central à 0 : « vers le normal » en vert à gauche,
    « vers la panne » en rouge à droite ; aucune graduation numérique.

    Le côté est imposé par la direction renvoyée par l'API (pas seulement par le signe).
    """
    rows = [
        {
            "cause": cause_label(f),
            "poids": abs(f["contribution"]) * (1 if f["direction"] == TOWARDS_FAILURE else -1),
            "sens": f["direction"],
        }
        for f in factors
    ]
    extent = max(abs(r["poids"]) for r in rows) * 1.15 or 1.0
    x_scale = alt.Scale(domain=[-extent, extent])
    bars = (
        alt.Chart(pd.DataFrame(rows))
        .mark_bar(height=22)
        .encode(
            x=alt.X("poids:Q", scale=x_scale, axis=None),
            y=alt.Y(
                "cause:N", sort=None, title=None, axis=alt.Axis(labelLimit=0, labelFontSize=13)
            ),
            color=alt.Color(
                "sens:N",
                scale=alt.Scale(
                    domain=[TOWARDS_FAILURE, TOWARDS_NORMAL],
                    range=[STATES["CRITIQUE"]["color"], STATES["OPTIMAL"]["color"]],
                ),
                legend=None,
            ),
        )
    )
    zero = pd.DataFrame({"x": [0.0]})
    axis = (
        alt.Chart(zero)
        .mark_rule(color="#202124", strokeWidth=1.5)
        .encode(x=alt.X("x:Q", scale=x_scale))
    )
    legend = pd.DataFrame(
        {"x": [0.0, 0.0], "text": ["← vers le normal", "vers la panne →"], "align": ["r", "l"]}
    )
    left = (
        alt.Chart(legend[legend["align"] == "r"])
        .mark_text(align="right", dx=-6, dy=-10, color=STATES["OPTIMAL"]["color"], fontSize=12)
        .encode(x=alt.X("x:Q", scale=x_scale), y=alt.value(0), text="text:N")
    )
    right = (
        alt.Chart(legend[legend["align"] == "l"])
        .mark_text(align="left", dx=6, dy=-10, color=STATES["CRITIQUE"]["color"], fontSize=12)
        .encode(x=alt.X("x:Q", scale=x_scale), y=alt.value(0), text="text:N")
    )
    return alt.LayerChart(
        layer=[bars, axis, left, right],
        height=36 * len(rows) + 10,
        padding={"top": 22, "left": 5, "right": 5, "bottom": 5},
    )


def causes_section(factors: list[dict[str, Any]]) -> None:
    st.subheader("Pourquoi ce verdict", anchor=False)
    if not factors:
        st.markdown("Aucune cause marquante.")
        return
    st.markdown("\n".join(f"- {cause_sentence(f)}" for f in factors))
    st.altair_chart(causes_chart(factors).configure_view(stroke=None), width="stretch")


def technical_details(result: dict[str, Any]) -> None:
    with st.expander("Détails techniques", expanded=False):
        columns = st.columns(3)
        for column, (name, (label, unit, decimals)) in zip(columns, DERIVED.items(), strict=True):
            column.metric(label, fmt_value(result["derived"][name], unit, decimals))
        version = result["model_version"]
        st.caption(
            f"Modèle {version['model_sha256'][:12]} · seuils {version['thresholds_sha256'][:12]} "
            f"· gel {version['git_commit'][:8]}"
        )


# --- Onglet Machine ---------------------------------------------------------------------


def _apply_scenario(name: str) -> None:
    for field, value in SCENARIOS[name].items():
        st.session_state[field] = value
    st.session_state["scenario"] = name


def _mark_custom() -> None:
    st.session_state["scenario"] = CUSTOM


def _init_state() -> None:
    for field, value in SCENARIOS["Machine saine"].items():
        st.session_state.setdefault(field, value)
    st.session_state.setdefault("scenario", "Machine saine")


def _preventive_hint() -> None:
    """Repère sous l'usure : seuil préventif FIGÉ lu dans /model/info (jamais écrit en dur)."""
    try:
        wear_min = fetch_model_info()["preventive_tool_wear_min"]
    except ApiError:
        return  # pas de repère plutôt qu'une valeur inventée
    st.caption(f"Remplacement préventif à partir de {fmt_smart(wear_min)}{NBSP}min")


def _measure_input(field: str) -> None:
    if field == "product_type":
        st.radio(
            "Type de produit", PRODUCT_TYPES, key=field, horizontal=True, on_change=_mark_custom
        )
        return
    label, _, low, high, step, decimals = MEASURES[field]
    st.markdown(f"{label} : **{format_measure(field, st.session_state[field])}**")
    st.slider(
        label,
        min_value=low,
        max_value=high,
        step=step,
        format="%d" if decimals == 0 else "%.1f",
        key=field,
        on_change=_mark_custom,
        label_visibility="collapsed",
    )


def machine_inputs() -> None:
    st.subheader("Scénarios", anchor=False)
    active = st.session_state["scenario"]
    columns = st.columns(2)
    for i, name in enumerate(SCENARIOS):
        columns[i % 2].button(
            name,
            on_click=_apply_scenario,
            args=(name,),
            type="primary" if name == active else "secondary",
            width="stretch",
        )
    st.caption(f"Scénario : {active}")

    for group, fields in GROUPS.items():
        st.subheader(group, anchor=False)
        for field in fields:
            _measure_input(field)
            if field == "tool_wear_min":
                _preventive_hint()


def machine_verdict() -> None:
    payload = {field: st.session_state[field] for field in FIELDS}
    try:
        with st.spinner("Analyse en cours…"):
            result = call_api("POST", "/predict", payload)
    except ApiError as error:
        error_card(error)
        return

    decision = result["decision"]
    status_card(decision)
    if result.get("recommended_action"):
        action_box(result["recommended_action"], decision)
    risk_gauge(result["failure_probability"], result["threshold"], decision)
    causes_section(result["top_factors"])
    technical_details(result)


def machine_tab() -> None:
    left, right = st.columns([2, 3], gap="large")
    with left:
        machine_inputs()
    with right:
        machine_verdict()


# --- Onglet Lot -------------------------------------------------------------------------


def kpi_cards(result: pd.DataFrame) -> None:
    for column, decision in zip(st.columns(3), STATE_ORDER, strict=True):
        state = STATES[decision]
        count = int((result["decision"] == decision).sum())
        with column:
            card(
                f"{count} machine{'s' if count > 1 else ''}",
                state["label"],
                state["color"],
                state["background"],
                state["icon"],
            )


def batch_tab() -> None:
    st.markdown(
        "Importez un fichier CSV avec une ligne par machine et les colonnes "
        f"`{', '.join(FIELDS)}` (au plus {MAX_BATCH} lignes, températures en kelvins)."
    )
    st.download_button(
        "Télécharger un fichier d'exemple",
        example_csv(),
        file_name="exemple_machines.csv",
        mime="text/csv",
    )
    uploaded = st.file_uploader("Fichier de mesures", type="csv")
    if uploaded is None:
        return
    try:
        items = parse_batch_csv(uploaded.getvalue())
        with st.spinner("Analyse en cours…"):
            response = call_api("POST", "/predict/batch", {"items": items})
    except ApiError as error:
        error_card(error)
        return

    result = batch_results(items, response["predictions"])
    kpi_cards(result)

    labels = {decision: state_text(decision) for decision in STATE_ORDER}
    chosen = st.multiselect(
        "Filtrer par état", list(labels.values()), default=list(labels.values())
    )
    shown = result[result["decision"].map(labels).isin(chosen)]
    table = pd.DataFrame(
        {
            "État": shown["decision"].map(labels),
            "Risque": shown["risk"].map(fmt_risk),
            "Action recommandée": shown["recommended_action"],
            "Cause principale": shown["main_cause"],
            "Type": shown["product_type"],
            "Usure (min)": shown["tool_wear_min"].map(lambda v: fmt_number(v, 0)),
            "Couple (Nm)": shown["torque_nm"].map(lambda v: fmt_number(v, 1)),
        }
    )
    st.dataframe(table, width="stretch", hide_index=True)
    st.download_button(
        "Télécharger le résultat (CSV)",
        result.to_csv(index=False).encode("utf-8"),
        file_name="verdicts_machines.csv",
        mime="text/csv",
    )


# --- Onglet Le modèle -------------------------------------------------------------------


def model_tab() -> None:
    try:
        info = fetch_model_info()
    except ApiError as error:
        error_card(error)
        return

    st.subheader("Comment lire le résultat", anchor=False)
    for decision in ("OPTIMAL", "MAINTENANCE_PREVENTIVE", "CRITIQUE"):
        state = STATES[decision]
        st.markdown(
            f"<span style='color:{state['color']};font-weight:700'>{state['icon']} "
            f"{state['label']}</span> — {state['sentence']}",
            unsafe_allow_html=True,
        )
    st.markdown(
        f"Une alerte est donnée dès que le risque atteint **{fmt_pct(info['threshold'])}** ; "
        "l'outil est remplacé à titre préventif à partir de "
        f"**{fmt_smart(info['preventive_tool_wear_min'])}{NBSP}min** d'usure."
    )

    st.subheader("Fiabilité mesurée", anchor=False)
    lines = reliability_lines(info.get("test_results") or [])
    st.markdown("\n\n".join(lines) if lines else "L'évaluation finale n'a pas encore été réalisée.")
    st.caption(
        f"Coûts estimés : {fmt_number(info['cost_missed_failure_eur'])}{NBSP}€ par panne non "
        f"anticipée, {fmt_number(info['cost_false_alarm_eur'])}{NBSP}€ par fausse alerte. "
        "Résultats obtenus sur des données synthétiques (AI4I 2020)."
    )

    st.subheader("Version", anchor=False)
    version = info["model_version"]
    st.caption(
        f"Modèle {version['model_sha256']} · seuils {version['thresholds_sha256']} "
        f"· gel {version['git_commit']}"
    )


# --- Application ------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(layout="wide", page_title="Surveillance machine", page_icon="🛠️")
    st.header("🛠️ Surveillance machine — risque de panne", anchor=False)
    st.caption("Saisissez les mesures de la machine ; le verdict se met à jour automatiquement.")
    _init_state()
    machine, batch, model = st.tabs(["Machine", "Lot", "Le modèle"])
    for tab, render in ((machine, machine_tab), (batch, batch_tab), (model, model_tab)):
        with tab:
            try:
                render()
            except Exception:  # dernier filet : jamais de trace Python à l'écran
                card(
                    "Erreur inattendue",
                    "Rechargez la page ; si le problème persiste, contactez le support.",
                    NEUTRAL,
                    NEUTRAL_BG,
                    "⚪",
                )


if __name__ == "__main__":
    main()
