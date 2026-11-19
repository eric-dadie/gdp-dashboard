"""Stress tests par sensibilités : scénarios historiques, hypothétiques et inversés."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class Factor:
    key: str
    label: str
    unit: str
    lo: float
    hi: float
    step: float


FACTORS: tuple[Factor, ...] = (
    Factor("eq", "Actions (Euro Stoxx)", "%", -50, 30, 1),
    Factor("ir", "Taux EUR, parallèle", "pb", -200, 200, 5),
    Factor("cs", "Spreads crédit IG", "pb", -50, 400, 5),
    Factor("fx", "EUR/USD", "%", -20, 20, 0.5),
    Factor("vol", "Volatilité implicite", "pts", -10, 80, 1),
    Factor("oil", "Pétrole (Brent)", "%", -70, 70, 1),
)

# Sensibilités en M€ par unité de choc. "gamma" est en M€ par %² de choc actions.
SENSITIVITIES: dict[str, dict[str, float]] = {
    "Taux": {"ir": -0.045},
    "Crédit": {"ir": -0.010, "cs": -0.055, "eq": 0.05},
    "Actions": {"eq": 1.10, "gamma": -0.018, "vol": 0.12, "ir": -0.002},
    "Change": {"fx": 0.65, "ir": 0.003, "vol": 0.02},
    "Matières premières": {"oil": 0.22, "eq": 0.05, "fx": -0.05, "ir": 0.001},
}

# DV01 en k€ pour +1 pb, par desk et point de courbe EUR.
DV01 = pd.DataFrame(
    {
        "1A": [-2, -1, 0, 1, 0],
        "2A": [-6, -2, 0, 2, 1],
        "5A": [-15, -4, -1, 0, 0],
        "10A": [-18, -3, -1, 0, 0],
        "30A": [-4, 0, 0, 0, 0],
    },
    index=list(SENSITIVITIES),
)


@dataclass(frozen=True)
class Scenario:
    name: str
    period: str
    shocks: dict[str, float] = field(default_factory=dict)


# Chocs calibrés sur la pire fenêtre de 10 jours ouvrés de chaque épisode (ordres de grandeur).
SCENARIOS: tuple[Scenario, ...] = (
    Scenario("Faillite de Lehman", "sept.–oct. 2008", {"eq": -22, "ir": -60, "cs": 150, "fx": -6, "vol": 35, "oil": -25}),
    Scenario("Covid-19", "mars 2020", {"eq": -28, "ir": -45, "cs": 120, "fx": -2, "vol": 55, "oil": -40}),
    Scenario("Choc d'inflation", "2022", {"eq": -8, "ir": 90, "cs": 40, "fx": -4, "vol": 8, "oil": 25}),
    Scenario("Référendum Brexit", "juin 2016", {"eq": -9, "ir": -25, "cs": 30, "fx": -3, "vol": 12, "oil": -6}),
    Scenario("Stagflation", "hypothétique", {"eq": -15, "ir": 120, "cs": 110, "fx": -8, "vol": 20, "oil": 45}),
)

STRESS_LIMIT = 40.0  # perte maximale tolérée en stress, M€
ALLOCATED_CAPITAL = 250.0  # M€

FACTOR_BREAKDOWN = ["Actions Δ", "Gamma actions", "Taux", "Crédit", "Change", "Volatilité", "Matières premières"]
_KEY_TO_LABEL = {"ir": "Taux", "cs": "Crédit", "fx": "Change", "vol": "Volatilité", "oil": "Matières premières"}


def stress_pnl(shocks: dict[str, float]) -> pd.DataFrame:
    """P&L stressé (M€), en matrice desk × composante de risque."""
    s = {f.key: float(shocks.get(f.key, 0.0)) for f in FACTORS}
    rows = {}
    for desk, sens in SENSITIVITIES.items():
        row = dict.fromkeys(FACTOR_BREAKDOWN, 0.0)
        row["Actions Δ"] = sens.get("eq", 0.0) * s["eq"]
        row["Gamma actions"] = sens.get("gamma", 0.0) * s["eq"] ** 2
        for key, label in _KEY_TO_LABEL.items():
            row[label] = sens.get(key, 0.0) * s[key]
        rows[desk] = row
    return pd.DataFrame.from_dict(rows, orient="index")[FACTOR_BREAKDOWN]


def total_stress_pnl(shocks: dict[str, float]) -> float:
    return float(stress_pnl(shocks).to_numpy().sum())


def breaking_multiplier(shocks: dict[str, float], limit: float = STRESS_LIMIT, k_max: float = 8.0) -> float | None:
    """Stress inversé : plus petit k tel que le scénario k·choc atteigne -limit.

    Renvoie None si la limite n'est pas atteinte pour k <= k_max.
    """
    def gap(k: float) -> float:
        return total_stress_pnl({f: v * k for f, v in shocks.items()}) + limit

    if gap(k_max) > 0:
        return None
    lo, hi = 0.0, k_max
    for _ in range(60):
        mid = (lo + hi) / 2
        if gap(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
