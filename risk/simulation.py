"""Génération d'un historique de P&L synthétique par desk.

Le portefeuille est fictif. Les P&L suivent un GARCH(1,1) commun à tous les
desks, avec des innovations Student t (queues épaisses) et un épisode de
stress de volatilité injecté pour rendre le backtesting instructif.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Desk:
    name: str
    scale: float  # volatilité de long terme du desk, en M€
    rho: float  # corrélation au facteur commun
    drift: float  # revenu moyen quotidien, en M€
    var_limit: float  # limite de VaR 99 % 1 j, en M€


DESKS: tuple[Desk, ...] = (
    Desk("Taux", 1.55, 0.55, 0.05, 5.0),
    Desk("Crédit", 1.10, 0.65, 0.03, 3.5),
    Desk("Actions", 2.30, 0.75, 0.06, 7.0),
    Desk("Change", 0.95, 0.30, 0.02, 3.0),
    Desk("Matières premières", 0.85, 0.40, 0.02, 2.5),
)

PORTFOLIO_VAR_LIMIT = 18.0  # M€


def _student_t(rng: np.random.Generator, nu: float, size) -> np.ndarray:
    """Tirages Student t normalisés à variance unitaire."""
    return rng.standard_t(nu, size=size) * np.sqrt((nu - 2) / nu)


def simulate_pnl(
    n_days: int = 760,
    end: str = "2026-11-18",
    seed: int = 218,
    nu: float = 4.0,
    alpha: float = 0.08,
    beta: float = 0.90,
    stress_start: int | None = 150,
    stress_length: int = 22,
    stress_intensity: float = 7.0,
) -> pd.DataFrame:
    """Simule les P&L quotidiens (M€) de chaque desk.

    ``stress_start`` compte le nombre de jours avant la fin de l'historique
    où commence l'épisode de stress ; ``None`` le désactive.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=end, periods=n_days)
    omega = 1.0 - alpha - beta  # variance de long terme normalisée à 1

    common = _student_t(rng, nu, n_days)
    idio = _student_t(rng, 5.0, (n_days, len(DESKS)))

    h = np.empty(n_days)
    h_prev, z_prev = 1.0, 0.0
    for t in range(n_days):
        boost = 1.0
        if stress_start is not None:
            first = n_days - stress_start
            if first <= t < first + stress_length:
                boost = stress_intensity
        h_prev = omega * boost + alpha * h_prev * z_prev**2 + beta * h_prev
        h[t] = h_prev
        z_prev = common[t]
    sigma = np.sqrt(h)

    columns = {}
    for j, d in enumerate(DESKS):
        shock = d.rho * common + np.sqrt(1 - d.rho**2) * idio[:, j]
        columns[d.name] = d.scale * sigma * shock + d.drift

    return pd.DataFrame(columns, index=dates).rename_axis("date")
