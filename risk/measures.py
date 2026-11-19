"""Mesures de risque : VaR historique, paramétrique, EWMA et Expected Shortfall.

Convention : la VaR et l'ES sont exprimées en perte positive (M€).
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Literal

import numpy as np
import pandas as pd

Model = Literal["historical", "parametric", "ewma"]

MODEL_LABELS: dict[str, str] = {
    "historical": "Historique",
    "parametric": "Paramétrique normal",
    "ewma": "EWMA (λ = 0,94)",
}


def z_score(confidence: float) -> float:
    return NormalDist().inv_cdf(confidence)


def historical_var(pnl, confidence: float = 0.99) -> float:
    """Quantile empirique des P&L, avec interpolation linéaire."""
    return float(-np.quantile(np.asarray(pnl), 1 - confidence))


def parametric_var(pnl, confidence: float = 0.99) -> float:
    """VaR gaussienne centrée : z · σ."""
    return float(z_score(confidence) * np.std(np.asarray(pnl), ddof=1))


def expected_shortfall(pnl, confidence: float = 0.975) -> float:
    """Moyenne des ceil(n·(1-c)) pires P&L (estimateur FRTB discret)."""
    x = np.sort(np.asarray(pnl))
    k = max(1, math.ceil(round(len(x) * (1 - confidence), 9)))
    return float(-x[:k].mean())


def ewma_variance(pnl: pd.Series, lam: float = 0.94, window: int = 250) -> pd.Series:
    """Variance EWMA connue à la veille de chaque date.

    Initialisée avec la variance empirique des ``window`` premiers jours ;
    la valeur à la date t n'utilise que les P&L strictement antérieurs.
    """
    x = pnl.to_numpy()
    out = np.full(len(x), np.nan)
    v = float(np.var(x[:window], ddof=1))
    for t in range(window, len(x)):
        out[t] = v
        v = lam * v + (1 - lam) * x[t] ** 2
    return pd.Series(out, index=pnl.index)


def rolling_var(
    pnl: pd.Series,
    model: Model = "historical",
    confidence: float = 0.99,
    window: int = 250,
) -> pd.Series:
    """VaR de la veille pour chaque date (fenêtre [t-window, t)).

    Les ``window`` premières dates n'ont pas d'historique suffisant : NaN.
    """
    if model == "ewma":
        return z_score(confidence) * np.sqrt(ewma_variance(pnl, window=window))
    fn = historical_var if model == "historical" else parametric_var
    x = pnl.to_numpy()
    out = np.full(len(x), np.nan)
    for t in range(window, len(x)):
        out[t] = fn(x[t - window : t], confidence)
    return pd.Series(out, index=pnl.index)


def diversification(desk_pnl: pd.DataFrame, confidence: float = 0.99) -> tuple[pd.Series, float]:
    """VaR historique autonome par desk et VaR du portefeuille agrégé.

    L'effet de diversification vaut ``total - standalone.sum()`` (négatif).
    """
    standalone = desk_pnl.apply(lambda s: historical_var(s, confidence))
    total = historical_var(desk_pnl.sum(axis=1), confidence)
    return standalone, total
